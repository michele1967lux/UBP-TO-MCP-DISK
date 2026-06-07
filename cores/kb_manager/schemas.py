"""Schema registry for structured KB items.

Stores domain schemas (menu_item, drink_item, etc.) in Redis,
validates item data against them, and builds text representations
for embedding.

v1.0.0: Initial release (KB-MANAGER)
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SchemaRegistry:
    """Validates structured items against domain schemas."""

    def __init__(self, redis_client, config: Dict[str, Any]):
        self.redis = redis_client
        self.prefix = config.get("redis", {}).get("keys", {}).get(
            "schema_prefix", "ubp:kb_mgr:schema:"
        )
        self._cache: Dict[str, Dict] = {}

    async def register(self, schema_name: str, schema_def: Dict[str, Any]) -> None:
        """Store schema in Redis. Called by apply-preset or admin API."""
        if self.redis:
            await self.redis.set(
                f"{self.prefix}{schema_name}",
                json.dumps(schema_def, ensure_ascii=False),
            )
        self._cache[schema_name] = schema_def
        logger.info("[KB-MGR] Schema registered: %s", schema_name)

    async def get(self, schema_name: str) -> Optional[Dict[str, Any]]:
        """Get schema by name. Cache-first, Redis fallback."""
        if schema_name in self._cache:
            return self._cache[schema_name]
        if self.redis:
            raw = await self.redis.get(f"{self.prefix}{schema_name}")
            if raw:
                schema = json.loads(raw)
                self._cache[schema_name] = schema
                return schema
        return None

    def validate(self, schema_name: str, data: Dict[str, Any]) -> bool:
        """Validate data against schema. Raises ValueError on failure.

        Permissive: if schema not in cache, skip validation.
        """
        schema = self._cache.get(schema_name)
        if not schema:
            return True
        fields = schema.get("fields", {})
        for field_name, field_def in fields.items():
            if field_def.get("required") and field_name not in data:
                raise ValueError(f"Missing required field: {field_name}")
            if field_name in data and "enum" in field_def:
                if data[field_name] not in field_def["enum"]:
                    raise ValueError(
                        f"Invalid value for {field_name}: {data[field_name]}. "
                        f"Allowed: {field_def['enum']}"
                    )
        return True

    def build_text_repr(self, schema_name: str, data: Dict[str, Any]) -> str:
        """Build searchable text from structured data using schema template.

        Falls back to concatenation of all values if no template found.
        """
        schema = self._cache.get(schema_name, {})
        template = schema.get("text_template")
        if template:
            try:
                # Format list values as comma-separated strings before template
                fmt_data = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        fmt_data[k] = ", ".join(str(x) for x in v)
                    else:
                        fmt_data[k] = v
                return template.format(**fmt_data)
            except (KeyError, IndexError):
                logger.debug(
                    "[KB-MGR] Template format failed for schema=%s, falling back",
                    schema_name,
                )
        # Fallback: concatenate all values
        parts = []
        for k, v in data.items():
            if isinstance(v, list):
                parts.append(f"{k}: {', '.join(str(x) for x in v)}")
            elif v is not None:
                parts.append(str(v))
        return " | ".join(parts)

    async def load_schemas_from_config(
        self, item_schemas: Dict[str, Dict[str, Any]]
    ) -> int:
        """Bulk-load schemas from preset domain_config.item_schemas.

        Returns count of schemas loaded.
        """
        count = 0
        for name, schema_def in item_schemas.items():
            await self.register(name, schema_def)
            count += 1
        return count
