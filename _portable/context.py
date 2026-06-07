from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PortableContext:
    client_id: str = "default"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    source: str = "portable"
    roles: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def extract_user_id(self) -> Optional[str]:
        return self.user_id

    def extract_client_id(self) -> str:
        return self.client_id or "default"

    @classmethod
    def normalize(cls, ctx: Any) -> PortableContext:
        if ctx is None:
            return cls()
        if isinstance(ctx, cls):
            return ctx
        if isinstance(ctx, dict):
            return cls(
                client_id=ctx.get("client_id", "default"),
                user_id=ctx.get("user_id"),
                session_id=ctx.get("session_id"),
                source=ctx.get("source", "dict"),
                roles=ctx.get("roles", []),
                metadata=ctx.get("metadata", {}),
            )
        if hasattr(ctx, "user") and ctx.user:
            user_id = getattr(ctx.user, "user_id", None)
            roles = getattr(ctx.user, "roles", [])
            client_id = getattr(ctx.user, "client_id", "default")
            if not isinstance(roles, (list, tuple)):
                roles = []
            return cls(
                client_id=str(client_id) if client_id else "default",
                user_id=str(user_id) if user_id else None,
                roles=list(roles),
                source="rest",
            )
        try:
            from ubp_enterprise_hybrid.modules.cores._shared.operation_context import OperationContext
            if isinstance(ctx, OperationContext):
                return cls(
                    client_id=ctx.client_id,
                    user_id=ctx.user_id,
                    session_id=ctx.session_id,
                    source=ctx.source,
                    roles=list(ctx.roles) if ctx.roles else [],
                    metadata=getattr(ctx, "metadata", {}),
                )
        except (ImportError, ModuleNotFoundError):
            pass
        return cls()
