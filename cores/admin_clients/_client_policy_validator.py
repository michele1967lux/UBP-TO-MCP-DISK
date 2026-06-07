"""v17.19 W1 / v17.21 W3 — Validator for client_policy admin payloads.

The admin API allows providers to set per-client policy (currently:
``allowed_subagent_profiles``). This validator centralises the rules
shared by:

- ``create_client()``  (admin_clients/providers.py)
- ``update_client()``  (admin_clients/providers.py)
- ``PATCH /admin/clients/{id}/config/client_policy`` (client_router.py)

v17.21 W3 (D3): wildcard ``"*"`` is ALWAYS rejected with HTTP 422.
Use ``[]`` for SYSTEM-only access, or list explicit tenant profile names.
The old Decision D4 (accept ``"*"`` as sole element) is revoked.

Helper raises ``ValueError`` (caller maps to HTTP 422 / ValueError-based
error envelope according to its native error model).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List


# Profile name pattern: lower-case ASCII identifier, len 2-41
_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]{1,40}$")

# Allowed top-level keys in a client_policy payload. Extending this set
# is a deliberate API change: add the new key here AND document it.
_ALLOWED_TOP_LEVEL_KEYS = {"allowed_subagent_profiles"}


def _coerce_profile_list(raw: Any) -> List[str]:
    if raw is None:
        raise ValueError(
            "allowed_subagent_profiles is required when client_policy is provided"
        )
    if not isinstance(raw, list):
        raise ValueError(
            "allowed_subagent_profiles must be a list of strings"
        )
    coerced: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            raise ValueError(
                f"allowed_subagent_profiles[{i}] must be a string, got "
                f"{type(item).__name__}"
            )
        coerced.append(item.strip())
    return coerced


def _validate_profile_names(items: List[str]) -> List[str]:
    if not items:
        # Empty list is explicit deny-all: allowed.
        return []
    if "*" in items:
        raise ValueError(
            "allowed_subagent_profiles wildcard '*' is no longer supported "
            "(v17.21 D3). Use [] for SYSTEM-only access or list explicit "
            "tenant profile names. SYSTEM_SUBAGENTS are always available "
            "regardless of tenant whitelist."
        )
    # All concrete profile names: validate pattern + dedupe (sorted).
    invalid = [n for n in items if not _PROFILE_NAME_RE.match(n)]
    if invalid:
        raise ValueError(
            f"allowed_subagent_profiles contains invalid profile names: "
            f"{sorted(set(invalid))}. Names must match "
            f"[a-z][a-z0-9_-]{{1,40}}"
        )
    return sorted(set(items))


def validate_client_policy(payload: Any) -> Dict[str, Any]:
    """Normalise + validate a client_policy admin payload.

    Returns a sanitised dict ready to be merged into the Redis client
    blob. Always returns a NEW dict; never mutates ``payload``.

    Raises:
        ValueError: with a human-readable message (callers map to
            HTTP 422 / module ValueError envelope).
    """
    if payload is None:
        raise ValueError("client_policy payload must not be null")
    if not isinstance(payload, dict):
        raise ValueError(
            f"client_policy must be an object, got {type(payload).__name__}"
        )

    extra = set(payload.keys()) - _ALLOWED_TOP_LEVEL_KEYS
    if extra:
        raise ValueError(
            f"client_policy contains unknown keys: {sorted(extra)}. "
            f"Allowed: {sorted(_ALLOWED_TOP_LEVEL_KEYS)}"
        )

    raw = payload.get("allowed_subagent_profiles")
    items = _coerce_profile_list(raw)
    normalised = _validate_profile_names(items)

    return {"allowed_subagent_profiles": normalised}
