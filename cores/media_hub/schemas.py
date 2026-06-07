"""media_hub — Pydantic request/response schemas for REST API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    # Stub for testing without pydantic
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    def Field(*args, **kwargs):
        return kwargs.get("default")


class PlanMediaRequest(BaseModel):
    """Request to plan media rendering."""
    request: Dict[str, Any] = Field(default_factory=dict)
    context: Optional[Dict[str, Any]] = None


class RenderMediaRequest(BaseModel):
    """Request to render media."""
    plan: Optional[Dict[str, Any]] = None
    request: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class GetMediaRequest(BaseModel):
    """Request to get a media asset."""
    asset_id: str


class ValidateMediaRequest(BaseModel):
    """Request to validate a media result."""
    result: Dict[str, Any] = Field(default_factory=dict)
    request: Dict[str, Any] = Field(default_factory=dict)


class ResolveSlotsRequest(BaseModel):
    """Request to resolve media slots."""
    slots: List[Dict[str, Any]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None


class MediaResponse(BaseModel):
    """Generic media response."""
    status: str = "ok"
    result: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cache_hit: bool = False
