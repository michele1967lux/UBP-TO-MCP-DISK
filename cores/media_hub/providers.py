"""media_hub — Provider functions (business logic).

All operations are implemented as standalone async functions
for testability without DI container.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ubp.media_hub.providers")

try:
    from ubp_enterprise_hybrid.shared.media_engine import (
        MediaRequest, RenderPlan, MediaResult, AssetArtifact,
        ValidationReport, ChartData,
        plan_media, validate_plan, validate_request, validate_result,
        compute_cache_key, format_preview_markdown,
    )
except ImportError:
    from shared.media_engine import (
        MediaRequest, RenderPlan, MediaResult, AssetArtifact,
        ValidationReport, ChartData,
        plan_media, validate_plan, validate_request, validate_result,
        compute_cache_key, format_preview_markdown,
    )


# ── Key naming ──────────────────────────────────────────

REDIS_PREFIX = "ubp:media"


def _asset_key(asset_id: str) -> str:
    return f"{REDIS_PREFIX}:asset:{asset_id}"


def _cache_key(cache_hash: str) -> str:
    return f"{REDIS_PREFIX}:cache:{cache_hash}"


# ══════════════════════════════════════════════════════════
# Provider Functions
# ══════════════════════════════════════════════════════════

async def plan_media_provider(
    request_dict: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Plan media rendering from a request dict."""
    request = MediaRequest.from_dict(request_dict)

    # Validate request
    valid, issues = validate_request(request)
    if not valid:
        return {
            "status": "invalid",
            "issues": issues,
            "plan": None,
        }

    # Apply config defaults to policy
    cfg = config or {}
    policy_defaults = cfg.get("policy_defaults", {})
    if not request.policy:
        try:
            from ubp_enterprise_hybrid.shared.media_engine import MediaPolicy
        except ImportError:
            from shared.media_engine import MediaPolicy
        request.policy = MediaPolicy.from_dict(policy_defaults)

    # Create plan
    render_plan = plan_media(request)

    # Validate plan
    plan_status, plan_issues = validate_plan(render_plan)

    return {
        "status": "ok",
        "plan": render_plan.to_dict(),
        "validation": {
            "request_valid": True,
            "plan_status": plan_status,
            "plan_issues": plan_issues,
        },
    }


async def render_media_provider(
    plan_dict: Optional[Dict[str, Any]] = None,
    request_dict: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    redis: Any = None,
    chart_renderer: Any = None,
    diagram_renderer: Any = None,
    image_provider: Any = None,
    event_bus: Any = None,
) -> Dict[str, Any]:
    """Execute render: plan → MediaResult.

    If plan_dict is None but request_dict is provided, auto-plans first.
    """
    cfg = config or {}

    # Auto-plan if needed
    if plan_dict:
        plan = RenderPlan.from_dict(plan_dict)
    elif request_dict:
        plan_result = await plan_media_provider(request_dict, config)
        if plan_result["status"] != "ok":
            return {
                "status": "error",
                "error": "Planning failed",
                "details": plan_result.get("issues", []),
                "result": None,
            }
        plan = RenderPlan.from_dict(plan_result["plan"])
    else:
        return {
            "status": "error",
            "error": "Either plan or request required",
            "result": None,
        }

    # Check cache
    cache_enabled = cfg.get("cache", {}).get("enabled", True)
    if cache_enabled and plan.cache_key and redis:
        cached = await _check_cache(redis, plan.cache_key)
        if cached:
            logger.info("[MEDIA-HUB] Cache hit for %s", plan.cache_key)
            cached["cache_hit"] = True
            cached["status"] = "cached"
            await _publish(event_bus, "media.cache.hit", {
                "cache_key": plan.cache_key, "plan_id": plan.plan_id,
            })
            return {"status": "ok", "result": cached, "cache_hit": True}

    # Execute render steps
    import time
    t0 = time.monotonic()

    asset_id = f"mh_{uuid.uuid4().hex[:12]}"
    result = MediaResult(
        asset_id=asset_id,
        request_id=plan.request_id,
        plan_id=plan.plan_id,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    await _publish(event_bus, "media.render.started", {
        "asset_id": asset_id, "plan_id": plan.plan_id,
    })

    try:
        for step in plan.render_steps:
            step_result = await _execute_render_step(
                step=step,
                plan=plan,
                config=cfg,
                chart_renderer=chart_renderer,
                diagram_renderer=diagram_renderer,
                image_provider=image_provider,
                original_request=request_dict,
            )
            if step_result.get("error"):
                result.status = "error"
                result.error_message = step_result["error"]
                break
            if step_result.get("artifacts"):
                result.artifacts.extend([
                    AssetArtifact(**a) for a in step_result["artifacts"]
                ])
            result.engine_used = step.engine
            result.provider_used = plan.provider_chain[0] if plan.provider_chain else "local"

        if not result.error_message:
            result.status = "ok"

    except Exception as e:
        result.status = "error"
        result.error_message = str(e)
        logger.error("[MEDIA-HUB] Render error: %s", e)

    result.render_duration_ms = (time.monotonic() - t0) * 1000
    result.preview_markdown = format_preview_markdown(result)

    result_dict = result.to_dict()

    if result.is_success:
        await _publish(event_bus, "media.render.completed", {
            "asset_id": asset_id, "duration_ms": result.render_duration_ms,
        })
    else:
        await _publish(event_bus, "media.render.failed", {
            "asset_id": asset_id, "error": result.error_message,
        })

    # Store in cache
    if cache_enabled and plan.cache_key and redis and result.is_success:
        await _store_cache(redis, plan.cache_key, result_dict, cfg)

    # Store asset metadata
    if redis and result.is_success:
        await _store_asset_metadata(redis, asset_id, result_dict, cfg)

    return {
        "status": "ok" if result.is_success else "error",
        "result": result_dict,
        "cache_hit": False,
    }


async def get_media_provider(
    asset_id: str,
    config: Optional[Dict[str, Any]] = None,
    redis: Any = None,
) -> Dict[str, Any]:
    """Retrieve asset metadata by ID."""
    if redis:
        key = _asset_key(asset_id)
        raw = await redis.get(key)
        if raw:
            return {"status": "ok", "result": json.loads(raw)}

    return {"status": "not_found", "result": None}


async def validate_media_provider(
    result_dict: Dict[str, Any],
    request_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate a MediaResult against request."""
    result = MediaResult.from_dict(result_dict)
    request = MediaRequest.from_dict(request_dict)
    report = validate_result(result, request)
    return {"status": "ok", "report": report.to_dict()}


async def resolve_slots_provider(
    slots: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    redis: Any = None,
    chart_renderer: Any = None,
    diagram_renderer: Any = None,
    image_provider: Any = None,
    event_bus: Any = None,
) -> Dict[str, Any]:
    """Resolve MediaSlots from requirements_collector.

    Each slot describes a media element needed by a section.
    This function generates the media and returns resolved slots + results.
    """
    resolved = []
    results = []

    for slot_dict in slots:
        slot_id = slot_dict.get("slot_id", "")
        media_type = slot_dict.get("media_type", "chart")
        description = slot_dict.get("description", "")
        section_id = slot_dict.get("section_id")

        intent = _map_slot_type_to_intent(media_type)

        # Check if slot has data ready for render
        has_chart_data = bool(slot_dict.get("chart_data"))
        has_diagram_spec = bool(slot_dict.get("diagram_spec"))
        has_image_spec = bool(slot_dict.get("image_spec"))
        data_ready = has_chart_data or has_diagram_spec or has_image_spec

        # v6.8.x: External image URL passthrough — skip the renderer
        # which only supports generative media (chart/diagram/AI image).
        # ImageSpec has no source_url field, so from_dict() drops it → prompt="" → validation fail → None.
        if has_image_spec and not has_chart_data and not has_diagram_spec:
            src_url = (slot_dict.get("image_spec") or {}).get("source_url", "")
            if src_url.startswith(("https://", "http://")):
                ext_result = {
                    "asset_type": "external_image",
                    "url": src_url,
                    "thumbnail_url": (slot_dict["image_spec"].get("thumbnail") or src_url),
                    "alt_text": (slot_dict["image_spec"].get("alt_text") or description),
                    "origin": slot_dict["image_spec"].get("origin", "external"),
                    "slot_id": slot_id,
                }
                results.append(ext_result)
                resolved_slot = dict(slot_dict)
                resolved_slot["resolved"] = True
                resolved_slot["media_ref"] = {
                    "media_type": "image",
                    "source": src_url,
                    "title": description,
                    "section_id": section_id,
                    "metadata": {"asset_type": "external_image", "origin": "passthrough"},
                }
                resolved.append(resolved_slot)
                logger.info("[MEDIA-HUB] External image passthrough: slot=%s url=%s", slot_id, src_url[:100])
                continue

        if data_ready:
            # Build full MediaRequest and render
            request_dict = {
                "request_id": slot_id,
                "intent": intent,
                "purpose": "report",
                "section_id": section_id,
                "metadata": {"from_slot": True, "slot_description": description},
            }
            if has_chart_data:
                request_dict["chart_data"] = slot_dict["chart_data"]
                request_dict["chart_spec"] = slot_dict.get("chart_spec", {"chart_type": "bar"})
            if has_diagram_spec:
                request_dict["diagram_spec"] = slot_dict["diagram_spec"]
            if has_image_spec:
                request_dict["image_spec"] = slot_dict["image_spec"]

            render_result = await render_media_provider(
                request_dict=request_dict,
                config=config,
                redis=redis,
                chart_renderer=chart_renderer,
                diagram_renderer=diagram_renderer,
                image_provider=image_provider,
                event_bus=event_bus,
            )

            result_data = render_result.get("result", {})
            results.append(result_data)

            resolved_slot = dict(slot_dict)
            resolved_slot["resolved"] = render_result.get("status") == "ok"
            if result_data:
                resolved_slot["media_ref"] = {
                    "media_type": media_type,
                    "source": result_data.get("asset_id", ""),
                    "title": description,
                    "section_id": section_id,
                    "metadata": {"artifacts": result_data.get("artifacts", [])},
                }
        else:
            # Data deferred: mark slot as planned (resolved=True, pending render)
            resolved_slot = dict(slot_dict)
            resolved_slot["resolved"] = True
            resolved_slot["deferred"] = True
            resolved_slot["media_ref"] = {
                "media_type": media_type,
                "source": "",
                "title": description,
                "section_id": section_id,
                "metadata": {"status": "planned", "intent": intent},
            }
            results.append({"status": "planned", "slot_id": slot_id})

        resolved.append(resolved_slot)

    resolved_count = sum(1 for s in resolved if s.get("resolved"))
    await _publish(event_bus, "media.slot.resolved", {
        "resolved_count": resolved_count, "total_count": len(slots),
    })

    return {
        "status": "ok",
        "resolved": resolved,
        "results": results,
        "resolved_count": resolved_count,
        "total_count": len(slots),
    }


# ── Internal Helpers ─────────────────────────────────────


async def _publish(event_bus: Any, event_name: str, data: Dict[str, Any]) -> None:
    """Publish event if event_bus is available. Silent on failure."""
    if not event_bus:
        return
    try:
        if hasattr(event_bus, "publish"):
            await event_bus.publish(event_name, data, source="media_hub")
    except Exception:
        pass


def _map_slot_type_to_intent(media_type: str) -> str:
    """Map MediaSlot type to MediaIntent."""
    mapping = {
        "image": "image",
        "chart": "chart",
        "table": "table_visual",
        "diagram": "diagram",
        "file": "chart",  # fallback
        "link": "chart",  # fallback
        "video": "image",  # fallback
    }
    return mapping.get(media_type, "chart")


async def _execute_render_step(
    step: Any,
    plan: RenderPlan,
    config: Dict[str, Any],
    chart_renderer: Any = None,
    diagram_renderer: Any = None,
    image_provider: Any = None,
    original_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute a single render step with provider chain fallback.

    Uses registered renderers if available, otherwise returns placeholder.
    Provider chain: each renderer can be a list (local, cluster, external).
    """
    action = step.action
    params = step.parameters if hasattr(step, 'parameters') else {}
    output_dir = config.get("storage", {}).get("base_dir", "data/media_assets")

    # Enrich params with data from original request
    enriched_params = dict(params)
    if original_request:
        if action == "render_chart" and original_request.get("chart_data"):
            enriched_params["data"] = original_request["chart_data"]
        elif action == "render_diagram" and original_request.get("diagram_spec"):
            enriched_params["dsl"] = original_request["diagram_spec"].get("dsl", "")
        elif action == "generate_image" and original_request.get("image_spec"):
            enriched_params["prompt"] = original_request["image_spec"].get("prompt", "")

    # Build renderer chain for the action
    renderer_chain = _build_renderer_chain(
        action, chart_renderer, diagram_renderer, image_provider, plan
    )

    if renderer_chain:
        for tier_name, renderer in renderer_chain:
            result = await _try_renderer(renderer, action, enriched_params, output_dir, config)
            if result and result.get("status") == "ok":
                result["provider_tier"] = tier_name
                return result
            logger.warning("[MEDIA-HUB] Renderer %s failed for %s, trying next", tier_name, action)

        # All renderers failed — use placeholder as last resort
        logger.warning("[MEDIA-HUB] All renderers failed for %s — using placeholder", action)
        result = _placeholder_result(action.replace("render_", ""), enriched_params, config)
        result["fallback_used"] = True
        return result

    # No renderers registered
    if action == "convert_format":
        return {"artifacts": [], "error": None}

    return _placeholder_result(action.replace("render_", "").replace("generate_", ""), enriched_params, config)


def _build_renderer_chain(
    action: str,
    chart_renderer: Any,
    diagram_renderer: Any,
    image_provider: Any,
    plan: RenderPlan,
) -> List[tuple]:
    """Build ordered chain of (tier_name, renderer) for an action."""
    chain = []

    if action == "render_chart" and chart_renderer:
        if isinstance(chart_renderer, (list, tuple)):
            for i, r in enumerate(chart_renderer):
                chain.append((f"chart_{i}", r))
        else:
            chain.append(("local", chart_renderer))

    elif action == "render_diagram" and diagram_renderer:
        if isinstance(diagram_renderer, (list, tuple)):
            for i, r in enumerate(diagram_renderer):
                chain.append((f"diagram_{i}", r))
        else:
            chain.append(("local", diagram_renderer))

    elif action == "generate_image" and image_provider:
        if isinstance(image_provider, (list, tuple)):
            for i, r in enumerate(image_provider):
                chain.append((f"image_{i}", r))
        else:
            chain.append(("local", image_provider))

    elif action == "render_table" and chart_renderer:
        # Tables reuse chart renderer
        rend = chart_renderer if not isinstance(chart_renderer, (list, tuple)) else chart_renderer[0]
        chain.append(("local", rend))

    return chain


async def _try_renderer(
    renderer: Any,
    action: str,
    params: Dict[str, Any],
    output_dir: str,
    config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Try a single renderer, return result or None on failure."""
    try:
        # Build spec dict from step params (chart_type, title, etc.)
        spec = {k: v for k, v in params.items() if k not in ("data", "dsl", "prompt")}
        data = params.get("data")
        constraints = {"output_format": params.get("output_format", "png")}
        if params.get("dpi"):
            constraints["dpi"] = params["dpi"]

        import asyncio
        if asyncio.iscoroutinefunction(renderer.render):
            result = await renderer.render(spec, data, constraints, output_dir)
        else:
            result = renderer.render(spec, data, constraints, output_dir)

        return result
    except Exception as e:
        logger.error("[MEDIA-HUB] Renderer error: %s", e)
        return None


def _placeholder_result(
    asset_type: str,
    params: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a placeholder result when no real renderer is available."""
    base_dir = config.get("storage", {}).get("base_dir", "data/media_assets")
    fmt = params.get("output_format", "png")
    return {
        "artifacts": [{
            "uri": f"{base_dir}/placeholder_{asset_type}.{fmt}",
            "format": fmt,
            "size_bytes": 0,
            "width_px": 800,
            "height_px": 600,
            "content_hash": "",
        }],
        "error": None,
    }


async def _safe_call(fn: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """Safely call a renderer function."""
    try:
        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return await fn(**params)
        return fn(**params)
    except Exception as e:
        logger.error("[MEDIA-HUB] Renderer error: %s", e)
        return {"error": str(e), "artifacts": []}


async def _check_cache(redis: Any, cache_hash: str) -> Optional[Dict[str, Any]]:
    """Check if a result is cached."""
    try:
        key = _cache_key(cache_hash)
        raw = await redis.get(key)
        if raw:
            return json.loads(raw)
    except Exception as e:
        logger.warning("[MEDIA-HUB] Cache check error: %s", e)
    return None


async def _store_cache(
    redis: Any,
    cache_hash: str,
    result_dict: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Store result in cache."""
    try:
        key = _cache_key(cache_hash)
        ttl = config.get("cache", {}).get("ttl_seconds", 86400)
        await redis.setex(key, ttl, json.dumps(result_dict, default=str))
    except Exception as e:
        logger.debug("[MEDIA-HUB] Cache store error: %s", e)


async def _store_asset_metadata(
    redis: Any,
    asset_id: str,
    result_dict: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Store asset metadata in Redis."""
    try:
        key = _asset_key(asset_id)
        ttl = config.get("storage", {}).get("cleanup_after_days", 30) * 86400
        await redis.setex(key, ttl, json.dumps(result_dict, default=str))
    except Exception as e:
        logger.debug("[MEDIA-HUB] Asset store error: %s", e)
