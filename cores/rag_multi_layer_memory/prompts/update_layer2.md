# Update Layer 2 Long-term Memory

You are a long-term memory curator. Given the current Layer 2 and the new Layer 1 block that was just created, determine if Layer 2 needs updating.

## Rules
- Layer 2 is EXTREMELY minimal and conservative.
- Only update Layer 2 if there is information that is TRULY important and PERSISTENT.
- `critical_facts`: facts that should survive the entire conversation.
- `stable_preferences`: preferences that have been consistent across multiple turns.
- `core_rules`: strong rules the user insists on.
- `core_specifications`: persistent technical specifications.
- `dynamic_context`: only truly stable, cross-cutting domain keys.
- Do NOT add transient or situational information.
- If nothing warrants updating, return the SAME Layer 2 unchanged with `updated: false`.

## Input

**Current Layer 2:**
{current_layer2}

**New Layer 1 block:**
{new_layer1_block}

## Output Format

Return ONLY valid JSON:

```json
{
  "updated": true,
  "layer2": {
    "critical_facts": ["..."],
    "stable_preferences": {
      "explicit": ["..."],
      "inferred": ["..."]
    },
    "core_rules": ["..."],
    "core_specifications": ["..."],
    "dynamic_context": {},
    "last_updated_turn": 22
  }
}
```

Or if no update needed:

```json
{
  "updated": false,
  "layer2": null
}
```
