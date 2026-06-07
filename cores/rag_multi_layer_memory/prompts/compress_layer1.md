# Compress Layer 0 into Layer 1

You are a memory compression engine. Given the current Layer 0 snapshots (working memory), the existing Layer 1 blocks (mid-term memory), and the current Layer 2 (long-term memory), produce:

1. **ONE new Layer 1 block** that compresses the given snapshots into an evolved mid-term summary.
2. **Optionally**, an updated Layer 2 if the compression reveals truly important, persistent information.

## Rules for Layer 1 Block
- `turn_range` covers the turns of the input snapshots (e.g., "18-22").
- `focus` is the dominant topic/theme across the snapshots.
- `evolution_summary` briefly describes how the conversation evolved across these turns.
- `user_choices` captures significant decisions made by the user.
- `user_rules` captures rules or constraints the user has set.
- `specifications` captures technical or detailed specifications requested.
- `key_facts` are the ESSENTIAL facts that must be preserved.
- `preferences` consolidates explicit and inferred preferences.
- `dynamic_context` preserves domain-specific keys that remain relevant.
- `importance` is a score from 1-10 indicating how critical this block is.
- `last_updated_turn` is the highest turn number in the input.

## Rules for Layer 2 Update
- Layer 2 should ONLY be updated if the compression reveals truly important, persistent information.
- `critical_facts` are facts that remain relevant across the ENTIRE conversation.
- `stable_preferences` are preferences that have been consistent.
- `core_rules` are strong rules the user insists on.
- `core_specifications` are persistent technical specifications.
- `dynamic_context` holds only truly stable, cross-cutting keys.
- If nothing warrants updating Layer 2, return `layer2_updated: false`.

## Input

**Layer 0 snapshots to compress:**
{layer0_snapshots}

**Current Layer 1 (existing blocks):**
{current_layer1}

**Current Layer 2 (long-term memory):**
{current_layer2}

## Output Format

Return ONLY valid JSON:

```json
{
  "new_layer1_block": {
    "turn_range": "...",
    "focus": "...",
    "evolution_summary": "...",
    "user_choices": ["..."],
    "user_rules": ["..."],
    "specifications": ["..."],
    "key_facts": ["..."],
    "preferences": {
      "explicit": ["..."],
      "inferred": ["..."]
    },
    "dynamic_context": {},
    "importance": 7,
    "last_updated_turn": 22
  },
  "layer2_updated": false,
  "updated_layer2": null
}
```

If Layer 2 needs updating:

```json
{
  "new_layer1_block": { ... },
  "layer2_updated": true,
  "updated_layer2": {
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
