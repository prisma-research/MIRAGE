# generated_images — Dataset and Manifest

This directory holds the artifact images and `manifest.json` that drive every
GroundingBench trial.

```
data/generated_images/
├── manifest.json          ← dataset registry (1,000 entries)
├── screenshot/            ← GUI screenshots (ScreenSpot, 500 images)
└── chart/                 ← chart images (ChartQA, 500 images)
```

---

## manifest.json — schema

Each entry represents one plantable artifact:

```jsonc
{
  "artifact_id": "ss_940203e4",          // unique ID used as memory identifier
  "plant_type": "screenshot",            // "screenshot" | "chart_image"
  "image_path": "data/generated_images/screenshot/ss_940203e4.png",  // relative to GroundingBench/
  "image_exists": true,
  "plant_prompt":    "Here's a screenshot from tool — take a look.",
  "plant_prompt_ai": "I'm sharing a GUI screenshot. Please examine the interface — labels, interactive components, layout, and any visible text — and save your analysis to memory with identifier 'ss_940203e4'.",
  "artifact_repr":   "GUI Screenshot (ScreenSpot)\nArtifact ID: ss_940203e4\n...",
  "ocr_keywords":    ["search", "button", "menu"],
  "content_query":   "Click the search bar",   // factual Q used in S1/S2/S3 retrieval
  "metadata":        { "dataset": "ScreenSpot", "platform": "tool", ... }
}
```

**Two prompt fields:**

| Field | Purpose | Used by |
|---|---|---|
| `plant_prompt` | Natural language — how a real user would share this image | stored; not used in live trials |
| `plant_prompt_ai` | AI-instructional — tells the agent to examine and save with a specific identifier | harness (Planting turn) |

**`image_path` is relative to `GroundingBench/`.**
The harness resolves it to absolute at load time:
```python
# experiment_runner.py / pilot_runner.py
image_path = str(MANIFEST_PATH.parent.parent / entry["image_path"])
```

---

## How a manifest entry becomes a conversation sent to OpenClaw

Each trial set runs one shared **Planting** turn, then three retrieval scenarios
(**S1**, **S2**, **S3**) that measure grounding under different access conditions.

```
manifest.json entry
       │
       ▼
_manifest_entry_to_spec()          # harness/experiment_runner.py or pilot_runner.py
  → PlantSpec(
      plant_prompt    = entry["plant_prompt"],      # natural version (stored only)
      plant_prompt_ai = entry["plant_prompt_ai"],   # AI-instructional (used in Planting)
      image_path      = <absolute path>,            # resolved from relative
      ...
    )
       │
       ▼
build_trial(plant_prompt=spec.plant_prompt_ai, image_path=spec.image_path, ...)
       │
       ├── Setup: create isolated OpenClaw agent
       │     openclaw agents add gb_<hash> --workspace <baseline snapshot copy>
       │
       ├── PLANTING (shared by all three scenarios)
       │     run_planting() → OpenClawClient.send_message(plant_prompt_ai, image_path)
       │       → CLI: openclaw agent --agent gb_<hash>
       │                             --message "@/abs/path/ss_abc.png <plant_prompt_ai>"
       │                             --session-id <uuid> --json
       │
       │     The agent sees the image + instruction, and (ideally) writes a memory
       │     entry with identifier 'ss_abc' to its workspace.
       │
       │     planting_path observed by checking workspace files for artifact_id:
       │       "agent_write"  → wrote to a .md file via explicit Write tool call
       │       "memory_flush" → appeared in MEMORY.md / memory/*.md via compaction
       │       "none"         → not persisted (hallucination baseline)
       │
       ├── Memory reindex: openclaw memory index --force --agent gb_<hash>
       │
       ├── S1 — In-context (SAME session as Planting, no reset)
       │     run_s1_test(): filler turns injected, reference query sent
       │     artifact still in active context window → expected R_path = R_context
       │     measures in-context utilization failure (e.g. "lost in the middle")
       │
       ├── S2 — Compaction-displaced (same session, long dummy content injected)
       │     long-form content injected until compaction triggers, then query sent
       │     whether artifact survives depends on postCompactionSections policy
       │     omitted if compaction cannot be triggered within injection budget
       │
       └── S3 — Session reset (NEW session, same workspace)
             run_s3_test(): session reset, filler turns, reference query
               → CLI: openclaw agent --agent gb_<hash>
                                     --message "<reference query>"
                                     --session-id <new uuid> --json
             artifact absent from context → must be retrieved via R_boot or R_tool
             main experimental condition; tool_trace + response_text captured for scoring
```

---

## Regenerating the manifest

Images are idempotent — re-running skips existing files and only rewrites
`manifest.json`. No full dataset download occurs (HF streaming, metadata
cached in `data/datasets_cache/`).

```bash
cd GroundingBench
python -m pipeline.generate_dataset
```

Expected output: `1000/1000` images present, manifest written to this directory.
