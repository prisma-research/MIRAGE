# MIRAGE

**Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation**

MIRAGE is a controlled empirical evaluation framework for studying how AI agents retrieve, attribute, and ground responses in multimodal artifacts across conversation sessions. It measures whether an agent can correctly retrieve artifacts (e.g., screenshots, charts) encountered in a previous session and produce responses that are faithfully grounded in those artifacts — even after context compaction.

---

## Overview

Long-context AI agents often lose access to earlier artifacts when conversation history is compacted or summarized. MIRAGE provides a systematic protocol to evaluate:

1. **Retrieval path** — How does the agent retrieve the artifact? (bootstrap memory, tool call, or not at all)
2. **Identification accuracy** — Did it find the correct artifact?
3. **Answer grounding** — Is the response actually supported by the artifact content?

The framework supports multiple memory states (pre-compaction, post-compaction, with/without prewritten memory), multiple VLM backends, and configurable mitigation strategies (CitationForce).

---

## Directory Structure

```
MIRAGE/
├── pipeline/                       # Data generation (planters + query templates)
│   ├── generate_dataset.py         # Generate data/generated_images/ + manifest
│   ├── planters/
│   │   ├── base_planter.py         # BasePlanter + PlantSpec dataclass
│   │   ├── datasets/               # Real-dataset planters
│   │   │   ├── screenspot_planter.py   # ScreenSpot -> screenshot (500 samples)
│   │   │   ├── chartqa_planter.py      # ChartQA -> chart_image (500 samples)
│   │   │   └── _hf_loader.py           # HuggingFace streaming + image save
│   │   └── archive/                # Legacy synthetic planters
│   └── queries/
│       └── reference_templates.py  # make_reference_query / make_grounded_query
├── harness/
│   ├── trial_runner.py             # Session A/B orchestration
│   ├── trial_log.py                # GroundingTrial Pydantic model + save/load
│   ├── scorer_pipeline.py          # Applies all 3 scoring axes to a trial
│   ├── experiment_runner.py        # Full batch runner
│   ├── pilot_runner.py             # Smoke test (3 trials)
│   ├── branch_probe_runner.py      # State-conditioned probe execution
│   ├── checkpoint.py               # Checkpoint save/restore for episodes
│   └── run_unified_pilot.py        # Unified pilot with state-conditioned probes
├── scorers/
│   ├── axis1_rpath.py              # R-path classifier (R_context / R_tool / R_none)
│   ├── axis2_identification.py     # Artifact identification (1 / 0 / empty)
│   └── axis3/
│       ├── claim_extractor.py      # LLM claim extraction
│       ├── claim_verifier.py       # LLM claim verification
│       ├── aggregator.py           # Gamma score + decision rule -> S_hat
│       └── ensemble.py             # 3-judge majority vote
├── mitigations/citation_force/
│   ├── bootstrap_hook.py           # Inject constraint into BOOTSTRAP.md
│   └── openclaw_artifact_recall_plugin.ts  # artifact_recall tool plugin
├── client/
│   ├── openclaw_client.py          # WebSocket client for OpenClaw gateway
│   └── usage_proxy.py              # Local reverse proxy for token accounting
├── constants/
│   ├── exp_constants.py            # Experiment parameters and scoring thresholds
│   └── model_constants.py          # Model provider configuration
├── configs/                        # Experiment presets and study configurations
├── scripts/                        # Utility and analysis scripts
├── tests/                          # Unit tests
├── baseline_snapshot/              # Agent bootstrap files (MEMORY, SOUL, etc.)
├── data/
│   └── generated_images/
│       ├── manifest.json           # ~1,000 artifact entries
│       ├── screenshot/             # 500 GUI screenshots (gitignored, regenerate)
│       └── chart/                  # 500 chart images (gitignored, regenerate)
├── pyproject.toml
├── environment.yml
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate mirage

# Configure environment variables
cp .env.example .env
# Edit .env — add your API keys (ANTHROPIC_API_KEY required for Axis 3 scoring)

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

- Python >= 3.11
- A running [OpenClaw](https://github.com/anthropics/openclaw) instance (default: `ws://127.0.0.1:18789`)
- API keys for at least one VLM provider (Anthropic, Shubiaobiao, or Volcengine)

---

## Data Generation

Fetches real images from HuggingFace and writes them to `data/`:

```bash
python -m pipeline.generate_dataset
```

**Output:**
- `data/generated_images/screenshot/` — 500 GUI/web/mobile screenshots (ScreenSpot)
- `data/generated_images/chart/` — 500 real charts (ChartQA)
- `data/generated_images/manifest.json` — 1,000 artifact entries with metadata

Each manifest entry contains the `artifact_id`, `image_path`, `plant_prompt`, `artifact_repr`, OCR keywords, and a `content_query` drawn directly from dataset annotations.

| Plant Type | Dataset | HuggingFace ID | Query Source |
|---|---|---|---|
| `screenshot` | ScreenSpot | `rootsautomation/ScreenSpot` | `instruction` field |
| `chart_image` | ChartQA | `ahmed-masry/ChartQA` | `query` field |

---

## Running Experiments

### Pilot (smoke test)
```bash
python -m harness.pilot_runner
```

### Unified state-conditioned pilot
```bash
python -m harness.run_unified_pilot --model shubiaobiao/gpt-5 --tag smoke
```

### Full experiment
```bash
python -m harness.experiment_runner              # full run
python -m harness.experiment_runner --dry-run    # print parameter matrix only
python -m harness.experiment_runner --plant-type screenshot
```

---

## Scoring

### Axis 1 — Retrieval Path
How did the agent retrieve the artifact?
- `R_context` — found in context window (bootstrap memory reads)
- `R_tool` — agent called `memory_search` / `artifact_recall` after query
- `R_none` — artifact not retrieved

### Axis 2 — Identification
Did the agent find the correct artifact?
- `1` — correct artifact_id in tool result
- `0` — retrieved wrong artifact
- `empty` — retrieval returned empty (hallucinated path)

### Axis 3 — Answer Support
Is the response grounded in the artifact content?
- `S_hat in {0, 0.5, 1, "indeterminate"}` via 3-judge LLM ensemble
- Requires `ANTHROPIC_API_KEY` in `.env`

### Composite Score
```
F = R_binary x I_strict x S_hat
```
Full grounding success requires all three axes to pass.

---

## Key Constants

| Item | Value |
|---|---|
| OpenClaw WebSocket | `ws://127.0.0.1:18789` |
| Auth token | `~/.openclaw/openclaw.json` -> `gateway.auth.token` |
| Workspace | `~/.openclaw/workspace/` |
| Memory file | `~/.openclaw/workspace/MEMORY.md` |

---

## License

This project is part of ongoing research. See the repository for license details.
