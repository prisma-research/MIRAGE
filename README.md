# MIRAGE

**Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation**

MIRAGE is a controlled empirical study of historical evidence use across conversation states in multimodal personal agents. It measures whether an agent can determine answerability, recover the correct source, and answer from that source rather than from a plausible guess — even after context compaction.

---

## Overview

Long-context AI agents often lose access to earlier artifacts when conversation history is compacted or summarized. MIRAGE keeps evidence objects, questions, and scoring fixed while varying only conversation state, isolating state as the sole experimental variable. The study follows a *what–why–how* progression:

- **RQ1:** What failure patterns emerge as conversation state changes?
- **RQ2:** What retrieval mechanisms produce these patterns?
- **RQ3:** When can retrieval pressure mitigate provenance failure across different states?

### Conversation States

MIRAGE evaluates across four named states spanning three qualitatively distinct regions of the context lifecycle:

| State | Description |
|---|---|
| `S1-d0` | Shallow pre-compaction (near planted evidence) |
| `S1-d50k` | Mid-range same-session depth (~50k effective input tokens) |
| `S1-d80k` | Near context-window boundary (~80k EIT) |
| `S2` | Post-compaction same-session continuation (~100k EIT, 1 compaction) |

### Query Conditions

| Condition | Description |
|---|---|
| `C0` | Natural query — model answers with whatever evidence-access behaviour it adopts by default |
| `Cm` | Retrieval pressure — model is required to invoke the `artifact_recall` tool before answering |

---

## Directory Structure

```
MIRAGE/
├── pipeline/                       # Data generation (planters + query templates)
│   ├── generate_dataset.py         # Generate data/generated_images/ + manifest
│   ├── planters/
│   │   ├── base_planter.py         # BasePlanter + PlantSpec dataclass
│   │   ├── datasets/               # Real-dataset planters
│   │   │   ├── screenspot_planter.py   # ScreenSpot -> screenshot
│   │   │   ├── chartqa_planter.py      # ChartQA -> chart_image
│   │   │   └── _hf_loader.py           # HuggingFace streaming + image save
│   │   └── archive/                # Legacy synthetic planters
│   └── queries/
│       └── reference_templates.py  # make_reference_query / make_grounded_query
├── harness/
│   ├── trial_runner.py             # Session orchestration
│   ├── trial_log.py                # GroundingTrial Pydantic model + save/load
│   ├── scorer_pipeline.py          # Deterministic scoring pipeline
│   ├── experiment_runner.py        # Full batch runner
│   ├── pilot_runner.py             # Smoke test
│   ├── branch_probe_runner.py      # State-conditioned probe execution
│   ├── checkpoint.py               # Checkpoint save/restore for episodes
│   └── run_unified_pilot.py        # Unified pilot with state-conditioned probes
├── scorers/
│   ├── axis1_rpath.py              # Retrieval-path classifier (R_context / R_tool)
│   ├── axis2_identification.py     # Source correctness (SC)
│   └── axis3/                      # Value correctness (VC) components
├── mitigations/citation_force/
│   ├── bootstrap_hook.py           # Inject Cm constraint into BOOTSTRAP.md
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
│       ├── screenshot/             # GUI screenshots (gitignored, regenerate)
│       └── chart/                  # Chart images (gitignored, regenerate)
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
# Edit .env — add your API keys (ANTHROPIC_API_KEY required for LLM-based scoring)

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
- `data/generated_images/screenshot/` — GUI/web/mobile screenshots (ScreenSpot)
- `data/generated_images/chart/` — chart images (ChartQA)
- `data/generated_images/manifest.json` — artifact entries with metadata

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
```

---

## Probe Protocol

Each probe is issued on a freshly restored copy of a checkpoint, guaranteeing per-probe independence. The model returns a single structured response capturing the full grounding chain:

```
ANSWERABLE=YES | NO
SOURCE=<artifact_id or NONE>
ANSWER=<short value or NONE>
```

This reveals whether the model judges the question answerable, identifies the correct source, and extracts the right content — without requiring separate probing stages.

---

## Evaluation

### Deterministic Scoring

All scoring is deterministic — no LLM judge required for the core metrics. For each probe, the model returns a structured response `(z_hat, src_hat, y_hat)` compared against gold annotations:

| Metric | Name | Definition | Computed Over |
|---|---|---|---|
| **AC** | Answerability Correctness | Predicted answerability matches gold | All probes |
| **SC** | Source Correctness | Canonicalized source matches gold artifact | Answerable probes |
| **VC** | Value Correctness | Normalized answer matches gold value | Answerable probes |
| **GC** | Grounded Correctness | SC = 1 AND VC = 1 | Answerable probes |
| **HR** | Hallucination Rate | Claims answerable or provides answer on unanswerable probes | Unanswerable probes |
| **WS** | Wrong Source | Cites a non-NONE artifact that is not the gold source | Answerable probes |
| **PF** | Parse Failure | Output does not satisfy the structured protocol | All probes |

Source identifiers are canonicalized to collapse artifact paths, derived files, and memory notes onto the underlying evidence identifier. Numeric and short string answers are normalized with fixed rules.

### Retrieval-Path Diagnostics

Beyond outcome scores, MIRAGE records the retrieval path for each probe:
- **R_context** — evidence accessed through same-session context continuity
- **R_tool** — evidence accessed through explicit tool-mediated retrieval (e.g., `artifact_recall`)

### Cross-State Diagnostic Indicators

| Indicator | Description |
|---|---|
| **DS** | Depth Sensitivity — GC drop from `S1-d0` to `S1-d80k` |
| **CI** | Compaction Impact — net GC change from `S1-d80k` to `S2` |
| **Ret** | Retention — fraction of `S1-d0`-correct probes still correct at `S1-d80k` |
| **FGR** | False Grounding Rate at `S1-d80k` |
| **OG** | Overestimation Gap — VC minus GC per state |
| **MI** | Mirage Index — gap between claimed answerability and actual grounded correctness |

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
