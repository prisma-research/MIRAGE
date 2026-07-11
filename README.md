# MIRAGE

**Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation**

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://mirage.github.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776ab.svg)](https://www.python.org/downloads/)
[![Code style: pytest](https://img.shields.io/badge/tests-pytest-0a9edc.svg)](#testing)
[![License](https://img.shields.io/badge/license-Research-lightgrey.svg)](#license)

MIRAGE is a controlled empirical study of historical evidence use across conversation states in multimodal personal agents. It measures whether an agent can determine **answerability**, recover the **correct source**, and **answer from that source** rather than from a plausible guess — even after context compaction.

> 📄 **Paper:** [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) &nbsp;•&nbsp; 🌐 **Project page:** [mirage.github.io](https://mirage.github.io)
>
> *(The arXiv identifier above is a placeholder — replace `XXXX.XXXXX` with the real ID once the preprint is live.)*

---

## Overview

Long-context AI agents often lose access to earlier artifacts when conversation history is compacted or summarized. MIRAGE keeps evidence objects, questions, and scoring fixed while varying only conversation state, isolating **state** as the sole experimental variable. The study follows a *what–why–how* progression:

- **RQ1:** What failure patterns emerge as conversation state changes?
- **RQ2:** What retrieval mechanisms produce these patterns?
- **RQ3:** When can retrieval pressure mitigate provenance failure across different states?

### Conversation States

MIRAGE evaluates across named states spanning qualitatively distinct regions of the context lifecycle:

| State | Description |
|---|---|
| `S1-d0` | Shallow pre-compaction (near planted evidence) |
| `S1-d50k` | Mid-range same-session depth (~50k effective input tokens) |
| `S1-d80k` | Near context-window boundary (~80k EIT) |
| `S2` | Post-compaction same-session continuation (~100k EIT, 1 compaction) |
| `S3` | Fresh-session continuation after compaction (optional; requires session restore) |

### Query Conditions

| Condition | Description |
|---|---|
| `C0` | Natural query — model answers with whatever evidence-access behaviour it adopts by default |
| `Cm` | Retrieval pressure — model is required to invoke the `artifact_recall` tool before answering |

Additional CitationForce ladder conditions (`Cp`, `C2`, `C3`, …) are available for the mitigation experiments; see [`configs/presets.py`](configs/presets.py).

---

## Directory Structure

```
MIRAGE/
├── pipeline/                       # Data generation (planters + query templates)
│   ├── generate_dataset.py         # Generate data/generated_images/ + manifest
│   ├── planters/
│   │   ├── base_planter.py         # BasePlanter + PlantSpec dataclass
│   │   └── datasets/               # ScreenSpot / ChartQA planters + HF loader
│   └── queries/
│       └── reference_templates.py  # make_reference_query / make_grounded_query
├── harness/
│   ├── experiment_runner.py        # Full manifest-driven batch runner
│   ├── run_unified_pilot.py        # Unified state-conditioned probe runner
│   ├── pilot_runner.py             # 3-trial smoke test
│   ├── trial_runner.py             # Session orchestration
│   ├── trial_log.py                # GroundingTrial Pydantic model + save/load
│   ├── scorer_pipeline.py          # Deterministic scoring pipeline
│   ├── checkpoint.py               # Checkpoint save/restore for episodes
│   └── run_*.py                    # Additional experiment drivers & smoke tests
├── scorers/
│   ├── axis1_rpath.py              # Retrieval-path classifier (R_context / R_tool)
│   ├── axis2_identification.py     # Source correctness (SC)
│   └── axis3/                      # Value correctness (VC) components
├── mitigations/citation_force/
│   ├── bootstrap_hook.py           # Inject Cm constraint into BOOTSTRAP.md
│   └── openclaw_artifact_recall_plugin.ts  # artifact_recall tool plugin
├── client/
│   ├── openclaw_client.py          # OpenClaw gateway driver (CLI, device-paired)
│   └── usage_proxy.py              # Local reverse proxy for token accounting
├── constants/
│   ├── exp_constants.py            # Experiment parameters and scoring thresholds
│   └── model_constants.py          # Model provider / registry configuration
├── configs/                        # Presets, study banks, subsets, annotations
├── scripts/                        # Serving, run wrappers, and analysis scripts
├── tests/                          # Unit tests (pytest)
├── baseline_snapshot/              # Agent bootstrap files (MEMORY, SOUL, etc.)
├── data/generated_images/          # manifest.json + regenerated images (gitignored)
├── pyproject.toml
├── environment.yml
├── requirements.txt
└── .env.example
```

> `logs/` and `results/` are generated by runs and are **gitignored** — they do not exist in a fresh checkout.

---

## Installation

### 1. Create the environment

```bash
# Conda (recommended — includes vLLM + ms-swift for local VLM serving)
conda env create -f environment.yml
conda activate mirage
```

Or, for the benchmark core only (no local serving) with an existing Python ≥ 3.11:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) local VLM serving extras

Only needed if you plan to serve open-weight VLMs locally with vLLM on GPU:

```bash
pip install flash-attn --no-build-isolation
# vLLM on H100:
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_TRITON_FLASH_ATTN=0
```

### 3. Configure secrets

```bash
cp .env.example .env
# Edit .env — see the Configuration section below
```

### Prerequisites

- **Python ≥ 3.11, < 3.13**
- The **`openclaw` CLI** on your `PATH` — the agent runtime (gateway) that runs each trial. See [OpenClaw](https://github.com/anthropics/openclaw).
- API key(s) for at least one VLM provider (Volcengine, Shubiaobiao, and/or Anthropic), **or** a local vLLM endpoint.
- For local serving: NVIDIA GPU(s) with CUDA (paper open-weight runs used 4× H100).

---

## Configuration

All configuration is read from `.env` (loaded via `python-dotenv`) plus [`constants/model_constants.py`](constants/model_constants.py).

### Environment variables (`.env`)

| Variable | Purpose |
|---|---|
| `OPENCLAW_TOKEN` | Gateway auth token (falls back to `~/.openclaw/openclaw.json → gateway.auth.token`) |
| `OPENCLAW_WS_URL` | Gateway WebSocket, default `ws://127.0.0.1:18789` |
| `OPENCLAW_WORKSPACE` | Agent workspace dir, default `~/.openclaw/workspace` |
| `OPENCLAW_SESSIONS_DIR` | Session store, default `~/.openclaw/agents/main/sessions` |
| `OPENCLAW_COMPACTION_MODEL` | Model used for compaction summarization, e.g. `local_qwen8b/qwen3-vl-8b-instruct` |
| `ANTHROPIC_API_KEY` | Enables Axis-3 (value-correctness) LLM scoring |
| `VOLCENGINE` | ByteDance Ark / Doubao API key |
| `SHUBIOABIAO` | Shubiaobiao proxy API key *(note the spelling used in code)* |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | Axis-3 judge override (point at a local model to avoid remote calls) |
| `LOCAL_VLM_API_KEY` | Placeholder key for local OpenAI-compatible endpoints |
| `LOCAL_QWEN30B_BASE_URL`, `LOCAL_QWEN8B_BASE_URL`, … | Base URLs of local vLLM servers |

### Model providers

The active provider is a single switch in `constants/model_constants.py`:

```python
ACTIVE_PROVIDER = PROVIDER_VOLCENGINE   # or PROVIDER_SHUBIAOBIAO / a local provider
```

| Provider group | Example model IDs | Key env |
|---|---|---|
| `volcengine` (Doubao / Ark) | `doubao-seed-1-6-vision-250815`, `doubao-1.5-vision-pro-250328` | `VOLCENGINE` |
| `shubiaobiao` (proxy) | `gpt-5`, `claude-sonnet-4-6`, `gemini-2.5-flash-nothinking`, `qwen3-vl-32b-instruct`, `glm-4.6v` | `SHUBIOABIAO` |
| `local_*` (vLLM) | `qwen3-vl-30b-instruct`, `qwen3-vl-8b-instruct`, `gemma-3-27b-it`, `internvl3_5-14b-instruct` | `LOCAL_VLM_API_KEY` |

Models are named `provider/model-id` on the command line, e.g. `--model shubiaobiao/gpt-5` or `--model local_qwen30b/qwen3-vl-30b-instruct`.

**Paper backbones** (`PAPER_CORE_MODELS`): `shubiaobiao/gpt-5`, `shubiaobiao/qwen3-vl-32b-instruct`, `shubiaobiao/claude-sonnet-4-6`, `shubiaobiao/gemini-2.5-flash-nothinking`, `volcengine/doubao-1.5-vision-pro-250328`.

---

## Quickstart

```bash
# 1. Generate the evidence artifacts (images + manifest)
python -m pipeline.generate_dataset

# 2. Verify the end-to-end pipeline on 3 trials
python -m harness.pilot_runner

# 3. Run a small state-conditioned probe sweep
python -m harness.run_unified_pilot --model shubiaobiao/gpt-5 --max-bq 5 --run-id smoke
```

---

## Data Generation

Fetches real images from HuggingFace (streaming) and writes them locally:

```bash
python -m pipeline.generate_dataset
```

Samples **500 per dataset** (seed 42) and writes:

- `data/generated_images/screenshot/` — GUI/web/mobile screenshots (ScreenSpot)
- `data/generated_images/chart/` — chart images (ChartQA)
- `data/generated_images/manifest.json` — ~1,000 artifact entries with metadata

| Plant Type | Dataset | HuggingFace ID | Query Source |
|---|---|---|---|
| `screenshot` | ScreenSpot | `rootsautomation/ScreenSpot` | `instruction` field |
| `chart_image` | ChartQA | `ahmed-masry/ChartQA` | `query` field |

> Images are gitignored — regenerate them with the command above. The manifest is the source of truth for all downstream runs.

---

## Running Experiments

### Unified state-conditioned probe runner (primary paper driver)

Runs a question bank across all states with the single-prompt protocol.

```bash
# Baseline (C0) across all states, first 20 questions
python -m harness.run_unified_pilot \
  --model shubiaobiao/gpt-5 \
  --max-bq 20 \
  --run-id gpt5_main

# Baseline vs mitigation (C0 + Cm), specific states, parallel probes
python -m harness.run_unified_pilot \
  --model volcengine/doubao-1.5-vision-pro-250328 \
  --cf-conditions C0,Cm \
  --states d0,d50k,d80k,S2 \
  --max-workers 4 \
  --run-id doubao_mitigation
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--model` | `local_qwen30b/qwen3-vl-30b-instruct` | Probing model (`provider/model-id`) |
| `--bank` | `unified_state_conditioned_bank.json` | Question bank JSON |
| `--states` | all | Comma-separated states, e.g. `d0,d50k,d80k,S2` |
| `--cf-conditions` | `C0` | Comma-separated conditions, e.g. `C0,Cm` |
| `--max-bq` / `--start-index` / `--count` | — | Slice the question bank |
| `--max-workers` | `1` | Concurrent, self-isolated probe branches |
| `--skip-s3` | off | Skip the fresh-session state (useful on HPC without systemd) |
| `--rescore` / `--merge` / `--analysis-only` | — | Post-hoc re-scoring, run merging, analysis-only |

Output: `logs/probes/unified_v1/<run_id>/` containing `results.jsonl`, `summary.json`, and `raw_conversations/`. Re-running with the same `--run-id` **resumes** (skips probes already recorded).

### Full manifest-driven batch runner

```bash
python -m harness.experiment_runner --preset main_backbone --model shubiaobiao/gpt-5
python -m harness.experiment_runner --dry-run                 # print the parameter matrix only
```

Key flags: `--preset` (`main_backbone`, `s1_depth`, `mitigation_s3`, `paper_s2s3_compaction`), `--scenarios S1 S2 S3`, `--plant-type`, `--conditions C0 C3`, `--reference-styles`, `--context-thresholds 25000 50000 75000 100000`, `--subset-file`, `--max-workers`, `--n-trials`, `--run-id`.

Output: `logs/trials_<run_id>/` (per-cell `GroundingTrial` JSONs) with a `run_config.json` provenance record and `logs/errors_<run_id>.jsonl`. After a non-dry run it automatically produces an evaluation summary (see below).

Convenience wrapper:

```bash
bash scripts/start.sh          # experiment_runner across S1/S2/S3, 20 workers
bash scripts/progress.sh       # count completed trials by scenario
bash scripts/stop.sh           # kill runner + gateway processes
```

---

## Local VLM Serving (open-weight backbones)

Serve OpenAI-compatible endpoints with vLLM / ms-swift, then point `--model local_*` at them:

```bash
bash scripts/serve_qwen30b.sh   # Qwen3-VL-30B on :8000  (answering model)
bash scripts/serve_qwen8b.sh    # Qwen3-VL-8B  on :8001  (Axis-3 judge / compaction)
bash scripts/start_vllm_gemma27b.sh   # Gemma-3-27B on :8004
```

The runners auto-start a **UsageProxy** (`client/usage_proxy.py`) when a provider needs it — it de-streams responses to recover accurate prompt/completion token counts and patches each worker's `openclaw.json`. To run it standalone for debugging:

```bash
python -m client.usage_proxy --port <PORT>
```

---

## HPC / SLURM

Batch scripts live in `scripts/*.sbatch` and are **resumable** (a fixed `--run-id` skips completed probes). Examples:

```bash
sbatch scripts/slurm_citationforce_ladder.sbatch     # CitationForce mitigation ladder (S2)
sbatch scripts/slurm_modext_openvlm_sweep.sbatch     # modality extension on open VLMs (4× H100)
sbatch scripts/build_modality_family.sbatch          # build native checkpoint family
sbatch scripts/probe_modext_doubao.sbatch            # Doubao probe over the modality family
```

---

## Results & Paper Tables

**Run → summary.** A non-dry `experiment_runner` run auto-invokes the eval summary; you can also run it manually:

```bash
bash scripts/eval.sh --trials-dir logs/trials_<run_id>     # → results/summary_N.{json,md}
```

**Summaries/trials → paper tables:**

```bash
python scripts/aggregate_paper_results.py \
  --backbone-dirs logs/trials_gpt5 logs/trials_doubao \
  --depth-dir logs/trials_depth \
  --mitigation-dir logs/probes/unified_v1/doubao_mitigation \
  --output results/paper_tables
```

Complementary table generators:

| Script | Produces |
|---|---|
| `scripts/axis_ablation.py` | `table_axis_ablation.csv` (R_path vs I_strict vs S_hat) |
| `scripts/reanalyze_mitigation.py` | CitationForce C0/Cp/Cm aggregate + paired + signals tables |
| `scripts/compaction_threshold_sweep.py` | Multi-threshold compaction robustness |
| `scripts/gen_fig3_rpath_stacked_bar.py` | Figure 3: R_path composition stacked bar |
| `scripts/analyze_modality_ext.py` | Per-modality × per-state analysis with Wilson CIs |

---

## Probe Protocol

Each probe is issued on a **freshly restored copy** of a checkpoint, guaranteeing per-probe independence. The model returns a single structured response capturing the full grounding chain:

```
ANSWERABLE=YES | NO
SOURCE=<artifact_id or NONE>
ANSWER=<short value or NONE>
```

This reveals whether the model judges the question answerable, identifies the correct source, and extracts the right content — without requiring separate probing stages.

---

## Evaluation

### Deterministic Scoring

Core metrics are **deterministic** — no LLM judge is required. For each probe, the model returns `(z_hat, src_hat, y_hat)`, compared against gold annotations:

| Metric | Name | Definition | Computed Over |
|---|---|---|---|
| **AC** | Answerability Correctness | Predicted answerability matches gold | All probes |
| **SC** | Source Correctness | Canonicalized source matches gold artifact | Answerable probes |
| **VC** | Value Correctness | Normalized answer matches gold value | Answerable probes |
| **GC** | Grounded Correctness | SC = 1 AND VC = 1 | Answerable probes |
| **HR** | Hallucination Rate | Claims answerable / answers on unanswerable probes | Unanswerable probes |
| **WS** | Wrong Source | Cites a non-NONE artifact that is not the gold source | Answerable probes |
| **PF** | Parse Failure | Output does not satisfy the structured protocol | All probes |

Source identifiers are canonicalized to collapse artifact paths, derived files, and memory notes onto the underlying evidence identifier. Numeric and short-string answers are normalized with fixed rules. (Value Correctness may optionally use an LLM judge configured via `LLM_*` / `ANTHROPIC_API_KEY`.)

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

## Testing

```bash
pytest                 # runs tests/ (async mode auto, configured in pyproject.toml)
pytest --cov           # with coverage
```

---

## Key Constants

| Item | Value |
|---|---|
| OpenClaw WebSocket | `ws://127.0.0.1:18789` |
| Auth token | `~/.openclaw/openclaw.json` → `gateway.auth.token` |
| Workspace | `~/.openclaw/workspace/` |
| Memory file | `~/.openclaw/workspace/MEMORY.md` |
| Per-worker gateway state | `~/.openclaw-run-<run_id>-<n>` (created & cleaned per run) |

---

## Citation

If you use MIRAGE in your research, please cite:

```bibtex
@article{mirage2026,
  title   = {MIRAGE: Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation},
  author  = {Zhang, Wenxiao and others},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

This project is part of ongoing research. See the repository for license details.
