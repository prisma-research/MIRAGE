"""
Experiment Runner — manifest-driven batch for MM-2 GroundingBench.

Each trial set shares one Session A planting event and produces up to three
GroundingTrial records (S1, S2 if compaction triggered, S3).

Artifacts are drawn from the pre-generated manifest (data/generated_images/manifest.json).
When a preset or --subset-file is specified, only the listed artifact IDs are used
(in manifest order for determinism); N_PER_TYPE is ignored.

Named presets (defined in configs/presets.py):
    main_backbone           — 8+8 artifacts, S1/S2/S3, definite+demonstrative styles, C0
    s1_depth                — 6+6 artifacts, S1 only, definite style, depths 0/16k/32k
    mitigation_s3           — 6+6 artifacts, S3 only, definite+demonstrative styles, C0+C3
    paper_s2s3_compaction   — 20+20 artifacts, S2/S3, definite, C0, thresholds 25k/50k/75k/100k

Usage:
    cd GroundingBench
    python -m harness.experiment_runner --preset main_backbone --dry-run
    python -m harness.experiment_runner --preset main_backbone
    python -m harness.experiment_runner --preset s1_depth --model shubiaobiao/claude-sonnet-4-6
    python -m harness.experiment_runner --dry-run    # legacy full matrix

    # Compaction threshold sensitivity sweep (reviewer response):
    python -m harness.experiment_runner --preset paper_s2s3_compaction \\
        --model shubiaobiao/gpt-5 --run-id paper_compactsens_gpt5_0324 --max-workers 4

    # Override thresholds via CLI:
    python -m harness.experiment_runner --preset paper_s2s3_compaction \\
        --context-thresholds 25000 100000 --model shubiaobiao/gpt-5 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import datetime
import hashlib
import json
import logging
import os
import plistlib
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from pipeline.planters.base_planter import PlantSpec
from pipeline.queries.reference_templates import make_reference_query, make_grounded_query
from harness.trial_runner import build_trial
from harness.trial_log import TrialConfig, GroundingTrial, load_trials
from harness.scorer_pipeline import score_trial
from constants import PLANT_TYPES, N_PER_TYPE, HISTORY_DEPTHS, REF_STYLES, CF_CONDITIONS, VLM_MODEL, S2_CONTEXT_THRESHOLD, RESERVE_TOKENS_FLOOR, EXPERIMENT_MODEL_REGISTRY
from client.usage_proxy import UsageProxy, patch_openclaw_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "generated_images" / "manifest.json"
# LOGS_DIR / ERRORS_LOG default to "trials" — overridden dynamically in __main__ per run.
LOGS_DIR   = Path(__file__).parent.parent / "logs" / "trials"
ERRORS_LOG = Path(__file__).parent.parent / "logs" / "errors_trials.jsonl"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Gateway lifecycle
# ---------------------------------------------------------------------------

GATEWAY_PLIST         = Path.home() / "Library" / "LaunchAgents" / "ai.openclaw.gateway.plist"
GATEWAY_LAUNCHD_LABEL = "ai.openclaw.gateway"
GATEWAY_LOG           = Path.home() / ".openclaw" / "logs" / "gateway.log"
GATEWAY_ERR           = Path.home() / ".openclaw" / "logs" / "gateway.err.log"

_gateway_proc: subprocess.Popen | None = None
_gateway_managed = False

# Per-worker gateway processes (parallel mode)
_worker_gateway_procs: list[subprocess.Popen] = []
_worker_state_dirs_this_run: list[Path] = []  # dirs owned by this process (for cleanup)


def _launchd_loaded() -> bool:
    """Return True if the OpenClaw gateway launchd service is loaded."""
    try:
        result = subprocess.run(
            ["launchctl", "list", GATEWAY_LAUNCHD_LABEL],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _start_managed_gateway() -> None:
    """Ensure OpenClaw gateway is running.

    If the gateway is already reachable (e.g. via launchd), use it as-is.
    The launchd gateway handles hot agent registration; restarting it would
    create a fresh process that doesn't see agents added after startup.
    Only start a fresh process if no gateway is currently reachable.
    """
    global _gateway_proc, _gateway_managed

    # Use existing gateway if reachable — avoids hot-reload issues with managed process
    result = subprocess.run(
        ["openclaw", "gateway", "status"], capture_output=True, timeout=5,
    )
    if b"RPC probe: ok" in result.stdout or b"Listening" in result.stdout:
        logger.info("Gateway already reachable — using existing process")
        return

    # No reachable gateway — kill any stale processes and start fresh
    if GATEWAY_PLIST.exists() and _launchd_loaded():
        subprocess.run(["launchctl", "unload", str(GATEWAY_PLIST)], check=False, timeout=10)
        time.sleep(1)

    for proc_name in ("openclaw-gateway", "openclaw-agent", "openclaw"):
        subprocess.run(["pkill", "-x", proc_name], check=False, timeout=5)
    time.sleep(0.5)

    if not GATEWAY_PLIST.exists():
        raise RuntimeError(f"Gateway plist not found: {GATEWAY_PLIST}")

    with GATEWAY_PLIST.open("rb") as f:
        plist = plistlib.load(f)

    cmd = plist.get("ProgramArguments", [])
    plist_env = plist.get("EnvironmentVariables", {})
    merged_env = {**os.environ, **plist_env}

    GATEWAY_LOG.parent.mkdir(parents=True, exist_ok=True)
    _gateway_proc = subprocess.Popen(
        cmd,
        stdout=GATEWAY_LOG.open("a"),
        stderr=GATEWAY_ERR.open("a"),
        env=merged_env,
    )

    # Poll for readiness (up to 15 s)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["openclaw", "gateway", "status"],
            capture_output=True, timeout=5,
        )
        if result.returncode == 0:
            _gateway_managed = True
            logger.info("OpenClaw gateway started (pid=%d)", _gateway_proc.pid)
            return
        time.sleep(0.5)

    _gateway_proc.kill()
    raise RuntimeError("OpenClaw gateway did not become ready within 15 s")


def _stop_managed_gateway() -> None:
    """Stop the managed gateway process (idempotent)."""
    global _gateway_managed
    if not _gateway_managed:
        return
    _gateway_managed = False

    if _gateway_proc is not None:
        try:
            _gateway_proc.terminate()
            _gateway_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _gateway_proc.kill()
        except Exception:
            pass

    for proc_name in ("openclaw-gateway", "openclaw-agent", "openclaw"):
        subprocess.run(["pkill", "-x", proc_name], check=False, timeout=5)

    logger.info("OpenClaw gateway stopped")


def _scan_used_ports() -> set[int]:
    """Return set of TCP ports currently in use on localhost."""
    import socket
    used: set[int] = set()
    # Also read ports from existing openclaw state dirs
    for d in Path.home().iterdir():
        if not d.is_dir():
            continue
        cfg = d / "openclaw.json"
        if cfg.exists():
            try:
                with cfg.open() as f:
                    data = json.load(f)
                port = data.get("gateway", {}).get("port")
                if port:
                    # Check if actually listening
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(0.2)
                        if s.connect_ex(("127.0.0.1", port)) == 0:
                            used.add(port)
            except Exception:
                pass
    return used


def _alloc_ports(n: int, start: int = 19500) -> list[int]:
    """Find n consecutive free ports starting from `start`."""
    import socket
    used = _scan_used_ports()
    ports: list[int] = []
    p = start
    while len(ports) < n:
        if p not in used:
            # Double-check with a real bind attempt
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    ports.append(p)
                except OSError:
                    pass
        p += 1
    return ports


def _alloc_state_dirs(n: int, run_id: str) -> list[Path]:
    """Generate n unique state dir paths for this run using run_id."""
    prefix = f".openclaw-run-{run_id}"
    return [Path.home() / f"{prefix}-{i}" for i in range(n)]


def _start_worker_gateways(
    n_workers: int,
    proxy_port: int | None = None,
    context_window: int | None = None,
) -> list[Path]:
    """Start N isolated gateway instances (one per worker slot).

    Each worker gets its own state dir with a patched openclaw.json.
    Patches (proxy baseUrl, model merge, contextWindow, compaction.model)
    are applied per-worker — the global ~/.openclaw/openclaw.json is
    never modified, so concurrent experiment_runner processes are safe.

    Returns list of state_dirs (index = worker slot).
    """
    global _worker_gateway_procs, _worker_state_dirs_this_run

    # Include PID so concurrent experiment_runner processes never share state dirs or ports
    run_id = f"{datetime.datetime.utcnow().strftime('%m%d%H%M%S')}_{os.getpid()}"
    # Spread port ranges by PID to avoid TOCTOU collisions between concurrent processes
    port_start = 19500 + (os.getpid() % 200) * 20
    ports = _alloc_ports(n_workers, start=port_start)
    state_dirs = _alloc_state_dirs(n_workers, run_id)
    _worker_state_dirs_this_run = state_dirs

    logger.info(
        "Allocating %d worker gateways: ports %d–%d, state dirs ~/.openclaw-run-%s-{0..%d}",
        n_workers, ports[0], ports[-1], run_id, n_workers - 1,
    )

    main_cfg_path = Path.home() / ".openclaw" / "openclaw.json"
    with main_cfg_path.open() as f:
        main_cfg = json.load(f)

    for i in range(n_workers):
        port = ports[i]
        state_dir = state_dirs[i]
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "logs").mkdir(parents=True, exist_ok=True)

        # Write config: copy main config (excluding web search), update port.
        # Exclude "web" key — omitting it disables web_search for trial agents.
        # Preserve "agents" block (defaults, model, etc.) but drop "agents.list"
        # (per-worker agents are registered dynamically via `openclaw agents add`).
        cfg = {k: v for k, v in main_cfg.items() if k != "web"}
        cfg["gateway"] = {**main_cfg.get("gateway", {}), "port": port}
        # Use "full" tools profile so plugin-provided tools (e.g. artifact_recall)
        # are available to trial agents. The "coding" profile blocks them.
        if "tools" not in cfg:
            cfg["tools"] = {}
        cfg["tools"]["profile"] = "full"
        # Drop per-agent list (workers register their own agents)
        agents_cfg = cfg.get("agents", {})
        agents_cfg.pop("list", None)
        # Set compaction defaults
        defaults_cfg = agents_cfg.get("defaults", {})
        compaction_cfg = defaults_cfg.get("compaction", {})
        compaction_cfg.update({
            "mode": "safeguard",
            "reserveTokensFloor": 20000,
            "memoryFlush": {"enabled": True},
        })
        defaults_cfg["compaction"] = compaction_cfg
        agents_cfg["defaults"] = defaults_cfg
        cfg["agents"] = agents_cfg

        # Apply experiment patches (proxy, model merge, contextWindow, compaction.model)
        # directly to this worker's config — no global config mutation.
        patch_openclaw_config(
            cfg,
            proxy_port=proxy_port,
            context_window_override=context_window,
        )

        with (state_dir / "openclaw.json").open("w") as f:
            json.dump(cfg, f, indent=2)

        # Seed auth-profiles for the main agent so all providers have API keys.
        # Copy from the global main agent, then inject entries for any
        # EXPERIMENT_MODEL_REGISTRY providers whose keys are in env.
        auth_dir = state_dir / "agents" / "main" / "agent"
        auth_dir.mkdir(parents=True, exist_ok=True)
        auth_file = auth_dir / "auth-profiles.json"
        main_auth = Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json"
        if main_auth.exists():
            auth_data = json.loads(main_auth.read_text())
        else:
            auth_data = {"version": 1, "profiles": {}}
        profiles = auth_data.setdefault("profiles", {})
        for pid, reg in EXPERIMENT_MODEL_REGISTRY.items():
            profile_key = f"{pid}:default"
            if profile_key not in profiles:
                for env_name in reg.get("apiKeyEnvs", []):
                    key_val = os.environ.get(env_name)
                    if key_val:
                        profiles[profile_key] = {
                            "type": "api_key",
                            "provider": pid,
                            "key": key_val,
                        }
                        break
        with auth_file.open("w") as f:
            json.dump(auth_data, f, indent=2)

        env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
        log_out = (state_dir / "logs" / "gateway.log").open("a")
        log_err = (state_dir / "logs" / "gateway.err.log").open("a")
        proc = subprocess.Popen(
            ["openclaw", "gateway", "run", "--port", str(port), "--force"],
            stdout=log_out,
            stderr=log_err,
            env=env,
        )
        _worker_gateway_procs.append(proc)
        logger.info("Worker gateway %d starting (port=%d, state=%s)", i, port, state_dir.name)

    # Wait for all gateways to be ready
    for i, state_dir in enumerate(state_dirs):
        env = {**os.environ, "OPENCLAW_STATE_DIR": str(state_dir)}
        deadline = time.monotonic() + 60.0
        ready = False
        while time.monotonic() < deadline:
            try:
                result = subprocess.run(
                    ["openclaw", "gateway", "status"],
                    capture_output=True, timeout=15, env=env,
                )
                if result.returncode == 0:
                    ready = True
                    logger.info("Worker gateway %d ready (port=%d)", i, ports[i])
                    break
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.5)
        if not ready:
            raise RuntimeError(f"Worker gateway {i} (port={ports[i]}) did not start within 60s")

    return state_dirs


def _stop_worker_gateways() -> None:
    """Stop all worker gateway processes."""
    global _worker_gateway_procs
    for proc in _worker_gateway_procs:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass
    _worker_gateway_procs.clear()
    logger.info("Worker gateways stopped")


# ---------------------------------------------------------------------------
# Manifest / artifact pool
# ---------------------------------------------------------------------------

_ARTIFACT_POOL: dict[str, list[dict]] = {}


def _load_artifact_pool() -> dict[str, list[dict]]:
    """Load manifest.json and group image-valid entries by plant_type."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_PATH}. "
            "Run: cd GroundingBench && python -m data.generate_dataset"
        )
    with MANIFEST_PATH.open(encoding="utf-8") as f:
        entries = json.load(f)

    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        if entry.get("image_exists"):
            by_type.setdefault(entry["plant_type"], []).append(entry)

    for pt, pool in by_type.items():
        logger.info("Artifact pool: %s → %d entries", pt, len(pool))

    return by_type


def _load_subset(path: str | Path) -> tuple[dict[str, list[str]], str]:
    """Load artifact ID lists from a subset config JSON.

    Returns (subset_dict, sha256_hex) for reproducibility tracking.
    subset_dict maps plant_type → list of artifact_ids.
    """
    p = Path(path)
    if not p.is_absolute():
        from configs.presets import resolve_subset_path
        p = resolve_subset_path(str(path))
    with p.open("rb") as f:
        raw = f.read()
    data = json.loads(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    subset: dict[str, list[str]] = {}
    for pt in PLANT_TYPES:
        if pt in data:
            subset[pt] = data[pt]
    return subset, sha256


def _get_pool(plant_type: str, subset_ids: set[str] | None = None) -> list[dict]:
    """Return artifact entries for a plant_type.

    If subset_ids is provided, filter manifest entries to only those IDs
    (in manifest order for determinism), ignoring N_PER_TYPE.
    Otherwise return the first N_PER_TYPE entries.
    """
    global _ARTIFACT_POOL
    if not _ARTIFACT_POOL:
        _ARTIFACT_POOL = _load_artifact_pool()
    pool = _ARTIFACT_POOL.get(plant_type, [])
    if subset_ids is not None:
        filtered = [e for e in pool if e["artifact_id"] in subset_ids]
        return filtered
    if len(pool) < N_PER_TYPE:
        logger.warning(
            "Pool for '%s' has only %d entries (N_PER_TYPE=%d).",
            plant_type, len(pool), N_PER_TYPE,
        )
    return pool[:N_PER_TYPE]


def _manifest_entry_to_spec(entry: dict) -> PlantSpec:
    """Reconstruct a PlantSpec from a manifest entry (no live dataset fetch needed)."""
    raw_path = entry.get("image_path")
    image_path = str(MANIFEST_PATH.parent.parent.parent / raw_path) if raw_path else None
    return PlantSpec(
        artifact_id=entry["artifact_id"],
        plant_type=entry["plant_type"],
        plant_prompt=entry["plant_prompt"],
        plant_prompt_ai=entry["plant_prompt_ai"],
        artifact_ocr_keywords=entry.get("ocr_keywords", []),
        artifact_repr=entry.get("artifact_repr", ""),
        image_path=image_path,
        metadata=entry.get("metadata", {}),
        content_query=entry.get("content_query", ""),
    )


# ---------------------------------------------------------------------------
# RunSpec — typed resolved configuration
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """Fully resolved configuration for a single experiment run.

    Built by _resolve_run_spec() from preset defaults + CLI overrides.
    _build_matrix() reads only from RunSpec (pure function).
    """
    scenarios: list[str]
    reference_styles: list[str]
    conditions: list[str]
    history_depths: list[int]
    # Artifact IDs per plant_type — None means use first N_PER_TYPE from manifest
    artifact_ids_by_modality: dict[str, list[str]] | None = None
    subset_file: str | None = None
    subset_sha256: str | None = None
    # Preset name (for run_config.json logging)
    preset: str | None = None
    # Context thresholds for S2/S3 compaction sweep
    context_thresholds: list[int] = field(default_factory=lambda: [50_000])
    # Annotation file for paper runs (optional)
    annotation_file: str | None = None
    annotation_sha256: str | None = None


def _resolve_run_spec(args: argparse.Namespace) -> RunSpec:
    """Merge preset defaults with CLI overrides into a RunSpec.

    Resolution order (later overrides earlier):
      1. Hardcoded defaults
      2. Preset values (if --preset given)
      3. Explicit CLI args (if provided)
    """
    from configs.presets import PRESETS, resolve_subset_path

    # --- defaults ---
    spec_scenarios = list(args.scenarios) if args.scenarios else ["S1", "S2", "S3"]
    spec_styles = list(args.reference_styles) if args.reference_styles else list(REF_STYLES)
    spec_conditions = list(args.conditions) if args.conditions else list(CF_CONDITIONS)
    spec_depths = [0]
    spec_thresholds = [S2_CONTEXT_THRESHOLD]
    spec_subset_file: str | None = getattr(args, "subset_file", None)
    spec_preset: str | None = getattr(args, "preset", None)

    # --- apply preset (fills in any unset values) ---
    if spec_preset and spec_preset in PRESETS:
        p = PRESETS[spec_preset]
        # Only apply preset values where CLI did not explicitly override
        if not args.scenarios:
            spec_scenarios = list(p.get("scenarios", spec_scenarios))
        if not args.reference_styles:
            spec_styles = list(p.get("reference_styles", spec_styles))
        if not args.conditions:
            spec_conditions = list(p.get("conditions", spec_conditions))
        spec_depths = list(p.get("history_depths", [0]))
        spec_thresholds = list(p.get("context_thresholds", [S2_CONTEXT_THRESHOLD]))
        if spec_subset_file is None:
            spec_subset_file = p.get("subset_file")

    # --- CLI override for context thresholds ---
    cli_thresholds = getattr(args, "context_thresholds", None)
    if cli_thresholds:
        spec_thresholds = cli_thresholds

    # --- load subset if specified ---
    artifact_ids: dict[str, list[str]] | None = None
    subset_sha256: str | None = None
    if spec_subset_file:
        subset_dict, subset_sha256 = _load_subset(spec_subset_file)
        artifact_ids = subset_dict

    # --- annotation file (optional, for paper runs) ---
    # CLI overrides preset; preset supplies default
    annotation_file: str | None = getattr(args, "annotation_file", None)
    if annotation_file is None and spec_preset and spec_preset in PRESETS:
        annotation_file = PRESETS[spec_preset].get("annotation_file")
    annotation_sha256: str | None = None
    if annotation_file:
        ann_path = Path(annotation_file)
        if not ann_path.is_absolute():
            ann_path = Path(__file__).parent.parent / ann_path
        if ann_path.exists():
            annotation_sha256 = hashlib.sha256(ann_path.read_bytes()).hexdigest()

    return RunSpec(
        scenarios=spec_scenarios,
        reference_styles=spec_styles,
        conditions=spec_conditions,
        history_depths=spec_depths,
        context_thresholds=spec_thresholds,
        artifact_ids_by_modality=artifact_ids,
        subset_file=spec_subset_file,
        subset_sha256=subset_sha256,
        preset=spec_preset,
        annotation_file=annotation_file,
        annotation_sha256=annotation_sha256,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cell_key(
    plant_type: str, artifact_id: str, history_depth: int, ref_style: str, cf: str,
    context_threshold: int = 50_000,
) -> tuple:
    return (plant_type, artifact_id, history_depth, ref_style, cf, context_threshold)


def _query_seed(
    plant_type: str, artifact_id: str, history_depth: int, ref_style: str, cf: str,
    context_threshold: int = 50_000,
) -> int:
    """Deterministic seed for query template selection."""
    return abs(hash(_build_cell_key(plant_type, artifact_id, history_depth, ref_style, cf, context_threshold))) % 100_000


def _build_matrix(spec: RunSpec, plant_type_filter: str | None = None) -> list[tuple]:
    """Return ordered list of (plant_type, entry, history_depth, ref_style, cf, context_threshold) cells.

    Pure function — reads only from RunSpec and the artifact pool.
    Each cell is: (plant_type, manifest_entry_dict, history_depth, ref_style, cf, context_threshold)
    """
    cells = []
    for pt in PLANT_TYPES:
        if plant_type_filter and pt != plant_type_filter:
            continue
        subset_ids: set[str] | None = None
        if spec.artifact_ids_by_modality and pt in spec.artifact_ids_by_modality:
            subset_ids = set(spec.artifact_ids_by_modality[pt])
        pool = _get_pool(pt, subset_ids=subset_ids)
        for entry in pool:
            for rs in spec.reference_styles:
                for cf in spec.conditions:
                    for hd in spec.history_depths:
                        for ct in spec.context_thresholds:
                            cells.append((pt, entry, hd, rs, cf, ct))
    return cells


def _log_error(cell: tuple, error: str) -> None:
    entry = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "cell": [cell[0], cell[1].get("artifact_id", "?"), cell[2], cell[3], cell[4], cell[5]],
        "error": error,
    }
    with ERRORS_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _load_completed_cells(logs_dir: Path) -> set[tuple]:
    """
    Scan saved trial JSON files to find already-completed cells.

    A cell is complete only if:
    - JSON file exists, AND
    - The loaded trial has both R_path and outcome set (not None)

    This prevents partially-scored trials (JSON written before scoring crashed)
    from being silently skipped.
    """
    completed: set[tuple] = set()

    for path in logs_dir.glob("*.json"):
        try:
            trial = GroundingTrial.model_validate_json(path.read_text())
            # Only count as complete if scoring is done
            if trial.R_path is None or trial.outcome is None:
                continue
            completed.add(_build_cell_key(
                trial.theta.plant_type,
                trial.artifact_id,
                trial.theta.history_depth,
                trial.theta.reference_style,
                trial.theta.citation_force_condition,
                trial.theta.context_threshold,
            ))
        except Exception:
            pass

    return completed


def _write_run_config(
    logs_dir: Path,
    run_id: str,
    model: str,
    spec: RunSpec,
    cells: list[tuple],
) -> None:
    """Write run_config.json to logs_dir for reproducibility and paper traceability."""
    # Get git commit
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unknown"

    # Count cells per model (all cells go to one model in this run)
    n_cells = len(cells)

    # Artifact IDs by modality from spec (or derive from cells if not in spec)
    if spec.artifact_ids_by_modality:
        artifact_ids_by_modality = spec.artifact_ids_by_modality
        n_artifacts = {pt: len(ids) for pt, ids in artifact_ids_by_modality.items()}
    else:
        # Derive from cells
        artifact_ids_by_modality = {}
        for pt, entry, *_ in cells:
            artifact_ids_by_modality.setdefault(pt, [])
            aid = entry["artifact_id"]
            if aid not in artifact_ids_by_modality[pt]:
                artifact_ids_by_modality[pt].append(aid)
        n_artifacts = {pt: len(ids) for pt, ids in artifact_ids_by_modality.items()}

    config = {
        "run_id": run_id,
        "model": model,
        "preset": spec.preset,
        "subset_file": spec.subset_file,
        "subset_sha256": spec.subset_sha256,
        "annotation_file": spec.annotation_file,
        "annotation_sha256": spec.annotation_sha256,
        "artifact_ids_by_modality": artifact_ids_by_modality,
        "n_artifacts_by_modality": n_artifacts,
        "scenarios": spec.scenarios,
        "reference_styles": spec.reference_styles,
        "conditions": spec.conditions,
        "history_depths": spec.history_depths,
        "context_thresholds": spec.context_thresholds,
        "n_cells_per_model": n_cells,
        "n_cells_total": n_cells,
        "git_commit": git_commit,
        "started_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_path = logs_dir / "run_config.json"
    with out_path.open("w") as f:
        json.dump(config, f, indent=2)
    logger.info("Run config written to %s", out_path)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

async def run_trial_set_cell(
    plant_type: str,
    entry: dict,
    history_depth: int,
    ref_style: str,
    cf: str,
    skip_s2: bool,
    scenarios: list[str] | None = None,
    retrieval_model: str | None = None,
    gateway_state_dir: Path | None = None,
    context_threshold: int = 50_000,
) -> list[GroundingTrial]:
    """Run one trial set cell and return scored GroundingTrial records."""
    spec = _manifest_entry_to_spec(entry)

    seed = _query_seed(plant_type, spec.artifact_id, history_depth, ref_style, cf, context_threshold)

    config = TrialConfig(
        plant_type=plant_type,
        history_depth=history_depth,
        reference_style=ref_style,
        seed=seed,
        context_threshold=context_threshold,
        citation_force_condition=cf,
    )

    # Screenshots use ScreenSpot-style action instructions as content_query ("click X", "enter Y"),
    # which are not evaluable by the claim-support Axis 3 scorer. Always use reference queries
    # for screenshots; only chart_image gets grounded content queries.
    if spec.content_query and plant_type != "screenshot":
        query = make_grounded_query(ref_style, plant_type, spec.content_query, seed=seed)
    else:
        query = make_reference_query(ref_style, plant_type, spec.artifact_id, seed=seed)

    try:
        active_scenarios = scenarios or ["S1", "S2", "S3"]
        if skip_s2 and "S2" in active_scenarios:
            active_scenarios = [s for s in active_scenarios if s != "S2"]

        trials = await build_trial(
            config=config,
            plant_prompt=spec.plant_prompt_ai,
            artifact_id=spec.artifact_id,
            artifact_ocr_keywords=spec.artifact_ocr_keywords,
            query=query,
            output_dir=LOGS_DIR,
            image_path=spec.image_path,
            scenarios=active_scenarios,
            retrieval_model=retrieval_model,
            citation_force_condition=cf,
            gateway_state_dir=gateway_state_dir,
            context_threshold=context_threshold,
        )

        run_axis3 = bool(os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
        scored_trials = []
        for trial in trials:
            trial = score_trial(
                trial,
                artifact_repr=trial.actual_artifact_repr or spec.artifact_repr,
                run_axis3=run_axis3,
                n_judges=3,
                output_dir=str(LOGS_DIR),
            )
            scored_trials.append(trial)

        return scored_trials

    except Exception:
        raise


async def run_experiment(
    spec: RunSpec | None = None,
    plant_type_filter: str | None = None,
    dry_run: bool = False,
    skip_s2: bool = False,
    n_trials: int | None = None,
    max_workers: int | None = None,
    retrieval_model: str | None = None,
    run_id: str = "run",
) -> None:
    # Build spec with defaults if not provided (legacy path)
    if spec is None:
        @dataclass
        class _DefaultArgs:
            scenarios: list | None = None
            reference_styles: list | None = None
            conditions: list | None = None
            preset: str | None = None
            subset_file: str | None = None
        spec = _resolve_run_spec(_DefaultArgs())

    cells = _build_matrix(spec, plant_type_filter)
    if n_trials:
        cells = cells[:n_trials]
    total = len(cells)
    active_scenarios = spec.scenarios

    if dry_run:
        print(f"Dry run — {total} trial set cells (scenarios={active_scenarios}, "
              f"styles={spec.reference_styles}, conditions={spec.conditions}, "
              f"depths={spec.history_depths}, thresholds={spec.context_thresholds}):")
        for i, (pt, entry, hd, rs, cf, ct) in enumerate(cells, 1):
            aid = entry["artifact_id"]
            seed = _query_seed(pt, aid, hd, rs, cf, ct)
            print(f"  [{i:5d}/{total}] plant={pt} art={aid} depth={hd:2d} style={rs} cf={cf} ct={ct} seed={seed}")
        return

    # Pre-load artifact pool to avoid lazy-init race under concurrency
    _load_artifact_pool()

    completed = _load_completed_cells(LOGS_DIR)
    logger.info("Found %d already-completed cells; %d remain", len(completed), total - len(completed))

    # Build cell keys for completed lookup
    pending = []
    for cell in cells:
        pt, entry, hd, rs, cf, ct = cell
        key = _build_cell_key(pt, entry["artifact_id"], hd, rs, cf, ct)
        if key not in completed:
            pending.append(cell)

    # Write run_config.json before starting (uses all cells, not just pending)
    _write_run_config(LOGS_DIR, run_id, retrieval_model or VLM_MODEL, spec, cells)

    # One gateway per pending trial by default; --max-workers caps the pool size
    n_workers = len(pending) if max_workers is None else min(max_workers, len(pending))
    if n_workers == 0:
        logger.info("No pending cells — nothing to do")
        return

    # Start the usage-fix proxy (HTTP server only, no global config mutation).
    # Config patches are applied per-worker in _start_worker_gateways, so
    # concurrent experiment_runner processes never race on ~/.openclaw/openclaw.json.
    needs_proxy = any(
        pspec.get("needsUsageProxy") for pspec in EXPERIMENT_MODEL_REGISTRY.values()
    )
    context_window = S2_CONTEXT_THRESHOLD + RESERVE_TOKENS_FLOOR
    proxy_ctx = UsageProxy(patch_global=False) if needs_proxy else None
    if proxy_ctx is not None:
        proxy_ctx.__enter__()
        logger.info("Usage-fix proxy on port %d (worker configs will be patched locally)", proxy_ctx.port)

    proxy_port = proxy_ctx.port if proxy_ctx else None
    worker_state_dirs = _start_worker_gateways(n_workers, proxy_port=proxy_port, context_window=context_window)
    atexit.register(_stop_worker_gateways)

    # Slot pool: each slot maps to its own gateway
    slot_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n_workers):
        await slot_queue.put(i)

    print_lock = asyncio.Lock()
    done_count = [0]

    async def _run_cell(cell: tuple) -> None:
        slot = await slot_queue.get()
        try:
            pt, entry, hd, rs, cf, ct = cell
            state_dir = worker_state_dirs[slot]
            aid = entry["artifact_id"]
            try:
                scored = await run_trial_set_cell(
                    pt, entry, hd, rs, cf, skip_s2=skip_s2,
                    scenarios=active_scenarios, retrieval_model=retrieval_model,
                    gateway_state_dir=state_dir,
                    context_threshold=ct,
                )
                done_count[0] += 1
                r_s1 = next((t.R_path for t in scored if t.scenario == "S1"), None)
                r_s3 = next((t.R_path for t in scored if t.scenario == "S3"), None)
                async with print_lock:
                    print(
                        f"[{done_count[0]:5d}/{total}] plant={pt} art={aid} depth={hd:2d} "
                        f"style={rs} cf={cf} ct={ct} R_s1={r_s1} R_s3={r_s3} [slot={slot}]"
                    )
            except Exception as exc:
                logger.exception("Cell %s failed: %s", (pt, aid, hd, rs, cf, ct), exc)
                _log_error(cell, str(exc))
        finally:
            await slot_queue.put(slot)

    try:
        await asyncio.gather(*[_run_cell(c) for c in pending])
    finally:
        if proxy_ctx is not None:
            proxy_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    from configs.presets import PRESET_NAMES
    parser = argparse.ArgumentParser(description="GroundingBench experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="Print parameter matrix without executing")
    parser.add_argument("--plant-type", choices=PLANT_TYPES, default=None, help="Run only this plant type")
    parser.add_argument("--no-s2", action="store_true", help="Skip S2 compaction condition")
    parser.add_argument(
        "--scenarios", nargs="+", choices=["S1", "S2", "S3"], default=None,
        help="Which scenarios to run (default: from preset or S1 S2 S3)",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Limit to first N trial cells")
    parser.add_argument("--max-workers", type=int, default=None, help="Cap gateway pool size")
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"OpenClaw model string for retrieval agent, e.g. 'shubiaobiao/gpt-4o-mini'. "
             f"Defaults to VLM_MODEL ({VLM_MODEL}) when image_path is present.",
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Human-readable run identifier used to name logs dir and gateway state dirs. "
             "Auto-generated from model name + timestamp if omitted.",
    )
    # Preset + override args
    parser.add_argument(
        "--preset", choices=PRESET_NAMES, default=None,
        help="Named experiment preset (e.g. main_backbone, paper_s2s3_compaction)",
    )
    parser.add_argument(
        "--subset-file", type=str, default=None,
        help="Path to subset config JSON; overrides preset subset_file",
    )
    parser.add_argument(
        "--reference-styles", nargs="+", default=None,
        metavar="STYLE",
        help="Reference styles to use (e.g. definite demonstrative); overrides preset",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        metavar="COND",
        help="CF conditions (e.g. C0 C3); overrides preset",
    )
    parser.add_argument(
        "--annotation-file", type=str, default=None,
        help="Path to annotation CSV for paper runs (recorded in run_config.json)",
    )
    parser.add_argument(
        "--context-thresholds", nargs="+", type=int, default=None,
        metavar="TOKENS",
        help="Context thresholds for S2/S3 compaction (e.g. 25000 50000 75000 100000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # --- Dynamic run identity -------------------------------------------
    _ts = datetime.datetime.utcnow().strftime("%m%d%H%M")
    if args.run_id:
        _run_id = args.run_id
    else:
        _model_slug = (args.model or VLM_MODEL).split("/")[-1].replace(".", "_")
        _preset_slug = f"_{args.preset}" if args.preset else ""
        _run_id = f"{_model_slug}{_preset_slug}_{_ts}"

    # Set module-level LOGS_DIR / ERRORS_LOG based on run_id
    _base_logs = Path(__file__).parent.parent / "logs"
    LOGS_DIR   = _base_logs / f"trials_{_run_id}"
    ERRORS_LOG = _base_logs / f"errors_{_run_id}.jsonl"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Patch into the module scope so helpers (score_trial, _log_error, etc.) see it
    import harness.experiment_runner as _self
    _self.LOGS_DIR   = LOGS_DIR
    _self.ERRORS_LOG = ERRORS_LOG

    logger.info("Run ID: %s  |  Logs: %s", _run_id, LOGS_DIR)

    # Resolve RunSpec from preset + CLI args
    _spec = _resolve_run_spec(args)
    logger.info(
        "RunSpec: preset=%s  scenarios=%s  styles=%s  conditions=%s  depths=%s  thresholds=%s  subset=%s",
        _spec.preset, _spec.scenarios, _spec.reference_styles,
        _spec.conditions, _spec.history_depths, _spec.context_thresholds, _spec.subset_file,
    )

    if not args.dry_run:
        def _on_signal(sig, frame):
            _stop_worker_gateways()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _on_signal)
        signal.signal(signal.SIGHUP, _on_signal)
        # SIGINT → KeyboardInterrupt → atexit handles it

    asyncio.run(run_experiment(
        spec=_spec,
        plant_type_filter=args.plant_type,
        dry_run=args.dry_run,
        skip_s2=args.no_s2,
        n_trials=args.n_trials,
        max_workers=args.max_workers,
        retrieval_model=args.model or VLM_MODEL,
        run_id=_run_id,
    ))

    if not args.dry_run:
        import subprocess as _sp
        import sys as _sys
        logger.info("Running eval summary...")
        _sp.run([_sys.executable, "results/eval_summary.py", "--trials-dir", str(LOGS_DIR)], check=False)
