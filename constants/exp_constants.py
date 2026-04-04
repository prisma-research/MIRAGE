"""Experiment parameters, R-path enumerations, and scoring thresholds."""

# ---------------------------------------------------------------------------
# Experiment parameter matrix
# ---------------------------------------------------------------------------

PLANT_TYPES    = ["screenshot", "chart_image"]
N_PER_TYPE     = 60     # artifacts per plant_type (easy / medium / hard stratified, 20 each)
HISTORY_DEPTHS = [0, 16_000, 32_000]                      # S1 only: token targets (0 = no filler)
REF_STYLES     = ["definite", "demonstrative", "temporal"]
CF_CONDITIONS  = ["C0", "C3", "Cm"]   # main experiment; C1/C2 restricted to Subset C diagnostic

# ---------------------------------------------------------------------------
# Scenario / R-path enumerations
# ---------------------------------------------------------------------------

SCENARIOS     = ["S1", "S2", "S3"]
VALID_R_PATHS = frozenset({"R_context", "heuristic_R_context", "R_boot", "R_tool", "R_none"})

# Snapshot tag written into every GroundingTrial record
SNAPSHOT_ID = "baseline_v1"

# ---------------------------------------------------------------------------
# S2 / S3 compaction control
# ---------------------------------------------------------------------------

S2_CONTEXT_THRESHOLD   = 50_000   # token estimate at which filler injection stops (~1-3 turns)
COMPACTION_TRIGGER_MAX = 5        # max summarization prompts sent to force compaction
COMPACTION_POLL_TIMEOUT = 120.0   # seconds; wall-clock budget for force_compaction_and_flush

# ---------------------------------------------------------------------------
# Axis 1 — R-path classifier
# ---------------------------------------------------------------------------

RETRIEVAL_TOOL_NAMES = frozenset({
    "memory_search",
    "artifact_recall",
    "memory_get",
    "memory_recall",
})

# Generic file-read paths that indicate memory/workspace access (OpenClaw "read" tool)
MEMORY_PATH_PATTERNS = frozenset({
    "MEMORY.md",
    "memory/",
    "BOOTSTRAP.md",
    "SOUL.md",
    "USER.md",
    "IDENTITY.md",
})

# ---------------------------------------------------------------------------
# Axis 3 — grounding aggregator thresholds
# (to be calibrated on 120-trial human set)
# ---------------------------------------------------------------------------

DEFAULT_ALPHA     = 0.5   # weight for critical claims in Γ
DEFAULT_BETA      = 0.5   # weight for non-critical claims in Γ
DEFAULT_LAMBDA    = 0.7   # blend parameter (critical vs. overall support)
DEFAULT_TAU_SUP   = 0.7   # Γ threshold → S_hat = 1.0
DEFAULT_TAU_UNSUP = 0.3   # Γ threshold → S_hat = 0.0

# ---------------------------------------------------------------------------
# OpenClaw native compaction
# ---------------------------------------------------------------------------

RESERVE_TOKENS_FLOOR = 20_000  # OpenClaw default, set explicitly for determinism
