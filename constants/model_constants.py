"""Model provider configuration and active-provider switch."""

import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Volcengine (ByteDance Ark) — VLM catalog
# API base: https://ark.cn-beijing.volces.com/api/v3
# Source: logs/volcengine_models.md (tested 2026-03-14)
# ---------------------------------------------------------------------------

class Volcengine:
    # Doubao 1.5 Vision (128K ctx)
    THINKING_VISION_PRO = "doubao-1-5-thinking-vision-pro-250428"  # VLM + deep reasoning (active default)
    VISION_PRO          = "doubao-1.5-vision-pro-250328"           # VLM Pro
    VISION_PRO_32K      = "doubao-1-5-vision-pro-32k-250115"       # VLM Pro 32K
    VISION_LITE         = "doubao-1.5-vision-lite-250315"          # VLM Lite
    UI_TARS             = "doubao-1.5-ui-tars-250328"              # GUI / screen automation

    # Seed 1.6 Vision (256K ctx)
    SEED_1_6_VISION     = "doubao-seed-1-6-vision-250815"          # dedicated visual reasoning + GUI agent

    # Qwen (text-only, for retrieval model slot)
    QWEN3_8B            = "qwen3-8b-20250429"                      # ✅ confirmed on Volcengine 2026-03-14

    # Legacy
    VISION_PRO_LEGACY   = "doubao-vision-pro-32k-241028"
    VISION_LITE_LEGACY  = "doubao-vision-lite-32k-241015"


# ---------------------------------------------------------------------------
# Shubiaobiao — VLM catalog
# Proxy: https://api.shubiaobiao.cn/v1  (288 models, updated 2026-03-17)
# Source: logs/shubiaobiao_models.md
# ✅ = confirmed working in VLM test; others listed in VLM support table
# ---------------------------------------------------------------------------

class Shubiaobiao:
    # ── Anthropic / Claude (all Claude models listed support vision) ───────
    CLAUDE_SONNET_4_6   = "claude-sonnet-4-6"                      # ✅
    CLAUDE_SONNET_4_5   = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4_0   = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_6     = "claude-opus-4-6"
    CLAUDE_OPUS_4_5     = "claude-opus-4-5-20251101"
    CLAUDE_OPUS_4_0     = "claude-opus-4-20250514"
    CLAUDE_HAIKU_4_5    = "claude-haiku-4-5-20251001"

    # ── OpenAI (vision-capable subset) ───────────────────────────────────
    GPT_5               = "gpt-5"
    GPT_5_MINI          = "gpt-5-mini"
    GPT_4O              = "gpt-4o"                                  # ✅
    GPT_4O_MINI         = "gpt-4o-mini"                             # ✅
    GPT_4_1             = "gpt-4.1"
    GPT_4_1_MINI        = "gpt-4.1-mini"
    GPT_4_1_NANO        = "gpt-4.1-nano"
    GPT_4_TURBO         = "gpt-4-turbo"
    O1                  = "o1"                                      # reasoning VLM
    O3                  = "o3"                                      # reasoning VLM
    O4_MINI             = "o4-mini"                                 # reasoning VLM

    # ── Google Gemini (all support image + video) ─────────────────────────
    GEMINI_3_PRO        = "gemini-3-pro-preview"
    GEMINI_3_FLASH      = "gemini-3-flash-preview"
    GEMINI_25_PRO       = "gemini-2.5-pro"
    GEMINI_25_FLASH     = "gemini-2.5-flash"
    GEMINI_25_FLASH_NT  = "gemini-2.5-flash-nothinking"             # ✅ (safe for non-streaming)
    GEMINI_25_FLASH_T   = "gemini-2.5-flash-thinking"               # use streaming mode
    GEMINI_20_FLASH     = "gemini-2.0-flash"
    GEMINI_20_FLASH_LITE = "gemini-2.0-flash-lite"

    # ── Qwen (vision-language variants only) ─────────────────────────────
    QWEN_25_VL_72B      = "qwen2.5-vl-72b-instruct"                # ✅
    QWEN_3_VL_235B      = "qwen3-vl-235b-a22b-instruct"
    QWEN_3_VL_32B       = "qwen3-vl-32b-instruct"                  # ✅
    QWEN_3_VL_PLUS      = "qwen3-vl-plus"

    # ── DeepSeek ─────────────────────────────────────────────────────────
    DEEPSEEK_CHAT       = "deepseek-chat"                           # ✅ (supports image input)

    # ── GLM / 智谱 ────────────────────────────────────────────────────────
    GLM_4_6V            = "glm-4.6v"                               # ✅ (needs ≥256px images; glm-4.5v dead)

    # ── Kimi / 月之暗面 ────────────────────────────────────────────────────
    KIMI_K2_5           = "kimi-k2.5"

    # ── Grok / xAI ────────────────────────────────────────────────────────
    GROK_2_IMAGE        = "grok-2-image"


# ---------------------------------------------------------------------------
# Provider configs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    openclaw_id: str   # provider name as registered in OpenClaw
    base_url: str      # OpenAI-compatible API base URL
    api_key_env: str   # env var name holding the API key
    vlm_model: str     # model ID for VLM / image tasks
    text_model: str    # model ID for Axis 3 plain-text scoring


PROVIDER_VOLCENGINE = ProviderConfig(
    openclaw_id="volcengine",
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key_env="VOLCENGINE",
    vlm_model=Volcengine.THINKING_VISION_PRO,
    text_model="doubao-seed-2-0-pro-260215",  # best text model (not in VLM catalog)
)

PROVIDER_SHUBIAOBIAO = ProviderConfig(
    openclaw_id="shubiaobiao",
    base_url="https://api.shubiaobiao.cn/v1",
    api_key_env="SHUBIOABIAO",               # matches existing .env key name
    vlm_model=Shubiaobiao.GPT_4O_MINI,
    text_model=Shubiaobiao.CLAUDE_SONNET_4_6,
)

PROVIDER_ANTHROPIC = ProviderConfig(
    openclaw_id="anthropic",
    base_url="https://api.anthropic.com",
    api_key_env="ANTHROPIC_API_KEY",
    vlm_model=Shubiaobiao.CLAUDE_HAIKU_4_5,  # claude-haiku-4-5-20251001
    text_model=Shubiaobiao.CLAUDE_SONNET_4_6,
)

# ── Single switch point ───────────────────────────────────────────────────────
# Change ACTIVE_PROVIDER to swap all model usage at once.
ACTIVE_PROVIDER: ProviderConfig = PROVIDER_ANTHROPIC

# ── Derived constants (consumed by the rest of the codebase) ─────────────────

# OpenClaw VLM planting model — format "openclaw_id/model_id"
VLM_MODEL: str = f"{ACTIVE_PROVIDER.openclaw_id}/{ACTIVE_PROVIDER.vlm_model}"

# Direct API config for Axis 3 LLM scoring.
# Env vars (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL) override if explicitly set.
LLM_API_KEY: str = (
    os.getenv("LLM_API_KEY")
    or os.getenv(ACTIVE_PROVIDER.api_key_env, "")
)
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL") or ACTIVE_PROVIDER.base_url
LLM_MODEL: str    = os.getenv("LLM_MODEL") or ACTIVE_PROVIDER.text_model

# ---------------------------------------------------------------------------
# Local providers — GPU machine vllm-served endpoints
# ---------------------------------------------------------------------------

PROVIDER_LOCAL_QWEN30B = ProviderConfig(
    openclaw_id="local_qwen30b",
    base_url=os.getenv("LOCAL_QWEN30B_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="qwen3-vl-30b-instruct",
    text_model="qwen3-vl-30b-instruct",
)

PROVIDER_LOCAL_QWEN8B = ProviderConfig(
    openclaw_id="local_qwen8b",
    base_url=os.getenv("LOCAL_QWEN8B_BASE_URL", "http://127.0.0.1:8001/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="qwen3-vl-8b-instruct",
    text_model="qwen3-vl-8b-instruct",
)

PROVIDER_LOCAL_QWEN4B = ProviderConfig(
    openclaw_id="local_qwen4b",
    base_url=os.getenv("LOCAL_QWEN4B_BASE_URL", "http://127.0.0.1:8002/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="qwen3-vl-4b-instruct",
    text_model="qwen3-vl-4b-instruct",
)

PROVIDER_LOCAL_GEMMA12B = ProviderConfig(
    openclaw_id="local_gemma12b",
    base_url=os.getenv("LOCAL_GEMMA12B_BASE_URL", "http://127.0.0.1:8003/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="gemma-3-12b-it",
    text_model="gemma-3-12b-it",
)

PROVIDER_LOCAL_GEMMA27B = ProviderConfig(
    openclaw_id="local_gemma27b",
    base_url=os.getenv("LOCAL_GEMMA27B_BASE_URL", "http://127.0.0.1:8004/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="gemma-3-27b-it",
    text_model="gemma-3-27b-it",
)

PROVIDER_LOCAL_GEMMA27B = ProviderConfig(
    openclaw_id="local_gemma27b",
    base_url=os.getenv("LOCAL_GEMMA27B_BASE_URL", "http://127.0.0.1:8004/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="gemma-3-27b-it",
    text_model="gemma-3-27b-it",
)

PROVIDER_LOCAL_INTERNVL14B = ProviderConfig(
    openclaw_id="local_internvl14b",
    base_url=os.getenv("LOCAL_INTERNVL14B_BASE_URL", "http://127.0.0.1:8005/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="internvl3_5-14b-instruct",
    text_model="internvl3_5-14b-instruct",
)

PROVIDER_LOCAL_INTERNVL20B = ProviderConfig(
    openclaw_id="local_internvl20b",
    base_url=os.getenv("LOCAL_INTERNVL20B_BASE_URL", "http://127.0.0.1:8006/v1"),
    api_key_env="LOCAL_VLM_API_KEY",
    vlm_model="internvl3_5-20b-a4b",
    text_model="internvl3_5-20b-a4b",
)

# ---------------------------------------------------------------------------
# Paper model groups (use provider-qualified strings for --model CLI arg)
# ---------------------------------------------------------------------------

PAPER_CORE_MODELS = [
    f"shubiaobiao/{Shubiaobiao.GPT_5}",
    f"shubiaobiao/{Shubiaobiao.QWEN_3_VL_32B}",
    f"shubiaobiao/{Shubiaobiao.CLAUDE_SONNET_4_6}",
    f"shubiaobiao/{Shubiaobiao.GEMINI_25_FLASH_NT}",
    f"volcengine/{Volcengine.VISION_PRO}",
]

PAPER_OPTIONAL_MODELS = [
    f"shubiaobiao/{Shubiaobiao.GLM_4_6V}",
]

# ---------------------------------------------------------------------------
# Experiment model registry — self-contained per-agent models.json generation
# ---------------------------------------------------------------------------

EXPERIMENT_MODEL_REGISTRY: dict[str, dict] = {
    "anthropic": {
        "baseUrl": "https://api.anthropic.com",
        "needsUsageProxy": False,
        "apiKeyEnvs": ["ANTHROPIC_API_KEY"],
        "api": "anthropic-messages",
        "models": [
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "input": ["text", "image"], "maxTokens": 64000},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "input": ["text", "image"], "maxTokens": 64000},
        ],
    },
    "shubiaobiao": {
        "baseUrl": "https://api.shubiaobiao.cn/v1",
        # needsUsageProxy: shubiaobiao returns misleading input_tokens: 0 alongside
        # correct prompt_tokens.  The local UsageProxy strips the bad fields so
        # OpenClaw's cascade reaches prompt_tokens.  When proxy_port is supplied to
        # _setup_trial_agent, baseUrl is overridden to http://127.0.0.1:{port}/v1.
        "needsUsageProxy": True,
        "apiKeyEnvs": ["SHUBIOABIAO", "SHUBIAOBIAO_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "gpt-5", "name": "GPT-5", "input": ["text", "image"], "maxTokens": 16384},
            {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5", "input": ["text", "image"], "maxTokens": 8192},
            {"id": "qwen3-vl-32b-instruct", "name": "Qwen3-VL-32B", "input": ["text", "image"], "maxTokens": 8192},
            {"id": "claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "input": ["text", "image"], "maxTokens": 16384},
            {"id": "gemini-2.5-flash-nothinking", "name": "Gemini 2.5 Flash NT", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "volcengine": {
        "baseUrl": "https://ark.cn-beijing.volces.com/api/v3",
        "apiKeyEnvs": ["VOLCENGINE", "VOLCENGINE_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "doubao-1.5-vision-pro-250328", "name": "Doubao 1.5 Vision Pro", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_qwen30b": {
        "baseUrl": os.getenv("LOCAL_QWEN30B_BASE_URL", "http://127.0.0.1:8000/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "qwen3-vl-30b-instruct", "name": "Qwen3-VL-30B (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_qwen8b": {
        "baseUrl": os.getenv("LOCAL_QWEN8B_BASE_URL", "http://127.0.0.1:8001/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "qwen3-vl-8b-instruct", "name": "Qwen3-VL-8B (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_qwen4b": {
        "baseUrl": os.getenv("LOCAL_QWEN4B_BASE_URL", "http://127.0.0.1:8002/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "qwen3-vl-4b-instruct", "name": "Qwen3-VL-4B (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_gemma12b": {
        "baseUrl": os.getenv("LOCAL_GEMMA12B_BASE_URL", "http://127.0.0.1:8003/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "gemma-3-12b-it", "name": "Gemma 3 12B IT (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_gemma27b": {
        "baseUrl": os.getenv("LOCAL_GEMMA27B_BASE_URL", "http://127.0.0.1:8004/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "gemma-3-27b-it", "name": "Gemma 3 27B IT (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_internvl14b": {
        "baseUrl": os.getenv("LOCAL_INTERNVL14B_BASE_URL", "http://127.0.0.1:8005/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "internvl3_5-14b-instruct", "name": "InternVL3.5-14B (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
    "local_internvl20b": {
        "baseUrl": os.getenv("LOCAL_INTERNVL20B_BASE_URL", "http://127.0.0.1:8006/v1"),
        "needsUsageProxy": True,
        "apiKeyEnvs": ["LOCAL_VLM_API_KEY", "OPENAI_API_KEY"],
        "api": "openai-completions",
        "models": [
            {"id": "internvl3_5-20b-a4b", "name": "InternVL3.5-20B-A4B (local)", "input": ["text", "image"], "maxTokens": 8192},
        ],
    },
}
