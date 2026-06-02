"""
Local reverse proxy + openclaw.json config patcher for GroundingBench experiments.

Handles two categories of problems when running experiments through non-standard
OpenAI-compatible endpoints (e.g. shubiaobiao):

HTTP proxy (request destreaming + usage fix):
  1. Destream: convert ``stream: true`` requests to non-streaming so the upstream
     returns a JSON response with ``prompt_tokens``/``completion_tokens``.
     Re-emit the JSON as SSE chunks so OpenClaw's streaming parser can consume it.
     (Shubiaobiao doesn't support ``stream_options: {include_usage: true}``.)
  2. Usage fix: overwrite misleading zero-valued ``input_tokens``/``output_tokens``
     with the correct values from ``prompt_tokens``/``completion_tokens``.
  3. Gzip: decompress gzip-encoded responses transparently.

openclaw.json config patching (via UsageProxy context manager):
  4. Proxy baseUrl: redirect shubiaobiao traffic through the local proxy.
  5. Experiment model merge: register models from EXPERIMENT_MODEL_REGISTRY
     (e.g. gpt-5) that may be missing from the user's global config.
  6. contextWindow override: set contextWindow on all models so OpenClaw fires
     compaction at the benchmark target threshold.
  7. compaction.model override: route compaction summarization through a non-Azure
     model (default: deepseek-chat) to avoid Azure content filter 400 errors.
  All patches are restored on exit (with atexit/SIGTERM safety net).

Usage:
    # Recommended — dynamic port, automatic lifecycle:
    from client.usage_proxy import UsageProxy
    with UsageProxy(context_window_override=70000) as proxy:
        print(proxy.port)  # OS-assigned free port
        # ... run experiments (openclaw.json is patched) ...
    # proxy shut down, openclaw.json restored

    # Standalone (for debugging):
    python -m client.usage_proxy [--port PORT]
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

UPSTREAM_BASE = "https://api.shubiaobiao.cn"

# Model-to-upstream routing table.  Built lazily from EXPERIMENT_MODEL_REGISTRY
# for providers with needsUsageProxy=True.  Maps model ID → upstream base URL
# (without trailing /v1).  Requests whose model isn't in the table fall back to
# UPSTREAM_BASE.
_MODEL_UPSTREAM_ROUTES: dict[str, str] | None = None


def _get_model_upstream_routes() -> dict[str, str]:
    """Build model→upstream mapping from EXPERIMENT_MODEL_REGISTRY (cached)."""
    global _MODEL_UPSTREAM_ROUTES
    if _MODEL_UPSTREAM_ROUTES is not None:
        return _MODEL_UPSTREAM_ROUTES
    from constants import EXPERIMENT_MODEL_REGISTRY
    routes: dict[str, str] = {}
    for pid, reg in EXPERIMENT_MODEL_REGISTRY.items():
        if not reg.get("needsUsageProxy"):
            continue
        base = reg["baseUrl"].rstrip("/")
        # Strip trailing /v1 — we'll re-add the request path
        if base.endswith("/v1"):
            base = base[:-3]
        for m in reg.get("models", []):
            routes[m["id"]] = base
    _MODEL_UPSTREAM_ROUTES = routes
    return routes


# Fields that the shubiaobiao proxy populates incorrectly (always 0)
# while the standard prompt_tokens/completion_tokens carry the real values.
MISLEADING_USAGE_FIELDS = {"input_tokens", "output_tokens", "input_tokens_details"}


def _strip_misleading_usage(body: dict) -> dict:
    """Normalise usage fields so OpenClaw's cascade always finds correct values.

    Handles two cases:
      1. Shubiaobiao: sends input_tokens: 0 alongside correct prompt_tokens.
         → overwrite input_tokens with prompt_tokens.
      2. Local vllm: only sends prompt_tokens/completion_tokens (no input_tokens).
         → inject input_tokens/output_tokens from prompt_tokens/completion_tokens.

    OpenClaw's usage cascade checks input_tokens → prompt_tokens in order.
    By ensuring input_tokens is always populated, we guarantee correct counting.
    """
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        # Ensure input_tokens is present and correct
        if "input_tokens" not in usage and prompt:
            usage["input_tokens"] = prompt
        elif usage.get("input_tokens") is not None and usage["input_tokens"] == 0 and prompt:
            usage["input_tokens"] = prompt

        # Ensure output_tokens is present and correct
        if "output_tokens" not in usage and completion:
            usage["output_tokens"] = completion
        elif usage.get("output_tokens") is not None and usage["output_tokens"] == 0 and completion:
            usage["output_tokens"] = completion

        # Remove details fields that are always null/zero
        val = usage.get("input_tokens_details")
        if val is None or val == 0:
            usage.pop("input_tokens_details", None)
    return body


# Doubao / GLM emit tool calls as native special-token text rather than as a
# structured `tool_calls` field when called via the generic openai-completions
# path (no `tools` param). OpenClaw can't parse that text, so it sees no tool
# call (→ R_context). This converts those native tokens into a structured
# `tool_calls` array so the rest of the pipeline (and OpenClaw) treats it as a
# real tool call (→ R_tool). VLM stays a VLM; only the tool-call wire format is
# normalized — implemented in our proxy, analogous to vLLM's --tool-call-parser.
import re as _re_tc

# Doubao emits the markers two ways: clean `<|FunctionCallEnd|>` and, frequently,
# an XML-close-style slashed variant `</|FunctionCallEnd|>`. Both Begin/End and the
# slash are optional so the adapter lifts either form (otherwise the slashed calls
# leak as text → R_context).
_DOUBAO_FC_RE = _re_tc.compile(
    r"</?\|FunctionCall(?:Begin)?\|>\s*(.*?)\s*</?\|FunctionCallEnd\|>",
    _re_tc.DOTALL,
)
_DOUBAO_FC_END_RE = _re_tc.compile(r"</?\|FunctionCallEnd\|>")
_DOUBAO_FC_ANY_RE = _re_tc.compile(r"</?\|FunctionCall(?:Begin|End)?\|>")


def _coerce_tool_call_objs(raw: str) -> list[dict]:
    """Parse the JSON payload inside a FunctionCall block into OpenAI tool_calls."""
    raw = raw.strip()
    objs = []
    try:
        parsed = json.loads(raw)
        objs = parsed if isinstance(parsed, list) else [parsed]
    except (json.JSONDecodeError, ValueError):
        # Fallback: name(args) python-ish or name + trailing json
        m = _re_tc.search(r'"?name"?\s*[:=]\s*"?([\w./-]+)"?', raw)
        if not m:
            return []
        argm = _re_tc.search(r'\{.*\}', raw, _re_tc.DOTALL)
        try:
            args = json.loads(argm.group(0)) if argm else {}
        except (json.JSONDecodeError, ValueError):
            args = {}
        objs = [{"name": m.group(1), "arguments": args}]
    out = []
    for i, o in enumerate(objs):
        if not isinstance(o, dict):
            continue
        name = o.get("name") or o.get("tool")
        if not name:
            continue
        args = o.get("arguments")
        if args is None:
            args = o.get("parameters", {})
        out.append({
            "id": f"call_doubao_{i}",
            "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)},
        })
    return out


def _normalize_native_tool_calls(data: dict) -> dict:
    """If a choice's message content carries native FunctionCall tokens, lift
    them into a structured `tool_calls` field and strip them from the text."""
    for choice in data.get("choices", []):
        msg = choice.get("message")
        if not isinstance(msg, dict) or msg.get("tool_calls"):
            continue
        content = msg.get("content")
        # Guard on the marker stem only ("FunctionCall"), so the slashed END-only
        # variant "</|FunctionCallEnd|>" (which lacks the literal "<|FunctionCall")
        # is still detected.
        if not isinstance(content, str) or "FunctionCall" not in content:
            continue
        blocks = _DOUBAO_FC_RE.findall(content)
        if not blocks:
            # Doubao sometimes emits only the END marker: "<text>{json}<|FunctionCallEnd|>"
            # (clean or slashed). Take the trailing JSON object before each End marker.
            for seg in _DOUBAO_FC_END_RE.split(content)[:-1]:
                jm = _re_tc.search(r'\{.*\}\s*$', seg.strip(), _re_tc.DOTALL)
                if jm:
                    blocks.append(jm.group(0))
        logger.info("Proxy tool-parse: found %d FunctionCall block(s); raw[:160]=%r",
                    len(blocks), content[:160])
        tool_calls = []
        for b in blocks:
            tool_calls.extend(_coerce_tool_call_objs(b))
        if tool_calls:
            msg["tool_calls"] = tool_calls
            # Strip both paired blocks and any leftover bare markers (END-only case).
            stripped = _DOUBAO_FC_RE.sub("", content)
            stripped = _DOUBAO_FC_ANY_RE.sub("", stripped).strip()
            msg["content"] = stripped or None
            choice["finish_reason"] = "tool_calls"
            logger.info("Proxy tool-parse: lifted %d tool_call(s): %s",
                        len(tool_calls), [t["function"]["name"] for t in tool_calls])
    return data


def _json_to_sse(data: dict) -> bytes:
    """Convert a non-streaming JSON chat completion response to SSE format.

    The OpenAI SDK expects streamed chunks with ``delta`` instead of ``message``.
    We emit:
      1. One content chunk per choice (with the full text in delta)
      2. One finish chunk per choice (with finish_reason)
      3. One usage-only chunk (choices=[], usage={...})
      4. ``data: [DONE]``
    """
    base = {
        "id": data.get("id", ""),
        "object": "chat.completion.chunk",
        "created": data.get("created", 0),
        "model": data.get("model", ""),
        "system_fingerprint": data.get("system_fingerprint"),
    }

    lines: list[bytes] = []
    for choice in (data.get("choices") or []):
        msg = choice.get("message") or {}
        # Content chunk
        content_chunk = {
            **base,
            "choices": [{
                "index": choice.get("index", 0),
                "delta": {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                },
                "finish_reason": None,
            }],
        }
        lines.append(b"data: " + json.dumps(content_chunk).encode())

        # Tool calls (vLLM emits "tool_calls": null when none → guard against None)
        for tc in (msg.get("tool_calls") or []):
            tc_chunk = {
                **base,
                "choices": [{
                    "index": choice.get("index", 0),
                    "delta": {"tool_calls": [tc]},
                    "finish_reason": None,
                }],
            }
            lines.append(b"data: " + json.dumps(tc_chunk).encode())

        # Finish chunk
        finish_chunk = {
            **base,
            "choices": [{
                "index": choice.get("index", 0),
                "delta": {},
                "finish_reason": choice.get("finish_reason", "stop"),
            }],
        }
        lines.append(b"data: " + json.dumps(finish_chunk).encode())

    # Usage chunk (empty choices, usage object)
    usage = data.get("usage")
    if usage:
        usage_chunk = {**base, "choices": [], "usage": usage}
        lines.append(b"data: " + json.dumps(usage_chunk).encode())

    lines.append(b"data: [DONE]")
    lines.append(b"")
    return b"\n\n".join(lines)


def _strip_usage_from_sse(body: bytes) -> bytes:
    """Strip misleading usage fields from SSE ``data:`` chunks.

    OpenClaw may request ``stream: true``, producing a ``text/event-stream``
    response.  Usage appears in the final chunk; we strip misleading fields
    from every chunk that contains a ``usage`` object.
    """
    lines = body.split(b"\n")
    result: list[bytes] = []
    for line in lines:
        if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                # Log raw usage before fix for diagnostics
                raw_usage = chunk.get("usage")
                if raw_usage and isinstance(raw_usage, dict):
                    logger.info("SSE raw usage BEFORE fix: %s", json.dumps(raw_usage))
                chunk = _strip_misleading_usage(chunk)
                # Log fixed usage
                fixed_usage = chunk.get("usage")
                if fixed_usage and isinstance(fixed_usage, dict):
                    logger.info("SSE raw usage AFTER  fix: %s", json.dumps(fixed_usage))
                line = b"data: " + json.dumps(chunk).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        result.append(line)
    return b"\n".join(result)


class ProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to upstream, strip misleading usage on return."""

    def _resolve_upstream(self, request_body: bytes) -> str:
        """Pick upstream base URL from the model field in the request body."""
        try:
            model = json.loads(request_body).get("model")
            if model:
                routes = _get_model_upstream_routes()
                upstream = routes.get(model)
                if upstream:
                    return upstream
        except (json.JSONDecodeError, ValueError):
            pass
        return UPSTREAM_BASE

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(content_length) if content_length > 0 else b""

        upstream_base = self._resolve_upstream(request_body)
        # Path normalization: OpenClaw calls us at /v1/chat/completions, but some
        # upstreams (e.g. Volcengine Ark) live under /api/v3, not /v1. If the
        # upstream base already carries its own version path, drop our /v1 prefix.
        # EXCEPTION: local vLLM servers (127.0.0.1/localhost) serve under /v1 even
        # though the route table strips the trailing /v1 from their base, so keep
        # the /v1 prefix for them (else we'd hit /chat/completions -> 404).
        req_path = self.path
        _is_local = upstream_base.startswith(("http://127.0.0.1", "http://localhost"))
        if (not _is_local
                and not upstream_base.rstrip("/").endswith("/v1")
                and req_path.startswith("/v1/")):
            req_path = req_path[3:]  # "/v1/chat/completions" -> "/chat/completions"
        upstream_url = f"{upstream_base}{req_path}"

        # Forward all headers except Host
        headers = {}
        for key in self.headers:
            if key.lower() != "host":
                headers[key] = self.headers[key]

        # Destream: convert streaming requests to non-streaming so we get usage
        # in the JSON response, then re-emit as SSE.
        #
        # OpenClaw's agent SDK sets supportsUsageInStreaming=false for non-standard
        # endpoints, so it omits stream_options.  Both shubiaobiao and local vllm
        # need this destreaming treatment to get usage data back to OpenClaw.
        was_streaming = False
        try:
            req_json = json.loads(request_body)
            dirty = False
            # Sanitize null message content: OpenAI allows content:null on
            # tool-call-only assistant turns, but the swift/Qwen3-VL chat template
            # iterates content and raises "'NoneType' object is not iterable" (400).
            # Replayed planting history (esp. shallow states like d0) carries such
            # turns, so coerce null content -> "".
            msgs = req_json.get("messages")
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict) and m.get("content") is None:
                        m["content"] = ""
                        dirty = True
            if req_json.get("stream"):
                was_streaming = True
                req_json["stream"] = False
                req_json.pop("stream_options", None)
                dirty = True
                logger.info(
                    "Proxy destream: model=%s upstream=%s",
                    req_json.get("model"), upstream_base,
                )
            if dirty:
                request_body = json.dumps(req_json).encode()
        except (json.JSONDecodeError, ValueError):
            pass

        # Update Content-Length after body modification
        headers["Content-Length"] = str(len(request_body))
        req = Request(upstream_url, data=request_body, headers=headers, method="POST")

        try:
            resp = urlopen(req, timeout=300)
            resp_body = resp.read()
            status = resp.status
            resp_headers = dict(resp.headers)
        except HTTPError as e:
            resp_body = e.read()
            status = e.code
            resp_headers = dict(e.headers)
            if status >= 400:
                logger.warning(
                    "Proxy upstream %d: %s upstream=%s (first 500 chars: %s)",
                    status, self.path, upstream_base, resp_body[:500],
                )

        # Decompress gzip if needed (OpenAI SDK sends Accept-Encoding: gzip).
        # Case-insensitive header lookup — Volcengine/Ark returns lowercase
        # `content-encoding: gzip`, which a case-sensitive .get() would miss.
        _enc_key = next((k for k in resp_headers if k.lower() == "content-encoding"), None)
        if _enc_key and resp_headers.get(_enc_key, "").lower() == "gzip":
            resp_body = gzip.decompress(resp_body)
            del resp_headers[_enc_key]

        # Normalize header lookup (vllm returns lowercase, shubiaobiao mixed case)
        content_type = ""
        for k, v in resp_headers.items():
            if k.lower() == "content-type":
                content_type = v
                break
        if was_streaming:
            logger.info(
                "Proxy destream response: status=%d content_type=%r was_streaming=%s",
                status, content_type, was_streaming,
            )

        if was_streaming and status == 200 and "application/json" in content_type:
            # Convert JSON response to SSE format for the OpenAI SDK.
            try:
                data = json.loads(resp_body)
                data = _strip_misleading_usage(data)
                data = _normalize_native_tool_calls(data)  # Doubao/GLM native tool tokens → structured tool_calls
                usage = data.get("usage")

                resp_body = _json_to_sse(data)
                resp_headers["Content-Type"] = "text/event-stream"

                logger.info(
                    "Proxy destream→SSE: %s prompt_tokens=%s completion_tokens=%s body_len=%d",
                    self.path,
                    (usage or {}).get("prompt_tokens"),
                    (usage or {}).get("completion_tokens"),
                    len(resp_body),
                )
            except Exception:
                logger.exception("Proxy destream→SSE conversion failed")
        elif "text/event-stream" in content_type:
            resp_body = _strip_usage_from_sse(resp_body)
        elif "application/json" in content_type:
            try:
                data = json.loads(resp_body)
                data = _strip_misleading_usage(data)
                resp_body = json.dumps(data).encode()
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        self.send_response(status)
        for key, val in resp_headers.items():
            if key.lower() not in ("transfer-encoding", "content-length", "content-encoding"):
                self.send_header(key, val)
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)

    def do_GET(self):
        # Forward GET requests (e.g. /v1/models) transparently.
        # GET has no body, so we can't route by model — use default upstream.
        upstream_url = f"{UPSTREAM_BASE}{self.path}"
        headers = {}
        for key in self.headers:
            if key.lower() != "host":
                headers[key] = self.headers[key]
        req = Request(upstream_url, headers=headers, method="GET")
        try:
            resp = urlopen(req, timeout=30)
            resp_body = resp.read()
            self.send_response(resp.status)
            for key, val in resp.headers.items():
                if key.lower() not in ("transfer-encoding", "content-length", "content-encoding"):
                    self.send_header(key, val)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


# ---------------------------------------------------------------------------
# Config patching — pure function, operates on a config dict in memory.
# Used by experiment_runner to patch per-worker configs without touching
# the global ~/.openclaw/openclaw.json, eliminating inter-process races.
# ---------------------------------------------------------------------------

# Provider IDs whose baseUrl should be routed through the local proxy.
# Built dynamically: all providers with needsUsageProxy=True in the registry.
def _proxy_provider_ids() -> tuple[str, ...]:
    from constants import EXPERIMENT_MODEL_REGISTRY
    return tuple(
        pid for pid, reg in EXPERIMENT_MODEL_REGISTRY.items()
        if reg.get("needsUsageProxy")
    )

PROXY_PROVIDERS = _proxy_provider_ids()

# Default compaction summarization model — text-only, no Azure content filter.
# Override with OPENCLAW_COMPACTION_MODEL env var for local serving.
DEFAULT_COMPACTION_MODEL = (
    os.environ.get("OPENCLAW_COMPACTION_MODEL") or "deepseek/deepseek-chat"
)


def patch_openclaw_config(
    cfg: dict,
    *,
    proxy_port: int | None = None,
    context_window_override: int | None = None,
    compaction_model: str | None = DEFAULT_COMPACTION_MODEL,
) -> dict:
    """Apply experiment patches to an openclaw config dict (in-place + returned).

    This is a pure function that mutates and returns ``cfg``.  It does NOT
    read or write any file — the caller controls I/O.  This makes it safe
    for per-worker config generation (no shared global state).

    Patches applied:
      1. Proxy baseUrl: redirect PROXY_PROVIDERS traffic through the local proxy.
      2. Experiment model merge: register models from EXPERIMENT_MODEL_REGISTRY
         that are missing from the config.
      3. contextWindow override: set contextWindow on all models so OpenClaw fires
         compaction at the benchmark target threshold.
      4. compaction.model override: route compaction summarization through a non-Azure
         model to avoid Azure content filter 400 errors.
    """
    from constants import EXPERIMENT_MODEL_REGISTRY

    providers = cfg.setdefault("models", {}).setdefault("providers", {})

    # 1. Experiment model merge — register missing providers AND models
    #    (must run BEFORE proxy rewrite so newly-created providers get proxied)
    for pid, reg in EXPERIMENT_MODEL_REGISTRY.items():
        spec = providers.get(pid)
        if not spec:
            # Provider missing from config — create it from the registry.
            # Resolve an API key from the env vars listed in apiKeyEnvs.
            api_key_env = None
            for env_name in reg.get("apiKeyEnvs", []):
                if os.environ.get(env_name):
                    api_key_env = env_name
                    break
            if not api_key_env:
                logger.warning(
                    "Config patch: skipping provider %s — no API key found in env (%s)",
                    pid, reg.get("apiKeyEnvs"),
                )
                continue
            spec = {
                "baseUrl": reg["baseUrl"],
                "apiKey": api_key_env,
                "api": reg.get("api", "openai-completions"),
                "models": [],
            }
            providers[pid] = spec
            logger.info("Config patch: created provider %s (baseUrl=%s, apiKey=$%s)", pid, reg["baseUrl"], api_key_env)

        existing_ids = {m["id"] for m in spec.get("models", [])}
        for reg_model in reg.get("models", []):
            if reg_model["id"] not in existing_ids:
                entry = {
                    "id": reg_model["id"],
                    "name": reg_model.get("name", reg_model["id"]),
                    "reasoning": False,
                    "input": reg_model.get("input", ["text"]),
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "maxTokens": reg_model.get("maxTokens", 8192),
                }
                spec["models"].append(entry)
                logger.info("Config patch: added model %s/%s", pid, reg_model["id"])

    # 2. Proxy baseUrl — redirect needsUsageProxy providers through the local proxy.
    #    Runs after model merge so newly-created providers are already in `providers`.
    if proxy_port is not None:
        for pid in PROXY_PROVIDERS:
            spec = providers.get(pid)
            if spec and "baseUrl" in spec:
                spec["baseUrl"] = f"http://127.0.0.1:{proxy_port}/v1"
                logger.info("Config patch: %s baseUrl → proxy port %d", pid, proxy_port)

    # 3. contextWindow override
    if context_window_override:
        for pid, spec in providers.items():
            for model in spec.get("models", []):
                model["contextWindow"] = context_window_override
        logger.info("Config patch: contextWindow → %d on all models", context_window_override)

    # 4. compaction.model override
    if compaction_model:
        compaction_cfg = (
            cfg.setdefault("agents", {})
            .setdefault("defaults", {})
            .setdefault("compaction", {})
        )
        compaction_cfg["model"] = compaction_model
        logger.info("Config patch: compaction.model → %s", compaction_model)

    return cfg


class UsageProxy:
    """Context manager: HTTP proxy + openclaw.json config patcher.

    Two usage modes:

    **Smoke / single-gateway mode** (patches global config):
        The smoke harness runs against the user's default gateway at
        ``~/.openclaw``.  In this mode UsageProxy patches the global
        ``openclaw.json`` on enter and restores it on exit.  This is NOT
        safe for parallel top-level processes — use worker-local mode for
        that.

    **Worker-local mode** (experiment_runner):
        ``experiment_runner`` creates per-worker state dirs with their own
        ``openclaw.json``.  It calls ``patch_openclaw_config()`` directly
        on each worker's config dict — no global mutation.  UsageProxy is
        started with ``patch_global=False`` so it only runs the HTTP proxy.

    The HTTP proxy itself is stateless and safe for concurrent use by
    multiple worker gateways.
    """

    def __init__(
        self,
        openclaw_state_dir: str | None = None,
        context_window_override: int | None = None,
        compaction_model: str | None = DEFAULT_COMPACTION_MODEL,
        patch_global: bool = True,
    ) -> None:
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0
        self._context_window_override = context_window_override
        self._compaction_model = compaction_model
        self._patch_global = patch_global
        self._openclaw_json: Path | None = None
        self._original_cfg: dict | None = None  # snapshot for restore

        if patch_global:
            state_dir = Path(openclaw_state_dir) if openclaw_state_dir else Path.home() / ".openclaw"
            candidate = state_dir / "openclaw.json"
            if candidate.exists():
                self._openclaw_json = candidate

    def __enter__(self) -> UsageProxy:
        import atexit, signal

        # Start HTTP proxy on an OS-assigned port
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Usage-fix proxy started on 127.0.0.1:%d → %s", self.port, UPSTREAM_BASE)

        # Patch global openclaw.json (smoke / single-gateway mode only)
        if self._patch_global and self._openclaw_json is not None:
            self._patch_global_config()

            # Safety net: restore on unclean exit
            def _emergency_restore(*_args):
                if self._original_cfg is not None:
                    self._restore_global_config()

            atexit.register(_emergency_restore)
            signal.signal(signal.SIGTERM, lambda *a: (_emergency_restore(), exit(0)))

        return self

    def __exit__(self, *exc) -> bool:
        if self._patch_global and self._original_cfg is not None:
            self._restore_global_config()
        if self._server:
            self._server.shutdown()
            logger.info("Usage-fix proxy stopped (port %d)", self.port)
        return False

    # -- global config patch/restore (smoke mode only) -----------------------

    def _patch_global_config(self) -> None:
        """Snapshot global config, then apply experiment patches in-place."""
        with self._openclaw_json.open() as f:
            cfg = json.load(f)

        # Save a full snapshot so restore is a simple overwrite — no field-by-field undo
        self._original_cfg = json.loads(json.dumps(cfg))  # deep copy

        patch_openclaw_config(
            cfg,
            proxy_port=self.port,
            context_window_override=self._context_window_override,
            compaction_model=self._compaction_model,
        )

        with self._openclaw_json.open("w") as f:
            json.dump(cfg, f, indent=4)
        logger.info("Patched global openclaw.json (will restore on exit)")

    def _restore_global_config(self) -> None:
        """Restore global config from the snapshot taken before patching."""
        try:
            with self._openclaw_json.open("w") as f:
                json.dump(self._original_cfg, f, indent=4)
            self._original_cfg = None
            logger.info("Restored global openclaw.json from snapshot")
        except Exception as e:
            logger.warning("Failed to restore global openclaw.json: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Shubiaobiao usage-fix reverse proxy")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: 0 = OS-assigned)")
    args = parser.parse_args()
    with UsageProxy() as proxy:
        if args.port:
            # User requested a specific port — rebind
            proxy._server.shutdown()
            proxy._server = ThreadingHTTPServer(("127.0.0.1", args.port), ProxyHandler)
            proxy.port = args.port
            proxy._thread = threading.Thread(target=proxy._server.serve_forever, daemon=False)
            proxy._thread.start()
        logger.info("Proxy running on 127.0.0.1:%d (Ctrl+C to stop)", proxy.port)
        try:
            proxy._thread.join()
        except KeyboardInterrupt:
            pass
