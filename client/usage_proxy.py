"""
Local reverse proxy + openclaw.json config patcher for MIRAGE experiments.

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

# Fields that the shubiaobiao proxy populates incorrectly (always 0)
# while the standard prompt_tokens/completion_tokens carry the real values.
MISLEADING_USAGE_FIELDS = {"input_tokens", "output_tokens", "input_tokens_details"}


def _strip_misleading_usage(body: dict) -> dict:
    """Fix misleading zero-valued usage fields from shubiaobiao API response.

    Strategy: overwrite misleading zero-valued fields with the correct values
    from prompt_tokens/completion_tokens. This ensures OpenClaw's ?? cascade
    reads the correct value regardless of which field it hits first.

    - input_tokens: 0 → overwrite with prompt_tokens value
    - output_tokens: 0 → overwrite with completion_tokens value
    - input_tokens_details: 0/null → remove
    """
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        # Overwrite misleading zeros with correct values
        if usage.get("input_tokens") is not None and usage["input_tokens"] == 0 and prompt:
            usage["input_tokens"] = prompt
        if usage.get("output_tokens") is not None and usage["output_tokens"] == 0 and completion:
            usage["output_tokens"] = completion

        # Remove details fields that are always null/zero
        val = usage.get("input_tokens_details")
        if val is None or val == 0:
            usage.pop("input_tokens_details", None)
    return body


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
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
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

        # Tool calls
        for tc in msg.get("tool_calls", []):
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


# JSON Schema keywords that Gemini's function-calling API rejects.
# Gemini uses a strict subset of JSON Schema and does not recognise these.
_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset({
    "patternProperties",
    "unevaluatedProperties",
    "$schema",
    "$id",
    "$comment",
    "if",
    "then",
    "else",
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "contentMediaType",
    "contentEncoding",
})

_GEMINI_MODEL_SUBSTRINGS = ("gemini",)


def _is_gemini_model(model: str) -> bool:
    if not model:
        return False
    model_lower = model.lower()
    return any(s in model_lower for s in _GEMINI_MODEL_SUBSTRINGS)


def _strip_gemini_schema_keys(obj: object) -> object:
    """Recursively remove JSON Schema keywords unsupported by Gemini from *obj*."""
    if isinstance(obj, dict):
        return {
            k: _strip_gemini_schema_keys(v)
            for k, v in obj.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(obj, list):
        return [_strip_gemini_schema_keys(item) for item in obj]
    return obj


def _sanitize_tools_for_gemini(req_json: dict) -> dict:
    """Strip unsupported schema keywords from tool function declarations for Gemini.

    Gemini rejects requests whose tool schemas contain keywords like
    ``patternProperties`` that are not part of Gemini's supported subset
    of JSON Schema.  This function strips those keys recursively from the
    ``tools`` array before the request is forwarded upstream.

    Only applied when the model name contains ``gemini``.
    """
    model = req_json.get("model", "")
    if not _is_gemini_model(model):
        return req_json
    tools = req_json.get("tools")
    if not tools:
        return req_json
    cleaned = _strip_gemini_schema_keys(tools)
    if cleaned != tools:
        logger.info("Gemini tool-schema sanitize: stripped unsupported keys from tools for model=%s", model)
        req_json = {**req_json, "tools": cleaned}
    return req_json


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

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(content_length) if content_length > 0 else b""

        upstream_url = f"{UPSTREAM_BASE}{self.path}"

        # Forward all headers except Host
        headers = {}
        for key in self.headers:
            if key.lower() != "host":
                headers[key] = self.headers[key]

        # Destream: convert streaming requests to non-streaming so we get usage
        # in the JSON response.  OpenClaw's pi-ai sets supportsUsageInStreaming=false
        # for non-standard endpoints, so the SDK omits stream_options.  Shubiaobiao
        # also doesn't support stream_options.  By destreaming, we get a JSON
        # response with prompt_tokens/completion_tokens, then re-emit it as SSE
        # so pi-ai's streaming parser can read the usage chunk.
        was_streaming = False
        try:
            req_json = json.loads(request_body)
            if req_json.get("stream"):
                was_streaming = True
                req_json["stream"] = False
                req_json.pop("stream_options", None)
                logger.info(
                    "Proxy destream: model=%s (converting stream→non-stream for usage)",
                    req_json.get("model"),
                )
            # Strip Gemini-incompatible JSON Schema keywords from tool declarations.
            req_json = _sanitize_tools_for_gemini(req_json)
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
                    "Proxy upstream %d: %s (first 300 chars: %s)",
                    status, self.path, resp_body[:300],
                )

        # Decompress gzip if needed (OpenAI SDK sends Accept-Encoding: gzip)
        if resp_headers.get("Content-Encoding", "").lower() == "gzip":
            resp_body = gzip.decompress(resp_body)
            del resp_headers["Content-Encoding"]

        content_type = resp_headers.get("Content-Type", "")
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
        # Forward GET requests (e.g. /v1/models) transparently
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
PROXY_PROVIDERS = ("shubiaobiao",)

# Default compaction summarization model — text-only, no Azure content filter.
DEFAULT_COMPACTION_MODEL = "deepseek/deepseek-chat"


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

    providers = cfg.get("models", {}).get("providers", {})

    # 1. Proxy baseUrl
    if proxy_port is not None:
        for pid in PROXY_PROVIDERS:
            spec = providers.get(pid)
            if spec and "baseUrl" in spec:
                spec["baseUrl"] = f"http://127.0.0.1:{proxy_port}/v1"
                logger.info("Config patch: %s baseUrl → proxy port %d", pid, proxy_port)

    # 2. Experiment model merge — add entire provider if missing, merge models if present
    for pid, reg in EXPERIMENT_MODEL_REGISTRY.items():
        spec = providers.get(pid)
        if not spec:
            # Provider absent from global config — add it from the registry.
            # Resolve API key from env.
            api_key = None
            for env_name in reg.get("apiKeyEnvs", []):
                api_key = os.environ.get(env_name)
                if api_key:
                    break
            if not api_key:
                logger.warning(
                    "Config patch: skipping provider %s — no API key found in env vars %s",
                    pid, reg.get("apiKeyEnvs", []),
                )
                continue
            base_url = reg["baseUrl"]
            if proxy_port is not None and pid in PROXY_PROVIDERS:
                base_url = f"http://127.0.0.1:{proxy_port}/v1"
            spec = {
                "baseUrl": base_url,
                "apiKey": api_key,
                "api": reg.get("api", "openai-completions"),
                "models": [],
            }
            cfg.setdefault("models", {}).setdefault("providers", {})[pid] = spec
            providers[pid] = spec
            logger.info("Config patch: added provider %s (baseUrl=%s)", pid, base_url)
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
