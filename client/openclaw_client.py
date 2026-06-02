"""
OpenClaw Client for GroundingBench.

Uses the `openclaw` CLI for session management and message sending,
which has full operator.admin scope via device pairing.

Supports per-trial workspace isolation via OpenClaw's multi-agent system:
each trial can use a dedicated agent with its own workspace directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPENCLAW_STATE_DIR = Path.home() / ".openclaw"
DEFAULT_AGENT_ID = "main"


def _load_token(state_dir: Path | None = None) -> str:
    """Load auth token from env or openclaw.json."""
    token = os.environ.get("OPENCLAW_TOKEN")
    if token:
        return token
    cfg_path = (state_dir or OPENCLAW_STATE_DIR) / "openclaw.json"
    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg["gateway"]["auth"]["token"]


class GatewayError(Exception):
    def __init__(self, code: str | None, message: str, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


class OpenClawClient:
    """
    Client for the OpenClaw gateway using CLI commands.

    Supports per-trial isolation via agent_id: each agent has its own
    workspace, session store, and memory index.

    Usage:
        async with OpenClawClient(agent_id="trial_001") as client:
            session_id = await client.start_session()
            response = await client.send_message("Hello")
    """

    def __init__(
        self,
        agent_id: str = DEFAULT_AGENT_ID,
        ws_url: str | None = None,
        token: str | None = None,
        state_dir: Path | None = None,
    ):
        self.agent_id = agent_id
        self.state_dir = state_dir or OPENCLAW_STATE_DIR
        self.ws_url = ws_url or os.environ.get("OPENCLAW_WS_URL", "ws://127.0.0.1:18789")
        self.token = token or _load_token(self.state_dir)

    @property
    def sessions_dir(self) -> Path:
        return self.state_dir / "agents" / self.agent_id / "sessions"

    @property
    def workspace_dir(self) -> Path:
        if self.agent_id == DEFAULT_AGENT_ID:
            return self.state_dir / "workspace"
        cfg_path = self.state_dir / "openclaw.json"
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = json.load(f)
            for a in cfg.get("agents", {}).get("list", []):
                if a.get("id") == self.agent_id:
                    ws = a.get("workspace")
                    if ws:
                        return Path(ws)
        return self.state_dir / f"workspace-{self.agent_id}"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Verify gateway is reachable.

        Primary check: HTTP probe of the gateway dashboard (mirrors the
        readiness poll in harness/checkpoint.py). On OpenClaw 2026.3.x the
        `gateway status` text emits the "Listening" line only conditionally
        (and is truncated under the short CLI timeout when run as a managed
        service is "stopped"), so the CLI-text heuristic is unreliable. The
        HTTP probe is authoritative: if the WS gateway is serving, the
        loopback HTTP endpoint returns a response.
        """
        # ws://host:port -> http://host:port
        http_url = self.ws_url.replace("ws://", "http://").replace("wss://", "https://")
        from urllib.request import urlopen
        from urllib.error import HTTPError, URLError
        try:
            await asyncio.to_thread(lambda: urlopen(http_url, timeout=5).read(0))
            logger.debug("Gateway connection verified via HTTP (%s)", http_url)
            return
        except HTTPError:
            # The server RESPONDED (e.g. 404 because the web dashboard is disabled
            # on episode/branch gateways via cfg.pop("web")). An HTTP status of any
            # kind proves the gateway is listening → reachable.
            logger.debug("Gateway reachable via HTTP (got HTTP error status) (%s)", http_url)
            return
        except (URLError, OSError, ConnectionError) as http_err:
            logger.debug("HTTP probe failed (%s); falling back to CLI status", http_err)

        result = await self._run_cli(["openclaw", "gateway", "status"], timeout=10)
        if "RPC probe: ok" not in result and "Listening" not in result:
            raise ConnectionError(f"Gateway not reachable: {result}")
        logger.debug("Gateway connection verified via CLI")

    async def disconnect(self) -> None:
        pass

    async def __aenter__(self) -> "OpenClawClient":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # CLI execution helper
    # ------------------------------------------------------------------

    async def _run_cli(
        self, cmd: list[str], timeout: float | None = None, input_text: str | None = None
    ) -> str:
        """Run an openclaw CLI command asynchronously."""
        t = timeout  # None = no timeout
        cli_env = {**os.environ, "OPENCLAW_STATE_DIR": str(self.state_dir)}
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                env=cli_env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=input_text.encode() if input_text else None),
                timeout=t,
            )
            output = stdout.decode().strip()
            if proc.returncode != 0:
                err_msg = stderr.decode().strip() or output
                raise GatewayError(
                    code=str(proc.returncode),
                    message=f"CLI command failed: {err_msg}",
                )
            return output
        except asyncio.TimeoutError:
            raise TimeoutError(f"CLI command timed out after {t}s: {' '.join(cmd)}")

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def start_session(self, reason: str = "new") -> str:
        """Start a fresh session. Returns the real session_id from disk.

        Retries up to 3 times if OpenClaw returns a startup-only response
        (e.g. "inherited auth-profiles from main agent") without creating a JSONL.
        """
        import uuid as _uuid
        import asyncio as _asyncio

        for attempt in range(3):
            forced_id = str(_uuid.uuid4())
            cmd = [
                "openclaw", "agent",
                "--agent", self.agent_id,
                "--session-id", forced_id,
                "--message", "Hello.",
                "--json",
            ]
            cli_output = await self._run_cli(cmd)

            # Read the session id OpenClaw wrote to disk
            real_session_id = self._read_latest_session_id()
            jsonl_path = self.sessions_dir / f"{real_session_id}.jsonl"

            # Wait briefly for JSONL to flush to disk
            if not jsonl_path.exists():
                for _ in range(20):
                    await _asyncio.sleep(0.5)
                    if jsonl_path.exists():
                        break

            if jsonl_path.exists():
                logger.info("Started new session: %s (agent=%s)", real_session_id, self.agent_id)
                return real_session_id

            # JSONL still missing — OpenClaw returned a startup-only response
            logger.warning(
                "start_session attempt %d: JSONL missing after 10s; CLI output=%s — retrying",
                attempt + 1, cli_output[:300] if cli_output else "<empty>",
            )
            await _asyncio.sleep(2.0)

        raise RuntimeError(
            f"Session JSONL not found after 3 start_session attempts. "
            f"Agent={self.agent_id}, sessions_dir={self.sessions_dir}"
        )

    def _read_latest_session_id(self) -> str:
        """Read the most recent session_id from this agent's sessions.json."""
        sessions_json = self.sessions_dir / "sessions.json"
        if not sessions_json.exists():
            raise RuntimeError(f"sessions.json not found: {sessions_json}")

        import json as _json
        with sessions_json.open() as f:
            data = _json.load(f)

        # Find the most recent session by timestamp or just pick the one entry
        best_id = None
        best_ts = 0
        for key, entry in data.items():
            sid = entry.get("sessionId")
            ts = entry.get("updatedAt", 0)
            if sid and ts >= best_ts:
                best_ts = ts
                best_id = sid

        if not best_id:
            raise RuntimeError(f"No sessionId found in {sessions_json}")

        return best_id

    async def send_message(
        self,
        text: str,
        session_id: str | None = None,
        image_path: str | None = None,
    ) -> dict:
        """Send a user turn and wait for the final assistant response.

        If image_path is provided, the image is attached using @file syntax.
        """
        # Prepend @filepath to message to attach image
        message = f"@{image_path} {text}" if image_path else text
        cmd = [
            "openclaw", "agent",
            "--agent", self.agent_id,
            "--message", message,
            "--json",
            "--timeout", "300",  # 5 min per LLM turn (GPT-5 with large context)
        ]
        if session_id:
            cmd.extend(["--session-id", session_id])

        output = await self._run_cli(cmd)

        try:
            result = json.loads(output)
        except json.JSONDecodeError:
            result = {"response_text": output}

        return result

    # ------------------------------------------------------------------
    # JSONL / workspace helpers
    # ------------------------------------------------------------------

    def get_session_jsonl(self, session_id: str) -> list[dict]:
        """Read and parse a session's JSONL transcript."""
        path = self.sessions_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        entries = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def get_session_compaction_count(self, session_id: str) -> int:
        """Read compactionCount from sessions.json (0 if absent)."""
        sessions_json = self.sessions_dir / "sessions.json"
        if not sessions_json.exists():
            return 0
        with sessions_json.open() as f:
            data = json.load(f)
        for key, entry in data.items():
            if entry.get("sessionId") == session_id:
                return entry.get("compactionCount", 0)
        return 0

    def get_effective_input_tokens(self, session_id: str) -> int | None:
        """Unified authoritative context-size signal: input + cacheRead from the latest turn.

        Returns effective_input_tokens = input + cacheRead for the most recent
        message entry in the session JSONL. This accounts for prompt caching
        where 'input' alone can be tiny (e.g. 47) while cacheRead carries the
        bulk of context (e.g. 18240).

        Robustly checks field name variants:
          input component: inputTokens, input_tokens, input
          cache component: cacheRead, cache_read, cachedInput

        Returns None if no provider-reported input field is found.
        """
        entries = self.get_session_jsonl(session_id)
        for entry in reversed(entries):
            if entry.get("type") != "message":
                continue
            usage = entry.get("message", {}).get("usage", {})
            if not isinstance(usage, dict):
                continue
            # Resolve input component
            raw_input = None
            for key in ("inputTokens", "input_tokens", "input"):
                val = usage.get(key)
                if isinstance(val, (int, float)) and val >= 0:
                    raw_input = int(val)
                    break
            if raw_input is None:
                continue  # skip entries without input reporting
            # Resolve cache component (0 if absent — not all providers cache)
            cache = 0
            for key in ("cacheRead", "cache_read", "cachedInput"):
                val = usage.get(key)
                if isinstance(val, (int, float)) and val > 0:
                    cache = int(val)
                    break
            return raw_input + cache
        return None

    def get_session_metadata(self, session_id: str) -> dict:
        """Read full session entry from sessions.json for recording in trial logs."""
        sessions_json = self.sessions_dir / "sessions.json"
        if not sessions_json.exists():
            return {}
        with sessions_json.open() as f:
            data = json.load(f)
        for key, entry in data.items():
            if entry.get("sessionId") == session_id:
                return entry
        return {}

    def get_system_prompt(self, session_id: str) -> str:
        """Extract the system prompt sent to the model in this session."""
        sessions_json = self.sessions_dir / "sessions.json"
        if not sessions_json.exists():
            return ""
        with sessions_json.open() as f:
            data = json.load(f)

        for key, entry in data.items():
            if entry.get("sessionId") == session_id:
                if entry.get("systemSent"):
                    bootstrap = self.workspace_dir / "BOOTSTRAP.md"
                    if bootstrap.exists():
                        return bootstrap.read_text()
                break

        entries = self.get_session_jsonl(session_id)
        for entry in entries:
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                if msg.get("role") == "user":
                    for content in msg.get("content", []):
                        if content.get("type") == "text":
                            text = content.get("text", "")
                            if "SOUL.md" in text or "AGENTS.md" in text or "system" in text.lower():
                                return text

        # Fallback: read BOOTSTRAP.md directly even when systemSent is missing.
        # OpenClaw may not set systemSent in all session modes, but BOOTSTRAP.md
        # content is still injected into the model prompt.
        bootstrap = self.workspace_dir / "BOOTSTRAP.md"
        if bootstrap.exists():
            return bootstrap.read_text()

        return ""

    def get_workspace_path(self) -> Path:
        return self.workspace_dir

    def get_memory_md_path(self) -> Path:
        return self.workspace_dir / "MEMORY.md"

    def get_daily_memory_path(self, date_str: str) -> Path:
        return self.workspace_dir / "memory" / f"{date_str}.md"
