"""
Render a synthetic code artifact (Python source) to a PNG via Pygments.
Full ground-truth control → reliable gold answers for the code modality object.

Usage: python -m pipeline.render_code_artifact
"""
from __future__ import annotations
from pathlib import Path
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import ImageFormatter

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "generated_images" / "code" / "code_01.png"

# Known facts embedded (used to author gold QA):
#   DEFAULT_TIMEOUT=30, MAX_RETRIES=5, API_BASE_URL=https://api.example.com/v2,
#   CACHE_TTL=3600, RETRY_BACKOFF=2.0, ENABLE_CACHE=True, LOG_LEVEL="INFO",
#   function fetch_user returns {} on failure, success check status_code==200,
#   default timeout param = DEFAULT_TIMEOUT.
SOURCE = '''# service_client.py — configuration and user fetch helper
import time

DEFAULT_TIMEOUT = 30          # seconds
MAX_RETRIES = 5
API_BASE_URL = "https://api.example.com/v2"
CACHE_TTL = 3600              # seconds
RETRY_BACKOFF = 2.0
ENABLE_CACHE = True
LOG_LEVEL = "INFO"


def fetch_user(user_id: int, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Fetch a user record by id. Returns {} on failure."""
    for attempt in range(MAX_RETRIES):
        resp = http_get(f"{API_BASE_URL}/users/{user_id}", timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(RETRY_BACKOFF ** attempt)
    return {}
'''

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fmt = ImageFormatter(font_size=28, line_numbers=True, style="default",
                         line_number_bg="#eeeeee", image_pad=20)
    png = highlight(SOURCE, PythonLexer(), fmt)
    OUT.write_bytes(png)
    print(f"Wrote {OUT} ({len(png)} bytes)")

if __name__ == "__main__":
    main()
