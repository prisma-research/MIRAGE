"""
Shared HuggingFace dataset download, sampling, and image-save utilities.

Used by all real-dataset planters.
"""

from __future__ import annotations

import logging
import os
import re
import string
from pathlib import Path
from typing import Callable

from PIL import Image

logger = logging.getLogger(__name__)

# Local HF cache directory (gitignored, lives in data/)
CACHE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "datasets_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── stop words ────────────────────────────────────────────────────────────────
_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "on",
    "at", "by", "for", "with", "from", "that", "this", "it", "its",
    "i", "you", "he", "she", "we", "they", "and", "or", "but", "not",
    "what", "which", "who", "how", "when", "where", "s", "t",
}


# ── HF loading ─────────────────────────────────────────────────────────────────

def load_hf_dataset(primary_id: str, split: str, aliases: tuple[str, ...] = (), streaming: bool = True):
    """
    Try primary_id, then each alias in turn.  Sets HF_DATASETS_CACHE env var.
    Returns a HuggingFace dataset object (possibly IterableDataset if streaming).
    Raises RuntimeError if all IDs fail.
    """
    try:
        from datasets import load_dataset as _load  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required.  Install with: pip install datasets>=2.19.0"
        ) from exc

    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR))

    for hf_id in (primary_id,) + tuple(aliases):
        try:
            logger.info("Loading HF dataset %s (split=%s, streaming=%s)", hf_id, split, streaming)
            ds = _load(hf_id, split=split, streaming=streaming,
                       cache_dir=str(CACHE_DIR))
            logger.info("Loaded %s", hf_id)
            return ds
        except Exception as exc:
            logger.warning("Failed to load %s: %s", hf_id, exc)

    raise RuntimeError(
        f"Could not load dataset from any of: {(primary_id,) + tuple(aliases)}"
    )


def get_samples(
    dataset_id: str,
    split: str,
    n: int,
    seed: int = 42,
    aliases: tuple[str, ...] = (),
    filter_fn: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """
    Return up to *n* distinct samples from the dataset, deterministically.

    Strategy:
      - Load with streaming=True (only fetches what's needed).
      - Shuffle with buffer_size = max(n * 2, 2000) and seed.
      - Iterate until n samples collected (applying optional filter_fn).
    If the dataset is exhausted before n samples are found, returns all available
    and logs a warning.
    """
    ds = load_hf_dataset(dataset_id, split, aliases=aliases, streaming=True)

    buffer_size = max(n * 2, 2000)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    collected: list[dict] = []
    for sample in ds:
        if filter_fn is None or filter_fn(sample):
            collected.append(sample)
        if len(collected) >= n:
            break

    if len(collected) < n:
        logger.warning(
            "Dataset %s (split=%s) only provided %d/%d samples after filtering.",
            dataset_id, split, len(collected), n,
        )

    return collected


# ── Image saving ───────────────────────────────────────────────────────────────

def save_image_from_sample(image_or_bytes, output_path: Path) -> None:
    """
    Normalise a PIL.Image, raw bytes, or {"bytes": ...} dict, then save as PNG (RGB).
    Skips write if output_path already exists (idempotent / cache guard).
    """
    if output_path.exists():
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    img: Image.Image | None = None

    if isinstance(image_or_bytes, Image.Image):
        img = image_or_bytes
    elif isinstance(image_or_bytes, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(image_or_bytes))
    elif isinstance(image_or_bytes, dict):
        raw = image_or_bytes.get("bytes") or image_or_bytes.get("path")
        if isinstance(raw, bytes):
            import io
            img = Image.open(io.BytesIO(raw))
        elif isinstance(raw, str):
            img = Image.open(raw)
        else:
            raise ValueError(f"Unsupported image dict format: {list(image_or_bytes.keys())}")
    else:
        raise TypeError(f"Unsupported image type: {type(image_or_bytes)}")

    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    img.save(str(output_path), format="PNG")


# ── Keyword extraction ─────────────────────────────────────────────────────────

def extract_keywords(*texts: str, max_n: int = 5) -> list[str]:
    """
    Tokenise on whitespace + punctuation, filter stop words, dedupe, return top max_n.
    """
    tokens: list[str] = []
    punct_re = re.compile(r"[^\w\s]")
    for text in texts:
        if not text:
            continue
        cleaned = punct_re.sub(" ", text.lower())
        tokens.extend(t for t in cleaned.split() if len(t) > 1 and t not in _STOP_WORDS)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    return unique[:max_n]


# ── Rico zip fallback ──────────────────────────────────────────────────────────

RICO_ZIP_URL = "https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1_semantic_annotations.zip"
RICO_CACHE_DIR = CACHE_DIR / "rico_raw"


def _download_rico_zip() -> Path:
    """Download Rico zip to datasets_cache/rico_raw/ if not already present."""
    import urllib.request
    import zipfile

    zip_path = RICO_CACHE_DIR / "rico.zip"
    RICO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        logger.info("Downloading Rico zip from %s …", RICO_ZIP_URL)
        urllib.request.urlretrieve(RICO_ZIP_URL, zip_path)
        logger.info("Rico zip downloaded to %s", zip_path)

    extracted = RICO_CACHE_DIR / "extracted"
    if not extracted.exists():
        logger.info("Extracting Rico zip …")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extracted)
        logger.info("Rico extracted to %s", extracted)

    return extracted


def get_rico_samples_local(n: int = 500, seed: int = 42) -> list[dict]:
    """
    Fall back: load Rico samples directly from the downloaded zip.
    Returns list of dicts with keys: image_path (Path), package_name, ui_category.
    """
    import json
    import random

    extracted = _download_rico_zip()

    # Find semantic annotation JSON files
    json_files = list(extracted.rglob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(json_files)

    samples: list[dict] = []
    for jf in json_files:
        if len(samples) >= n:
            break
        try:
            with jf.open() as f:
                data = json.load(f)
            img_path = jf.with_suffix(".jpg")
            if not img_path.exists():
                img_path = jf.with_suffix(".png")
            if not img_path.exists():
                continue
            samples.append({
                "image_path": img_path,
                "package_name": data.get("activity", {}).get("root", {}).get("package", ""),
                "ui_category": data.get("activity", {}).get("root", {}).get("class", ""),
                "nodes": data.get("activity", {}).get("root", {}).get("children", []),
            })
        except Exception:
            continue

    if len(samples) < n:
        logger.warning("Rico local fallback: only %d/%d samples available.", len(samples), n)

    return samples
