"""
Shared image generation utilities for artifact planters.

All images are written to data/generated/ (created on demand).
"""

from __future__ import annotations

from pathlib import Path

GENERATED_DIR = Path(__file__).parent.parent / "generated"


def get_output_path(filename: str) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    return GENERATED_DIR / filename


# ---------------------------------------------------------------------------
# Fonts — use PIL default (no external font required)
# ---------------------------------------------------------------------------

def _font(size: int = 14):
    """Return a PIL font. Falls back to default if truetype not available."""
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _font_bold(size: int = 14):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

BG = (255, 255, 255)
TEXT_DARK = (30, 30, 30)
TEXT_GRAY = (100, 100, 100)
TEXT_LIGHT = (160, 160, 160)
ACCENT_BLUE = (26, 115, 232)
ACCENT_GREEN = (52, 168, 83)
ACCENT_RED = (234, 67, 53)
ACCENT_ORANGE = (251, 188, 4)
BORDER = (218, 220, 224)
CARD_BG = (248, 249, 250)
HEADER_BG = (32, 33, 36)
HEADER_TEXT = (255, 255, 255)


def draw_rounded_rect(draw, xy, radius: int, fill, outline=None):
    """Draw a rounded rectangle."""
    from PIL import ImageDraw
    x0, y0, x1, y1 = xy
    draw.rectangle([x0 + radius, y0, x1 - radius, y1], fill=fill)
    draw.rectangle([x0, y0 + radius, x1, y1 - radius], fill=fill)
    draw.ellipse([x0, y0, x0 + 2 * radius, y0 + 2 * radius], fill=fill)
    draw.ellipse([x1 - 2 * radius, y0, x1, y0 + 2 * radius], fill=fill)
    draw.ellipse([x0, y1 - 2 * radius, x0 + 2 * radius, y1], fill=fill)
    draw.ellipse([x1 - 2 * radius, y1 - 2 * radius, x1, y1], fill=fill)
    if outline:
        draw.rectangle([x0 + radius, y0, x1 - radius, y1], outline=outline)
        draw.rectangle([x0, y0 + radius, x1, y1 - radius], outline=outline)


def text_width(draw, text: str, font) -> int:
    """Get text width in pixels."""
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]
