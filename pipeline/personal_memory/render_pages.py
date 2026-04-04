"""Screenshot renderer for the T × M personal memory study.

Reads the case bank, renders each case's page_spec via HTML template +
Playwright, saves PNGs, and updates rendered_image_path in the case bank.

IMPORTANT: --output-dir must be under the repo root (GB_ROOT) because
case bank stores repo-relative paths consumed by the runner.

Stale-path safety: all rendered_image_path values are cleared at the
start of each run. Only successfully rendered cases get a path written
back. This holds even if the renderer crashes mid-run.

Usage:
    cd MIRAGE
    python -m pipeline.personal_memory.render_pages
    python -m pipeline.personal_memory.render_pages --output-dir data/generated_personal_ui

Requirements:
    pip install playwright
    playwright install chromium
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

GB_ROOT = Path(__file__).parent.parent.parent
DEFAULT_BANK = GB_ROOT / "configs" / "study" / "personal_memory_tm_cases_v1.json"
DEFAULT_OUTPUT = GB_ROOT / "data" / "generated_personal_ui"
TEMPLATES_DIR = Path(__file__).parent / "templates"

VIEWPORT = {"width": 420, "height": 720}


def render_html_with_spec(template_path: Path, page_spec: dict) -> str:
    """Inject page_spec into HTML template as a JS global."""
    html = template_path.read_text(encoding="utf-8")
    spec_json = json.dumps(page_spec, ensure_ascii=False)
    inject = f"<script>const PAGE_SPEC = {spec_json};</script>"
    html = html.replace("</head>", f"{inject}\n</head>")
    return html


def render_all(bank_path: Path, output_dir: Path, *, update_bank: bool = True) -> list[dict]:
    """Render all cases and optionally update the case bank.

    Stale-path safety:
      1. All rendered_image_path values are cleared to None upfront.
      2. Only successfully rendered cases get a new path.
      3. Bank is written back in a finally block, so even renderer-level
         crashes leave the bank in a clean (cleared) state rather than
         with stale success paths.
    """
    # Validate output_dir is under GB_ROOT
    try:
        output_dir.resolve().relative_to(GB_ROOT.resolve())
    except ValueError:
        print(f"ERROR: --output-dir must be under repo root ({GB_ROOT})")
        print(f"  Got: {output_dir.resolve()}")
        print(f"  Case bank stores repo-relative paths; external dirs are not supported.")
        sys.exit(1)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: playwright not installed. Run:")
        print("  pip install playwright && playwright install chromium")
        sys.exit(1)

    bank = json.load(bank_path.open())
    cases = bank["cases"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clear ALL rendered_image_path upfront (stale-path safety)
    for case in cases:
        case["rendered_image_path"] = None

    rendered = []
    skipped = []

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport=VIEWPORT)

            for case in cases:
                case_id = case["case_id"]
                template_id = case.get("page_template_id", "")
                page_spec = case.get("page_spec", {})

                template_path = TEMPLATES_DIR / f"{template_id}.html"
                if not template_path.exists():
                    print(f"  SKIP {case_id}: template {template_id}.html not found")
                    skipped.append(case_id)
                    continue

                tmp_path = None
                try:
                    html_content = render_html_with_spec(template_path, page_spec)

                    with tempfile.NamedTemporaryFile(suffix=".html", mode="w",
                                                     encoding="utf-8", delete=False) as tmp:
                        tmp.write(html_content)
                        tmp_path = tmp.name

                    page.goto(f"file://{tmp_path}")
                    page.wait_for_load_state("networkidle")

                    png_name = f"{case_id}.png"
                    png_path = output_dir / png_name
                    page.screenshot(path=str(png_path), full_page=True)

                    rel_path = str(png_path.relative_to(GB_ROOT))
                    case["rendered_image_path"] = rel_path
                    rendered.append({"case_id": case_id, "path": rel_path})
                    print(f"  OK   {case_id} → {rel_path}")

                except Exception as exc:
                    print(f"  FAIL {case_id}: {exc}")
                    skipped.append(case_id)
                finally:
                    if tmp_path:
                        Path(tmp_path).unlink(missing_ok=True)

            browser.close()

    finally:
        # Always write bank back — even on renderer-level crash, the cleared
        # paths ensure no stale success state persists.
        if update_bank:
            with bank_path.open("w") as f:
                json.dump(bank, f, indent=2, ensure_ascii=False)

    # Summary (always print both lines)
    print(f"\n--- Render Summary ---")
    print(f"  Rendered:       {len(rendered)}/{len(cases)}")
    print(f"  Skipped/Failed: {len(skipped)}/{len(cases)}"
          + (f" — {skipped}" if skipped else ""))
    if update_bank:
        print(f"  Case bank updated: {bank_path}")

    return rendered


def main():
    p = argparse.ArgumentParser(description="Render T×M case screenshots")
    p.add_argument("--case-bank", type=Path, default=DEFAULT_BANK,
                   help="Path to case bank JSON")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Output dir for PNGs (must be under repo root)")
    p.add_argument("--no-update-bank", action="store_true",
                   help="Don't write rendered_image_path back to case bank")
    args = p.parse_args()

    render_all(args.case_bank, args.output_dir,
               update_bank=not args.no_update_bank)


if __name__ == "__main__":
    main()
