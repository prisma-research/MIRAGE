"""
pdf2text — minimal PDF text-extraction tool used to ingest PDF evidence objects
into the (text-native) MIRAGE planting pipeline. PDFs are not a perceptual input
modality for the backbones (the chat APIs accept text + images, not PDF), so PDF
artifacts are preprocessed by this tool and the extracted text is what the agent
grounds on — exactly how a real personal agent would handle an attached PDF.

CLI:  python -m pipeline.pdf2text <file.pdf>
API:  from pipeline.pdf2text import pdf_to_text; pdf_to_text(path) -> str
"""
from __future__ import annotations
import sys
from pathlib import Path
from pdfminer.high_level import extract_text


def pdf_to_text(pdf_path: str | Path) -> str:
    """Extract the text layer of a PDF, normalised to single blank-line breaks."""
    raw = extract_text(str(pdf_path)) or ""
    # collapse form-feeds / excess blank lines
    lines = [ln.rstrip() for ln in raw.replace("\x0c", "\n").splitlines()]
    out, blank = [], 0
    for ln in lines:
        if ln.strip():
            out.append(ln); blank = 0
        else:
            blank += 1
            if blank <= 1:
                out.append("")
    return "\n".join(out).strip()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m pipeline.pdf2text <file.pdf>", file=sys.stderr); sys.exit(2)
    print(pdf_to_text(sys.argv[1]))
