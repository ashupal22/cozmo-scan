"""Render docs/technical_report.md to docs/technical_report.pdf (A4) with headless Chrome, and print the page count.

    python scripts/report_pdf.py                                   # docs/technical_report.md, at most 6 pages
    python scripts/report_pdf.py docs/capture_protocol.md 1        # any page, with its page limit
"""
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 13mm 14mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 8.9pt; line-height: 1.33; color: #1c1c1c; }
h1 { font-size: 15pt; margin: 0 0 4pt; }
h2 { font-size: 11pt; margin: 9pt 0 3pt; border-bottom: 0.6pt solid #999; padding-bottom: 1pt; }
p, ul, ol { margin: 3pt 0; }
li { margin: 1pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; font-size: 8pt; page-break-inside: avoid; }
th, td { border: 0.5pt solid #bbb; padding: 2pt 4pt; vertical-align: top; text-align: left; }
th { background: #eef0f3; }
code { font-family: Menlo, monospace; font-size: 7.8pt; }
pre { font-size: 7.2pt; background: #f5f5f5; padding: 4pt; line-height: 1.2; page-break-inside: avoid; }
"""


def _list_breaks(text: str) -> str:
    """Python-Markdown needs a blank line before a list that follows a paragraph (GitHub does not)."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        is_item = line.lstrip().startswith("- ") or line[:3].rstrip(".").isdigit() and line[1:3] in (". ", ".")
        prev = out[-1] if out else ""
        prev_item = prev.lstrip().startswith("- ") or prev[:2].rstrip(".").isdigit()
        if not fenced and is_item and prev.strip() and not prev_item and not prev.startswith("|"):
            out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def main():
    src = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "docs" / "technical_report.md"
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    html = markdown.markdown(_list_breaks(src.read_text()), extensions=["tables", "fenced_code"])
    page = src.with_suffix(".html")
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
    out = src.with_suffix(".pdf")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()],
                   check=True, capture_output=True)
    page.unlink()
    from pypdf import PdfReader
    pages = len(PdfReader(str(out)).pages)
    print(f"wrote {out}: {pages} pages")
    return 0 if pages <= max_pages else 1


if __name__ == "__main__":
    sys.exit(main())
