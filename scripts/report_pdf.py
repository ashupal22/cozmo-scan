"""Render docs/technical_report.md to docs/technical_report.pdf (A4) with headless Chrome, and print the page count.

    python scripts/report_pdf.py
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


def main():
    src = ROOT / "docs" / "technical_report.md"
    html = markdown.markdown(src.read_text(), extensions=["tables", "fenced_code"])
    page = ROOT / "docs" / "technical_report.html"
    page.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>")
    out = ROOT / "docs" / "technical_report.pdf"
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}", page.as_uri()],
                   check=True, capture_output=True)
    page.unlink()
    from pypdf import PdfReader
    pages = len(PdfReader(str(out)).pages)
    print(f"wrote {out}: {pages} pages")
    return 0 if pages <= 6 else 1


if __name__ == "__main__":
    sys.exit(main())
