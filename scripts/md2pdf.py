"""Team documents: markdown -> A4 PDF (pandoc + Chromium). Not part of the study.

Run: python scripts/md2pdf.py docs/PLAN.md PLAN.pdf
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

CSS = """
body{max-width:none!important;padding:0!important;font-family:'DejaVu Sans',Arial,sans-serif;font-size:10.5pt;line-height:1.45;color:#1a1a1a;max-width:none;margin:0}
h1{font-size:14pt;color:#1d3b6e;border-bottom:1.5px solid #1d3b6e;padding-bottom:2px;margin:16px 0 6px}
h2{font-size:12pt;color:#1d3b6e;margin:12px 0 4px}
header h1.title{font-size:18pt;border:none;margin-bottom:2px}
p.subtitle{color:#555;margin-top:0;font-size:10pt}
table{border-collapse:collapse;margin:6px 0 10px;width:100%}
th,td{border:1px solid #c8cdd6;padding:4px 7px;text-align:left;vertical-align:top;font-size:9.5pt}
th{background:#eef1f6}
code{background:#f3f4f6;padding:0 3px;border-radius:3px;font-size:9.5pt}
pre{background:#f3f4f6;padding:8px 10px;border-radius:4px;font-size:9.5pt}
li{margin:2px 0}
h1,h2{break-after:avoid;page-break-after:avoid}
tr,pre{break-inside:avoid;page-break-inside:avoid}
"""


def main() -> None:
    from playwright.sync_api import sync_playwright  # only needed for this helper

    src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    html = subprocess.run(["pandoc", str(src), "-s", "--metadata", "pagetitle=doc", "-t", "html5"],
                          capture_output=True, text=True, check=True).stdout
    html = html.replace("</head>", f"<style>{CSS}</style></head>")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.set_content(html)
        pg.pdf(path=str(dst), format="A4", margin=dict(top="16mm", bottom="16mm", left="16mm", right="16mm"),
               display_header_footer=True, header_template="<span></span>",
               footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span> / <span class='totalPages'></span></div>")
        b.close()
    print("wrote", dst)


if __name__ == "__main__":
    main()
