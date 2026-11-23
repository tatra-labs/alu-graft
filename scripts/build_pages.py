"""Wrap docs/demo_body.html (the artifact fragment) into docs/index.html for GitHub Pages."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "docs"
body = (D / "demo_body.html").read_text(encoding="utf-8")
cut = body.index('<div class="wrap">')
head, rest = body[:cut], body[cut:]
html = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"description\" content=\"The extra weights grafted into Qwen3-0.6B, running in your browser: "
        "an exact 6-digit adder/subtractor built with zero gradients.\">\n"
        "<style>html{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
        + head + "</head>\n<body>\n" + rest + "</body>\n</html>\n")
(D / "index.html").write_text(html, encoding="utf-8")
print("wrote", D / "index.html")
