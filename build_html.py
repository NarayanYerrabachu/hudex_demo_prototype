"""Inject fresh findings.json into hudex_demo.html, replacing the inline DATA block."""
import json, re, sys

html_path = sys.argv[1] if len(sys.argv) > 1 else "hudex_demo.html"
json_path  = sys.argv[2] if len(sys.argv) > 2 else "findings.json"
out_path   = sys.argv[3] if len(sys.argv) > 3 else "hudex_demo.html"

with open(html_path) as f:
    html = f.read()

with open(json_path) as f:
    data = json.load(f)

replacement = "const DATA = " + json.dumps(data, indent=2) + ";"
html = re.sub(r"const DATA = \{.*?\};", lambda _: replacement, html, flags=re.DOTALL)

with open(out_path, "w") as f:
    f.write(html)

print(f"injected {json_path} → {out_path}")
