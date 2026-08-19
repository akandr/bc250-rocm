#!/usr/bin/env python3
"""Check the markdown links in README.md and INVESTIGATION.md.

Two failure modes, both of which have occurred here. A link can point at a path
that does not exist, which is the obvious one. A link can also be broken by line
wrapping, where the target is split across a newline and renders as literal text
rather than a link; a checker that scans line by line cannot see that, and two
such links were shipped before this script existed.

Run from the repository root.
"""
import re, os, sys
bad = []
for f in ("README.md", "INVESTIGATION.md"):
    s = open(f).read()
    # links broken across a newline inside the target
    for m in re.finditer(r"\]\(([^)]*\n[^)]*)\)", s):
        bad.append((f, "SPLIT ACROSS LINES", m.group(1).replace("\n", "\\n")[:70]))
    # image embeds use the same target syntax and are checked the same way
    for m in re.finditer(r"!?\]\(([A-Za-z0-9._/#-]+)\)", s):
        t = m.group(1)
        if t.startswith("#") or t.startswith("http"): continue
        if not os.path.exists(t.split("#")[0]):
            bad.append((f, "MISSING TARGET", t))
print(f"{len(bad)} link problems")
for f, k, t in bad: print(f"  [{f}] {k}: {t}")
