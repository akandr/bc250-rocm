#!/usr/bin/env python3
"""Check that every distinctive figure quoted in this repository still has a log behind it.

Three figures turned out to be cited from memory, with no surviving artifact
anywhere: a perplexity for the architecture-macro-removed build, one for the
precision-patch-removed build, and one for the integrated-flag-reverted build.
All three were load-bearing, since each is the evidence that its defect is real
rather than an artifact of the environment variable or flag that masks it. None
was caught by re-reading, and the third was caught only when this script was
widened past README.md.

That widening is the point. Patch headers and the notes prepared for upstream
carry figures too, and those leave the repository, so they are checked here as
well. Some misses are legitimate (derived means, percentages, a DOI), so the
output is a list to check rather than a list of errors.

Known blind spot, stated because it let one through. The pattern matches figures
with two to four decimal places, so a whole number is invisible to it. The
integrated-flag patch header cites a corrupted perplexity of "167" with no
backing log, and this script does not flag it; that one was found by reading the
patch. Matching bare integers would drown the output in version numbers, line
counts and byte sizes, so the limitation is documented rather than fixed, and
whole-number claims need checking by hand.

Usage: run from the repository root. Pass --all to include patch headers and
any drafts directory found beside the repository.
"""
import re, subprocess, sys, os, glob

def figures(path):
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    out = {}
    for line in text.split("\n"):
        if line.startswith("    ") or line.startswith("+++") or line.startswith("---"):
            continue
        for m in re.findall(r"(?<![\w.\-/])(\d+\.\d{2,4})(?![\w.])", line):
            if float(m) >= 1:
                out.setdefault(m, (path, line.strip()[:120]))
    return out

targets = ["README.md", "INVESTIGATION.md"]
if "--all" in sys.argv:
    targets += sorted(glob.glob("patches/**/*.patch", recursive=True))
    drafts = os.path.join("..", "validation-2026-08", "drafts")
    if os.path.isdir(drafts):
        targets += sorted(glob.glob(os.path.join(drafts, "*.md")))

allfigs = {}
for t in targets:
    for k, v in figures(t).items():
        allfigs.setdefault(k, v)

print(f"{len(allfigs)} distinctive figures across {len(targets)} file(s)")
missing = []
for n, (src, ctx) in allfigs.items():
    r = subprocess.run(["grep", "-rlaF", n, "logs"], capture_output=True, text=True)
    if not r.stdout.strip():
        missing.append((n, src, ctx))
print(f"\nnot found anywhere in logs/ ({len(missing)}):")
for n, src, ctx in sorted(missing, key=lambda x: float(x[0])):
    print(f"  {n:>10}  [{src}]  {ctx}")
