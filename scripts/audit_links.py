#!/usr/bin/env python3
"""Check the markdown links in README.md and INVESTIGATION.md.

Four failure modes, all of which have occurred here. A link can point at a path
that does not exist, which is the obvious one. A link can also be broken by line
wrapping, where the target is split across a newline and renders as literal text
rather than a link; a checker that scans line by line cannot see that, and two
such links were shipped before this script existed. Rewrapping later introduced
three more kinds: a repository path quoted in inline code that resolves\nnowhere, a link label split across lines, which still resolves and so
passes a target-only check while rendering with a visible space, and an ordinary
hyphenated word broken at the hyphen.

Run from the repository root.
"""
import glob
import os
import re, os, sys

# Run from anywhere: this resolves to the repository root the same way
# audit_logs.sh does. Without it, every glob below matches nothing and the audit
# reports a clean result while having checked no files at all, which is how the
# environment-variable audit went several review passes reporting fourteen
# findings that were really zero.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# The two top documents were all this checked for as long as it existed, so the
# links on 136 log pages, and every relative path they use to reach a sibling
# directory or a script three levels up, were never validated. They resolve, but
# nothing was keeping them resolving: the same one-level blind spot the orphan,
# formatting and figure audits each turned out to have.
bad = []
targets = ["README.md", "INVESTIGATION.md"]
targets += sorted(glob.glob("logs/**/*.md", recursive=True))
targets = sorted(dict.fromkeys(os.path.normpath(t) for t in targets))
for f in targets:
    s = open(f, errors="ignore").read()
    # links broken across a newline inside the target
    for m in re.finditer(r"\]\(([^)]*\n[^)]*)\)", s):
        bad.append((f, "TARGET SPLIT ACROSS LINES", m.group(1).replace("\n", "\\n")[:70]))
    # A backticked path used as a link label must not contain whitespace: line
    # wrapping inside one renders as a visible space and the target still
    # resolves, so a target-only check cannot see it.
    for m in re.finditer(r"\[`([^`]*\s[^`]*)`\]", s):
        bad.append((f, "LABEL SPLIT ACROSS LINES", " ".join(m.group(1).split())[:70]))
    # A prose line ending in a hyphen is a word broken by wrapping.
    for m in re.finditer(r"(?m)(?<=\w)-$", s):
        line = s[:m.end()].split("\n")[-1]
        bad.append((f, "WORD BROKEN BY WRAPPING", line[-50:]))
    # image embeds use the same target syntax and are checked the same way
    # Resolve relative to the file holding the link, not to the repository root.
    # Those were the same thing while this read only the two top documents; on a
    # log page "../sibling/" is correct and root-relative resolution calls it
    # missing, which is how extending the file list produced six false reports
    # before this line was fixed.
    here = os.path.dirname(f) or "."
    for m in re.finditer(r"!?\]\(([A-Za-z0-9._/#-]+)\)", s):
        t = m.group(1)
        if t.startswith("#") or t.startswith("http"): continue
        if not os.path.exists(os.path.normpath(os.path.join(here, t.split("#")[0]))):
            bad.append((f, "MISSING TARGET", t))
    # A repository path quoted in inline code is a reference a reader will
    # follow even though it is not a markdown link, so it needs the same check.
    # Anything that looks like one of our own directories and resolves nowhere
    # is either a typo or a shorthand that does not exist on disk.
    # Paths belonging to another project. They are quoted here because the work
    # used them, and the prose says whose they are, so they cannot resolve.
    FOREIGN = {"patches/bc250-sdma-trap-instrumentation.patch"}
    for m in re.finditer(r"`((?:logs|scripts|patches|figures)/[A-Za-z0-9._/-]+)`", s):
        t = m.group(1).rstrip("/")
        if t in FOREIGN:
            continue
        if os.path.exists(t) or os.path.exists(t + "/"):
            continue
        # a numbered-prefix shorthand is acceptable only if exactly one file matches
        d, base = os.path.split(t)
        matches = [x for x in os.listdir(d)] if os.path.isdir(d) else []
        if sum(1 for x in matches if x.startswith(base)) == 1:
            continue
        bad.append((f, "INLINE PATH RESOLVES NOWHERE", t))

print(f"{len(bad)} link problems")
for f, k, t in bad: print(f"  [{f}] {k}: {t}")
