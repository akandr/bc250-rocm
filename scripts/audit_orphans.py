#!/usr/bin/env python3
"""Every log directory has to be reachable from a document.

Nested directories are reported separately and do not fail the audit, because a
parent page often names its cells by a scheme rather than by directory name: the
factorial's README says the boots run in the order A B C D D C B A and that each
numbered directory is one boot, which names A_1 through D_5 exactly without
containing any of those strings. A script cannot read that. The separate list is
still worth an eye, since it is what surfaced a result directory no document
linked and a byte-identical copy of one cell left at a doubled path.

A directory that no page links to is invisible: the work is on disk and nobody
can find it. This audit found nine such directories, all of them real, with
READMEs, simply never added to an index. It reports any log directory that no
Markdown file outside itself refers to, and any script that no document
mentions.
"""
import glob
import os
import re
import sys

# Run from anywhere: this resolves to the repository root the same way
# audit_logs.sh does. Without it, every glob below matches nothing and the audit
# reports a clean result while having checked no files at all, which is how the
# environment-variable audit went several review passes reporting fourteen
# findings that were really zero.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

docs = []
# Nested log directories carry READMEs too, and leaving them out had two
# effects: a script referenced only from one read as orphaned, and the nested
# directories themselves were never checked for reachability at all, since the
# glob below stopped at one level.
for pat in ("*.md", "logs/*/README.md", "logs/*/*/README.md"):
    docs += glob.glob(pat, recursive=True)

orphan_dirs = []
nested_dirs = []
for d in sorted(glob.glob("logs/*/") + glob.glob("logs/*/*/")):
    name = os.path.relpath(d.rstrip("/"), "logs")
    # A mention in prose is not navigation; require an actual link into the
    # directory, either to it or to a file inside it. A nested directory is
    # normally linked from its own parent with a relative href, so accept the
    # last path component too when the referring file is that parent.
    needle = "](logs/%s/" % name
    parent = os.path.join(os.path.dirname(d.rstrip("/")), "README.md")
    base = os.path.basename(name)
    rel = "](%s/" % base
    nested = os.sep in name
    # Cells are often named on the parent page by the part that identifies
    # them rather than by the whole directory name: a kernel rung directory
    # 6.18.16-200.fc43.x86_64 is "6.18.16" in the table that lists it. Accept
    # that leading component, but only from the parent page itself.
    stem = base.split("-")[0] if len(base.split("-")[0]) >= 4 else ""
    hit = False
    for f in docs:
        if os.path.abspath(f) == os.path.abspath(d + "README.md"):
            continue
        text = open(f).read()
        if needle in text:
            hit = True
            break
        # A nested directory is one cell of an experiment, and its parent
        # page normally enumerates the cells in a table by name rather than
        # linking each one. That is navigation enough here: the reader is
        # already in the right directory. So for a nested directory accept any
        # mention of its own name on the parent page, linked or not, and only
        # report one the parent never names.
        if nested and (
                rel in text or base in text or
                (stem and stem in text and os.path.abspath(f) == os.path.abspath(parent))):
            hit = True
            break
    if not hit:
        (nested_dirs if nested else orphan_dirs).append(name)

orphan_scripts = []
for s in sorted(glob.glob("scripts/*")):
    if os.path.basename(s).startswith("audit_"):
        continue
    if not any(os.path.basename(s) in open(f).read() for f in docs):
        orphan_scripts.append(os.path.basename(s))

for n in orphan_dirs:
    print("orphan log directory, linked from nothing: %s" % n)
for n in nested_dirs:
    print("nested, parent does not name it: %s" % n)
for n in orphan_scripts:
    print("orphan script, mentioned by nothing: %s" % n)
print("%d orphan log directories, %d orphan scripts, %d nested to eyeball"
      % (len(orphan_dirs), len(orphan_scripts), len(nested_dirs)))
sys.exit(1 if orphan_dirs or orphan_scripts else 0)
