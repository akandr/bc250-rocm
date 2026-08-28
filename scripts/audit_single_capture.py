#!/usr/bin/env python3
"""Flag pages that draw a conclusion from one capture without saying so.

This is the failure mode that cost the most time in this investigation. A single
netconsole capture of the reset showed prints letting the KIQ resume run further,
and that one run became a stated mechanism, then a build, then a falsified
prediction, then a second build that could not have measured anything. One
success out of one attempt is one sample, and saying so in the write-up is what
keeps the next person, including the author a day later, from building on it.

Netconsole makes this worse: it is UDP with no retransmission, so a dropped
packet and a stall produce the same evidence. "This line printed and nothing
after it did" assumes delivery, and only repetition separates the two.

Informational, not a gate. A directory can legitimately hold one artifact, and
plenty of them do. What is reported is the pair: one capture, and a README that
never concedes it.
"""
import glob
import os
import re

# Run from anywhere: this resolves to the repository root the same way
# audit_logs.sh does. Without it, every glob below matches nothing and the audit
# reports a clean result while having checked no files at all, which is how the
# environment-variable audit went several review passes reporting fourteen
# findings that were really zero.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

HEDGES = re.compile(
    r"\b(once|one run|single run|single capture|one capture|n = ?1|one sample|"
    r"not repeated|repetition|repeated|repeat|three times|twice|trials?)\b", re.I)
SUPPORT = (".md", ".py", ".sh", ".c", ".cpp", ".hip", ".patch", ".txt")

flagged = []
for d in sorted(glob.glob("logs/*/") + glob.glob("logs/*/*/")):
    readme = os.path.join(d, "README.md")
    if not os.path.exists(readme):
        continue
    # A file named exactly "log" is this repository's convention for a
    # harness's own progress log across many runs, so it is not one capture.
    caps = [f for f in sorted(glob.glob(d + "*"))
            if os.path.isfile(f) and not f.endswith(SUPPORT)
            and os.path.basename(f) != "log"]
    if len(caps) != 1:
        continue
    if not HEDGES.search(open(readme, encoding="utf-8", errors="ignore").read()):
        flagged.append((os.path.basename(d.rstrip("/")), os.path.basename(caps[0])))

for name, cap in flagged:
    print("one capture, and the page does not say so: %s (%s)" % (name, cap))
print("%d page(s) to check" % len(flagged))
