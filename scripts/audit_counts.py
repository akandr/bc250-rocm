#!/usr/bin/env python3
"""Check counted claims, which the figure audit structurally cannot see.

`audit_figures.py` matches numbers with two to four decimal places. A claim like
"five of five" or "16 of 17" has no decimal point, so it was never checked, and
the blind spot turned out to matter: an interleaved A/B quoted for weeks as
"flush-off arm faulted five of five, flush-on arm clean five of five" had no
retained logs at all, and a reboot campaign quoted as "sixteen of seventeen
boots" disagreed with the only surviving contemporaneous note, which said ten of
eleven. Both were corrected once this audit existed to find them.

Counted claims are harder to verify mechanically than figures. A perplexity value
appears verbatim in a log; "three of three" is usually a summary the author
computed. So this reports where each claim is and whether the same pair appears
anywhere under logs/, and leaves the judgement to a reader. It is a list to work
through, not a pass or fail.

Run from anywhere.
"""
import glob
import os
import re
import subprocess

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

WORD = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
        "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
        "twenty": 20}
NUM = r"(?:\d+|" + "|".join(WORD) + r")"
PAT = re.compile(r"\b(" + NUM + r")\s+(?:of|out of)\s+(" + NUM + r")\b", re.I)

def value(tok):
    tok = tok.lower()
    return int(tok) if tok.isdigit() else WORD.get(tok)

def digits(tok):
    v = value(tok)
    return str(v) if v is not None else tok

claims = {}
# Counted claims appear on log pages too, and those were outside this check
# entirely: "3 of 5 boots" in a write-up got no more scrutiny than a
# sentence with no number in it. The two top documents were all it read.
DOCS = ["README.md", "INVESTIGATION.md"] + sorted(
    glob.glob("logs/**/*.md", recursive=True))

for f in DOCS:
    for n, line in enumerate(open(f).read().split("\n"), 1):
        if line.startswith("|"):
            continue
        for m in PAT.finditer(line):
            a, b = value(m.group(1)), value(m.group(2))
            if a is None or b is None or b == 0 or a > b:
                continue          # "one of two ways", "ten of one" and similar prose
            claims.setdefault((a, b), []).append("%s:%d" % (f, n))

# Bare integer counts are the third blind spot, and the one that has cost most.
# audit_figures.py skips whole numbers deliberately, because matching them would
# drown the output in version numbers, line counts and byte sizes, and this audit
# only caught the "N of M" shape. That left "439 iterations", "253 rounds" and
# "twenty boots" unchecked, and the first of those turned out to be quoted from
# working notes whose logs were never kept. Narrowing to an integer immediately
# followed by a unit of experimental work makes the class checkable without the
# noise.
UNITS = r"(?:iterations?|rounds?|runs?|boots?|attempts?|trials?|repetitions?|sweeps?)"
# "N=2048 runs at about 456 GFLOP/s" is not a count of 2048 runs: "runs" is the
# verb. Exclude a number introduced by "=" and a unit word followed by a verb's
# preposition, which removes that whole shape without hiding real counts.
BARE = re.compile(r"(?<![\w.=])(\d{2,6})\s+" + UNITS + r"\b(?!\s+(?:at|in|on)\b)", re.I)

bare = {}
for f in DOCS:
    for n, line in enumerate(open(f).read().split("\n"), 1):
        if line.startswith("|"):
            continue
        for m in BARE.finditer(line):
            bare.setdefault(m.group(0).lower(), []).append("%s:%d" % (f, n))

# Searching logs for a bare integer proves nothing: any two-to-six digit number
# finds a coincidental match somewhere, and 439 duly matched a benchmark log that
# has no connection to the soak it was quoted from. So these are listed for a
# reader to judge rather than filtered by a test that cannot work. The question
# to ask of each is whether the write-up says where the count came from.

unfound = []
for (a, b), where in sorted(claims.items()):
    # Look for the pair in captured output, in either digit or word form, the
    # way the write-up might have recorded it.
    pats = [r"%d\s*(?:of|/|out of)\s*%d" % (a, b)]
    for k, v in WORD.items():
        if v == a:
            for k2, v2 in WORD.items():
                if v2 == b:
                    pats.append(r"%s\s+(?:of|out of)\s+%s" % (k, k2))
    hit = ""
    for p in pats:
        r = subprocess.run(["grep", "-rlaiE", p, "logs"],
                           capture_output=True, text=True)
        if r.stdout.strip():
            hit = r.stdout.strip().split("\n")[0]
            break
    if not hit:
        unfound.append((a, b, where))

for a, b, where in unfound:
    print("%-12s found nowhere under logs/   %s" % ("%d of %d" % (a, b), where[0]))
print()
print("bare counts, listed for judgement rather than tested (see the note above):")
for phrase, where in sorted(bare.items()):
    print("  %-24s %s" % (phrase, ", ".join(where[:2])))
print()
print("%d counted claim(s), %d with no match under logs/; %d bare counts listed"
      % (len(claims), len(unfound), len(bare)))
