#!/usr/bin/env python3
"""Check that every distinctive figure quoted in this repository still has a log behind it.

Three figures turned out to be cited from memory, with no surviving artifact
anywhere: a perplexity for the architecture-macro-removed build, one for the
precision-patch-removed build, and one for the integrated-flag-reverted build.
All three were load-bearing, since each is the evidence that its defect is real
rather than an artifact of the environment variable or flag that masks it. None
was caught by re-reading, and the third was caught only when this script was
widened past README.md.

That widening is the point. Patch headers carry figures too, so they are checked here as well. So
does `scripts/make_figures.py`, because its literals become images and
an image is opaque to every check in this repository: a figure claimed "same
build, same boot" for weeks under a paragraph correcting that very phrase, and
five GEMM throughputs in another had nothing behind them, both found by opening
the PNGs rather than by any audit. Some misses are legitimate (derived means, percentages, a DOI), so the
output is a list to check rather than a list of errors.

Known blind spot, stated because it let one through. The pattern matches figures
with one to four decimal places, so a whole number is invisible to it. The
integrated-flag patch header cites a corrupted perplexity of "167" with no
backing log, and this script does not flag it; that one was found by reading the
patch. Matching bare integers would drown the output in version numbers, line
counts and byte sizes, so the limitation is documented rather than fixed, and
whole-number claims need checking by hand.

The one-decimal case used to be invisible too, and was widened in on 26 August
after a figure with a single decimal turned out to be quoting a different run.
The widening roughly doubled what this script examines, from about 760
occurrences to about 1065, and it immediately reached a table column whose four
values have no capture behind them. Anything narrowed for output volume is worth
re-testing on that evidence rather than assumed harmless.

A third limitation, in the disclosure window rather than the pattern. The window
is a few lines either side of the figure, which suits hard-wrapped prose but not
a table: a table's provenance note sits below the whole table, six or more rows
from its first value, so figures disclosed there are still reported as missing.
Widening the window globally would be worse, since it would let an unrelated
retraction nearby mark a live figure as conceded. The pre-fix `-fa off` column in
the decode-ladder table is the standing example, disclosed in the paragraph
under it and reported here regardless.

A caution on the rounding fallback below, which produces false positives at the
midpoint. Python rounds half to even and binary floats fall either side of a
decimal midpoint, so a captured 178.45 does not round to a quoted 178.5 and the
figure is reported as missing when its log is sitting there. That happened, and
the risk is not the noise but the response to it: correcting a figure and
establishing a figure are different operations, and a midpoint miss looks exactly
like a real one. Check the candidate list this prints before believing a miss.

A second blind spot, same reason. A harness log often prints the reference it is
comparing against, as in "(7.1.5: pp512 805.53, tg64 113.50)". That line quotes a
figure; it does not measure one. This script counts it as an artifact, which is
how a prefill reference used in four write-ups passed while no run producing it
survives anywhere. Telling a quoted reference from a measurement inside a log
without excluding real captures is not something this check can do, so it is
documented rather than fixed: a figure whose only hit is inside a header or a
parenthetical needs looking at by hand.

Usage: run from the repository root. Pass --all to include patch headers.
"""
import os
import re, subprocess, sys, os, glob

# Run from anywhere: this resolves to the repository root the same way
# audit_logs.sh does. Without it, every glob below matches nothing and the audit
# reports a clean result while having checked no files at all, which is how the
# environment-variable audit went several review passes reporting fourteen
# findings that were really zero.
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def figures(path):
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return {}
    out = {}
    # These documents are hard-wrapped, so the sentence that discloses a figure
    # as unbacked often ends on the following line. Judge disclosure over a
    # two-line window; still show the figure's own line.
    lines = text.split("\n")
    # The indent skip exists to ignore fenced code blocks in Markdown. In a
    # Python source file it skips exactly the lines that matter, since a data
    # table is indented inside its dict, so applying it there made reading the
    # figure generator almost pointless: fifteen figures instead of the whole
    # table. Skip by file type rather than by shape.
    md = path.endswith(".md") or path.endswith(".patch")
    for i, line in enumerate(lines):
        if md and (line.startswith("    ") or line.startswith("+++")
                   or line.startswith("---")):
            continue
        # a DOI is an identifier, not a measurement
        line = re.sub(r"10\.\d{4,9}/\S+", " ", line)
        # These documents hard-wrap near 100 characters, so the sentence that
        # discloses a figure as unbacked routinely runs three or four lines past
        # it. A one-line window on either side missed every such disclosure and
        # reported figures that the prose had already conceded.
        window = " ".join(lines[max(0, i - 3):i + 4])
        for m in re.findall(r"(?<![\w.\-/])(\d+\.\d{1,4})(?![\w.])", line):
            # A power of ten written with a trailing .0 is a unit conversion,
            # not a reading: the figure generator divides by 1000.0 to plot
            # TFLOP/s. Reading source files brought these in; they are the only
            # false positives that did.
            if re.fullmatch(r"10*\.0", m):
                continue
            if float(m) >= 1:
                out.setdefault(m, (path, line.strip()[:120], window))
        # Scientific notation was invisible: the pattern above needs two to four
        # decimals, so 1.799e-05 was caught by its mantissa alone while 1.1e-8
        # and 3.4e-7 were not caught at all. Tolerances and agreements are
        # written this way throughout, and they are claims like any other.
        for m in re.findall(r"(?<![\w.\-/])(\d+\.\d+e[-+]?\d+)(?![\w.])", line):
            out.setdefault(m, (path, line.strip()[:120], window))
    return out

# The figure generator holds its data as literals, and those literals become
# images that no check here can read. Two defects lived in exactly that gap: a
# figure asserting "same boot" under a paragraph correcting that claim, and five
# GEMM throughputs backed by nothing. Reading the generator is the only way to
# audit a PNG's contents from text.
targets = ["README.md", "INVESTIGATION.md", "scripts/make_figures.py"]
# The README inside each log directory makes claims too, and figures quoted only
# there were invisible to this check for as long as it read the top two
# documents alone. A set of throughput figures survived that way with nothing
# behind them, in the write-up of a mitigation.
targets += sorted(glob.glob("logs/*/README.md"))
# Nested log pages state figures too, and were outside this check entirely: a
# re-measurement written up in a sub-directory could quote anything.
targets += sorted(glob.glob("logs/*/*/README.md"))
targets += sorted(glob.glob("logs/*/*.md"))
targets = sorted(dict.fromkeys(os.path.normpath(t) for t in targets))
if "--all" in sys.argv:
    targets += sorted(glob.glob("patches/**/*.patch", recursive=True))

# Keep every occurrence, not the first. Recording only the first file that
# mentions a figure meant a value disclosed as retracted in one document counted
# as disclosed everywhere: INVESTIGATION.md says plainly that 20.64 came from a
# run that was never kept, and an earlier write-up went on quoting it as a result
# with nothing flagging that. Disclosure has to be judged where the figure is
# used, not globally.
allfigs = {}
for t in targets:
    for k, v in figures(t).items():
        allfigs.setdefault((t, k), v)

print(f"{len(allfigs)} figure occurrence(s) across {len(targets)} file(s)")
missing = []
derived = []
disclosed = []
for (_tgt, n), (src, ctx, window) in allfigs.items():
    # A plain substring search lets a figure pass on a coincidental match inside
    # a longer number: "35.40" is a substring of "35.404118" in a tensor dump,
    # and "37.34" of "737.341". Both passed this check for weeks while having no
    # artifact behind them. Require a numeric boundary on each side.
    # Some tools on this board print with a comma decimal separator, because the
    # shell runs under a Polish locale: llama-cli writes "Generation: 115,3 t/s"
    # where the write-up says 115.3. Searching for the dotted form alone reports
    # such a figure as unbacked when its log is sitting right there, which is how
    # two correct figures in corpus-instrument came to look unsupported.
    # A second printing habit, from bc rather than the locale: a value below one
    # comes out as ".97" with no leading zero, so a page saying 0.97 finds
    # nothing in its own sweep. That is what hid the ratio column of
    # sdma-sizes-2026-08-19 from this check.
    forms = {n, n.replace(".", ",")}
    if n.startswith("0."):
        forms.add(n[1:])
        forms.add(n[1:].replace(".", ","))
    pat = (r"(^|[^0-9.])(" + "|".join(re.escape(f) for f in sorted(forms)) +
           r")($|[^0-9])")
    # Search captured output only. The README in each log directory is prose I
    # wrote, so counting it as an artifact makes the check circular: a figure
    # stated in a log README would validate itself. Two figures passed for weeks
    # that way, their only occurrence anywhere being the sentence asserting them.
    # Per-tensor instrumentation dumps hold about 79000 arbitrary floats between
    # them and not one result line, so they cannot be evidence for a throughput
    # or perplexity figure, while being large enough that almost any two-decimal
    # figure finds a chance match inside them. Excluded from both searches.
    EX = ["--exclude=README.md", "--exclude=*-stats.txt"]
    r = subprocess.run(["grep", "-rlaE"] + EX + [pat, "logs"],
                       capture_output=True, text=True)
    if not r.stdout.strip():
        # The write-up rounds: a perplexity quoted as 14.09 is 14.0876 in the
        # log. Accept any captured number that rounds to the figure at the
        # figure's own precision, so rounding is not reported as a missing
        # artifact. Checked numerically, since no string pattern expresses it.
        whole, _, frac = n.partition(".")
        cand = subprocess.run(
            ["grep", "-rhoaE"] + EX +
            [r"(^|[^0-9.])" + re.escape(whole) + r"\.[0-9]+", "logs"],
            capture_output=True, text=True).stdout
        target = float(n)
        for c in re.findall(r"[0-9]+\.[0-9]+", cand):
            if round(float(c), len(frac)) == target:
                r = subprocess.CompletedProcess(r.args, 0, stdout="rounded:" + c, stderr="")
                break
    if not r.stdout.strip():
        # A figure absent from captured output but present in a log directory's
        # own README is a derived one: a mean, a spread, a difference. That is
        # legitimate but unverifiable by search, so it is reported separately
        # rather than as missing. Only a figure appearing in no file at all is
        # unsupported, which is the case worth acting on.
        inprose = subprocess.run(
            ["grep", "-rlaE", "--include=README.md", pat, "logs"],
            capture_output=True, text=True).stdout.strip()
        # A figure the prose itself marks as unbacked is not a miss. Retracted
        # and superseded values are quoted deliberately, to say what an earlier
        # revision claimed and that nothing supports it, and having no artifact
        # is exactly the point being made. Recognised only from the same line,
        # and only from wording that concedes the absence, so a figure asserted
        # as a result still fails.
        # A correction concedes an absence as plainly as a retraction does, and
        # the corrections written into this repository use their own vocabulary:
        # "used to quote", "previously read", "was not kept". Without these, the
        # act of fixing a figure and saying what it used to be made the audit
        # report the superseded value forever, which trains the reader of this
        # output to skip entries rather than check them.
        DISCLOSED = ("never shipped", "an earlier revision quoted",
                     "an earlier draft", "never kept", "not retained",
                     "survive in no", "survives in no", "cited from memory",
                     "no surviving artifact", "with no backing log",
                     "used to quote", "used to say", "used to read",
                     "previously read", "previously gave", "previously quoted",
                     "not kept", "not what this run measured", "no log behind it",
                     "appearing in any capture", "appear in no capture",
                     "has no capture", "no capture supports", "recollections")
        if any(d in window for d in DISCLOSED):
            disclosed.append((n, src, ctx))
        else:
            (derived if inprose else missing).append((n, src, ctx))

print(f"\nno artifact and no stated derivation ({len(missing)}):")
for n, src, ctx in sorted(missing, key=lambda x: float(x[0])):
    print(f"  {n:>10}  [{src}]  {ctx}")
print(f"\ndisclosed by the prose as unbacked, no action ({len(disclosed)}):")
for n, src, ctx in sorted(disclosed, key=lambda x: float(x[0])):
    print(f"  {n:>10}  [{src}]  {ctx}")
print(f"\nderived, stated in a log README, verify the inputs ({len(derived)}):")
for n, src, ctx in sorted(derived, key=lambda x: float(x[0])):
    print(f"  {n:>10}  [{src}]  {ctx}")
