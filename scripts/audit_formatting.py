#!/usr/bin/env python3
"""Catch the damage that automated edits leave in prose.

Two consecutive review passes found the same class of problem by reading: a
References section with six doubled "- -" list markers and spaces stranded before
colons, and punctuation separated from the link it belonged to. A rewrap written
during one of those passes made it worse, stranding bare "and" and "," on their
own lines, and was reverted.

None of it is visible to the other audits. Links still resolve, figures still have
artifacts, every log directory still has a README. Only reading catches it, which
is why it survived for weeks, so it is worth a detector.

Quoted material is exempt. Upstream commit titles genuinely read
"ggml-cuda : restore prop.integrated", and error text genuinely reads "Illegal
seek for GPU arch : gfx1013"; flagging those would train the reader to ignore
this audit.
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

PATTERNS = [
    # The trailing space in the original pattern was load-bearing in the wrong
    # direction: a doubled marker whose item wrapped onto the next line reads
    # "- -" with nothing after it, and went unreported in the table of contents
    # for as long as this check existed. Match the end of the line too.
    ("doubled list marker", re.compile(r"^\s*- -(\s|$)")),
    ("space before punctuation", re.compile(r"\S \s*[,.;:](\s|$)")),
    ("space before closing paren", re.compile(r"\S \)")),
    ("orphaned tiny line", re.compile(r"^\s{0,3}(and|in|of|the|a|to|,|\.|;)\s*$")),
    ("double space mid-sentence", re.compile(r"[a-z]\s{2,}[a-z]")),
    # This repository writes without em or en dashes, and without typographic
    # quotes and ellipses, so that its prose reads as written rather than as
    # passed through something. The rule held everywhere in the repository and
    # had drifted only in three older working notes outside it, which is exactly
    # the sort of thing a rule needs a check to keep true.
    ("em or en dash", re.compile(r"[\u2013\u2014]")),
    ("typographic quote or ellipsis", re.compile(r"[\u2018\u2019\u201c\u201d\u2026]")),
]

def strip_quoted(line, in_quote):
    """Blank out backticked and double-quoted spans, preserving adjacency.

    Quotes are tracked across lines, because these documents wrap and a quoted
    commit title routinely opens on one line and closes on the next.

    The replacement is a single word character, never whitespace and never
    nothing. An earlier version deleted the span and rejoined with spaces, which
    manufactured exactly the artefacts this audit looks for: `as "some quote",
    and` became `as  , and`, reporting both a double space and a space before
    punctuation on a line that was correct.
    """
    line = re.sub(r"`[^`]*`", "Q", line)
    out = []
    for k, part in enumerate(line.split('"')):
        # split alternates outside/inside, starting outside unless we entered
        # this line already inside a quote.
        inside = (k % 2 == 1) if not in_quote else (k % 2 == 0)
        out.append("Q" if inside else part)
    if line.count('"') % 2 == 1:
        in_quote = not in_quote
    return "".join(out), in_quote


targets = ["README.md", "INVESTIGATION.md"]
targets += sorted(glob.glob("logs/*/README.md"))
# Nested log directories carry READMEs too and were outside every audit here
# until 26 August; leaving them out of this one would mean prose damage in a
# sub-experiment's page goes unreported.
targets += sorted(glob.glob("logs/*/*/README.md"))
targets += sorted(glob.glob("logs/*/*.md"))

# The globs above overlap: logs/*/README.md and logs/*/*.md match the same
# files. Counting a file twice would make the "across N file(s)" line wrong,
# which is the number a reader checks a clean result against.
targets = sorted(dict.fromkeys(os.path.normpath(t) for t in targets))

hits = 0
for t in targets:
    fenced = False
    in_quote = False
    for n, raw in enumerate(open(t, encoding="utf-8", errors="ignore").read().split("\n"), 1):
        if raw.startswith("```"):
            fenced = not fenced
            continue
        if fenced or raw.startswith("|") or raw.startswith("    "):
            continue
        line, in_quote = strip_quoted(raw, in_quote)
        for name, p in PATTERNS:
            if p.search(line):
                print("%-27s %s:%d  %s" % (name, t, n, raw.strip()[:76]))
                hits += 1

print("%d formatting artefact(s) across %d file(s)" % (hits, len(targets)))
sys.exit(1 if hits else 0)
