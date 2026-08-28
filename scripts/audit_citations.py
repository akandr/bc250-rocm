#!/usr/bin/env python3
"""A quoted string should live in the directory the sentence cites.

Two findings of the same shape prompted this. A sentence gave one experiment's
identity and another's coverage, and a front-page sentence attributed a
netconsole capture to a directory whose own page says nothing survived in its
journal. Both were true statements citing the wrong directory, so no figure or
link audit could see them: the figures were backed, the links resolved, and only
the pairing was wrong.

This pairs each link into logs/<dir>/ with the backticked strings near it and
asks whether that directory contains them. A miss is not automatically a defect,
since a sentence often quotes a string it is contrasting with or naming from
source code, so the output is a list to read.

On a clean tree it reports four, and knowing which is which is what makes a
fifth worth looking at:

  - `smu send message` against reset-dyndbg, which is real and already
    disclosed on that page: the capture it shipped is from another experiment.
    This is the case the audit was written to catch, and it still catches it.
  - `rc=134 FAILED` against historical-sources/inv39, which belongs to the
    context-2026-08-14 link earlier in the same sentence and is present there.
  - `STATS <name> <op> n= sum= sumsq= maxabs=`, a description of the format an
    instrument prints rather than a line it printed.
  - `sched_policy=2` against kernel-7.1.5, a boot setting rather than output.

So a new entry means a new pairing to check, not necessarily a new defect.
"""
import glob, os, re, subprocess, sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

LINK = re.compile(r"\]\(logs/([A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)/?\)")
# Match any code span and filter by length afterwards. With the length bound
# inside the pattern, a short span like `0x34` fails to match at its own
# backtick, the scanner resumes at that span's closing backtick, and the next
# match runs from there to the following span's opening one, so the "quote"
# reported is the prose between two code spans. That produced four phantom
# entries, two of them in text written to fix something else.
CODE = re.compile(r"`([^`\n]{1,200})`")
# Strings that are code, paths or parameters rather than captured output.
# Boot parameters and environment settings are configuration, not captured
# output: a page names the command line it ran under without that string having
# to appear in its logs. They were the dominant false positive.
SKIP = re.compile(r"^(scripts/|patches/|logs/|\.\./|~/|https?:)|\.(py|sh|c|cpp|h|md|log|txt|bin|xz)$"
                  r"|^[A-Za-z_]+\(\)$|^-|^[A-Z_]+=[^ ]*$"
                  r"|^(amdgpu|ttm)\.|^[A-Z][A-Z0-9_]+=|\]\(|^#")

targets = ["README.md", "INVESTIGATION.md"] + sorted(
    glob.glob("logs/**/*.md", recursive=True))
rows = []
for t in targets:
    lines = open(t, errors="ignore").read().split("\n")
    for i, line in enumerate(lines):
        for m in LINK.finditer(line):
            d = os.path.join("logs", m.group(1))
            if not os.path.isdir(d):
                continue
            # Same sentence only. A wider window pairs a quote with whichever
            # link happens to be nearby, which is how two adjacent citations of
            # different directories produced a false hit on the first run.
            joined = " ".join(lines[max(0, i - 2):i + 2])
            pos = joined.find(m.group(0))
            left = joined.rfind(". ", 0, pos) + 1
            right = joined.find(". ", pos)
            window = joined[left:right if right > 0 else len(joined)]
            # Only pair a quote with this link when no other logs/ citation
            # sits between them. A sentence that quotes one directory's output
            # and then cites a second directory for a different fact otherwise
            # reports the first quote against the second directory, which is
            # exactly the mistake this audit is looking for and would have
            # manufactured three of them.
            for qm in CODE.finditer(window):
                q = qm.group(1).strip()
                if not (8 <= len(q) <= 80):
                    continue
                if SKIP.search(q) or not re.search(r"[ :=]", q):
                    continue
                lpos = window.find(m.group(0))
                between = window[min(lpos, qm.start()):max(lpos, qm.end())]
                if len(LINK.findall(between)) > 1:
                    continue
                r = subprocess.run(["grep", "-rqaF", "--exclude=README.md", q, d])
                if r.returncode != 0:
                    rows.append((t, m.group(1), q, line.strip()[:70]))

seen = set()
out = []
for t, d, q, ctx in rows:
    if (t, d, q) in seen:
        continue
    seen.add((t, d, q))
    out.append((t, d, q, ctx))
print(f"{len(out)} quoted string(s) not found in the directory cited beside them")
for t, d, q, ctx in out:
    print(f"  [{t}] cites logs/{d}/")
    print(f"      quotes: {q}")
