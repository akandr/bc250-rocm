#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Tables for logs/context-ceilings-2026-09-30/, from the files in that directory.

ceilings.tsv       one row per cell: deepest depth that generated, shallowest that failed, where the pass was
                   measured, its decode rate and fill time, and how the failure ended
kfd-accounting.txt for every ROCm pass of this campaign, KFD's "System mem used" once the context exists,
                   against TTM's count and the context length

Usage: ceilings_analyse.py logs/context-ceilings-2026-09-30
"""
import csv, io, os, re, sys, tarfile

D = sys.argv[1]
RUN = re.compile(r"\[(\S+ \S+)\] (\S+) (\S+) (\S+) d(\d+): (\S+) (\S+) (\d+)s rc=(\d+)(.*?) edge=.* swapped-out-pages=(\d+)")
RESULT = re.compile(r"RESULT (\S+) (\S+) (\S+): generates at (\d+), fails at (\d+)")
MEM = re.compile(r"(\d+) \| System mem used (\d+)M out of (\d+)M TTM mem used (\d+)M")

runs, cells = {}, []
for line in open(os.path.join(D, "results.log")):
    m = RUN.match(line)
    if m:
        end, arm, model, kv, d, res, rate, sec, rc, why, swo = m.groups()
        runs[(arm, model, kv, int(d))] = dict(res=res, rate=rate, sec=int(sec), rc=int(rc))
    m = RESULT.search(line)
    if m:   # a cell refined later appears twice; its last result stands
        cells = [c for c in cells if c[:3] != (m[1], m[2], m[3])] + [(m[1], m[2], m[3], int(m[4]), int(m[5]))]

earlier = {}
for name in ("compare-summary.tsv", "bisect-summary.tsv"):
    for r in csv.reader(open(os.path.join(D, "earlier-passes", name)), delimiter="\t"):
        if len(r) > 5 and r[4] == "ok":
            earlier[(r[0], r[1], r[2], int(r[3]))] = name.split("-")[0]

mem = {}
with tarfile.open(os.path.join(D, "runs-mem.tar.xz")) as t:
    for ti in t.getmembers():
        if ti.isfile():
            mem[os.path.basename(ti.name)[:-4]] = t.extractfile(ti).read().decode()


def run_log(arm, model, kv, d):
    p = os.path.join(D, "runs", f"{arm}_{model}_{kv}_d{d}.log")
    return open(p, errors="replace").read() if os.path.exists(p) else ""


def ended(arm, model, kv, d):
    """How a failing run ended, from its exit code and its log."""
    r, txt = runs[(arm, model, kv, d)], run_log(arm, model, kv, d)
    if r["rc"] == 137:
        return "OOM killer"
    if "failed to create context" in txt:
        return "context not created"
    if r["rc"] == 139:
        return "segfault"
    if r["rc"] == 134 and "ROCm error" in txt:
        # the first frame below ggml's own error path names the operation that failed
        frames = re.findall(r"#\d+\s+0x[0-9a-f]+ in (?:void )?([A-Za-z_][\w:]*)", txt)
        frames = [f for f in frames if not f.startswith(("ggml_print_backtrace", "ggml_abort", "ggml_cuda_error",
                                                        "wait4", "__syscall", "__internal"))]
        return "ROCm error in " + (frames[0].removesuffix("_impl") if frames else "?")
    return f"rc={r['rc']}"


def kfd(arm, model, kv, d):
    """KFD's System and TTM counts once the context exists (TTM at its peak), and the limit."""
    pts = [tuple(map(int, m.groups())) for m in map(MEM.match, mem.get(f"{arm}_{model}_{kv}_d{d}", "").splitlines()) if m]
    if not pts:
        return None
    peak = max(p[3] for p in pts)
    sysv = sorted(p[1] for p in pts if p[3] == peak and p[1] > 0)
    return sysv[len(sysv) // 2], peak, pts[0][2]


out = io.StringIO()
w = csv.writer(out, delimiter="\t", lineterminator="\n")
w.writerow(["model", "kv", "arm", "generates_at", "fails_at", "pass_measured_in", "tg64_at_pass", "fill_seconds",
            "failure", "seconds_to_failure"])
for arm, model, kv, ok, fail in cells:
    p = runs.get((arm, model, kv, ok))
    src = "this campaign" if p and p["res"] == "ok" else earlier.get((arm, model, kv, ok), "?") + " pass"
    w.writerow([model, kv, arm, ok, fail, src, p["rate"] if p and p["res"] == "ok" else "",
                p["sec"] if p and p["res"] == "ok" else "", ended(arm, model, kv, fail), runs[(arm, model, kv, fail)]["sec"]])
open(os.path.join(D, "ceilings.tsv"), "w").write(out.getvalue())

lines = ["KFD 'System mem used' once the context exists, every ROCm pass of this campaign (MiB).",
         "rest = System - TTM - context/1024, i.e. what System counts beyond the GTT buffers and a 1 KiB-per-token",
         "pinned input buffer (the flash-attention KQ mask, 512 rows x context x 2 bytes).",
         "", f"{'arm':6} {'model':22} {'kv':5} {'context':>8} {'System':>7} {'TTM':>6} {'rest':>6}  limit"]
for (arm, model, kv, d), r in sorted(runs.items(), key=lambda x: (x[0][1], x[0][2], x[0][0], x[0][3])):
    if arm == "vk" or r["res"] != "ok":
        continue
    k = kfd(arm, model, kv, d)
    if k:
        s, t, cap = k
        lines.append(f"{arm:6} {model:22} {kv:5} {d:8} {s:7} {t:6} {s - t - d // 1024:6}  {cap}")
open(os.path.join(D, "kfd-accounting.txt"), "w").write("\n".join(lines) + "\n")
print(f"{len(cells)} cells, {sum(1 for r in runs.values())} runs")
