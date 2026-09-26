#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Reweight the MoE per-op replay by how often each shape actually runs.

logs/moe-decode-2026-09-20/ replayed one decode graph op by op and summed the result, concluding that
ROCm was at or ahead of Vulkan on the shapes the graph contains. Two things are wrong with that sum.

It weights every distinct shape equally. A shape that runs 100 times a token counts the same as the
output head, which runs once.

Correcting that by weighting each shape by how often it runs turns out not to be available across
backends, and this script shows why instead of doing it anyway: ROCm and Vulkan do not issue the same
matmul dispatches for this graph. ROCm fuses pairs that Vulkan runs separately, so its trace shows
437.7 matvec dispatches a token against Vulkan's 511, and the iq2_xxs expert gate/up runs 39.4 times at
41.4 us on ROCm where Vulkan runs that shape 80 times at 19.3 us -- and the ROCm kernel does more, since
has_fusion folds in the GLU that Vulkan dispatches separately. Weighting one backend's isolated
per-shape time by the other's dispatch count mixes two different graphs. An earlier version of this
script did exactly that and reported a spurious 6.5 percent.

What does not need weighting, and is what this reports: the two backends replayed with the same
instrument, shape by shape.

And one row of the Vulkan replay is an artifact. test-backend-ops measures MUL_MAT
shared_expert_gate-0, a 1x1 output from a 2048-long dot product, at 131.71 us on Vulkan. In the real
graph ggml-vulkan dispatches that shape as MUL_MAT_VEC f32 m=1 n=1 k=2048 and it costs 4.26 us. The
replay row is 31 times its real cost, and it is larger than the entire margin the published sum rests
on.

Run from the repo root. Reads only files already in logs/, writes nothing.
"""

import re
import sys
from pathlib import Path

REPLAY_HIP = "logs/moe-decode-2026-09-20/ops-moe-n1-hip.log"
REPLAY_VK = "logs/moe-decode-2026-09-20/ops-moe-n1-vk.log"
PERF_VK = "logs/floor-vs-vulkan-2026-09-23/vk-perf-last-block.txt"

# The one shape whose Vulkan replay reading is an artifact, and its real cost from the perf logger.
ARTIFACT = ("MUL_MAT", "shared_expert_gate-0")
ARTIFACT_REAL_US = 4.258

# ROCm kernel time per token, from the rocprof trace in logs/moe-decode-2026-09-20/:
# 775.832 ms over 64 tokens.
ROCM_KERNEL_MS = 775.832 / 64
ROCM_TOKEN_MS = 13.93
VULKAN_TOKEN_MS = 11.54


def parse_replay(path):
    """test-backend-ops perf output -> {(op, name, ne): (us_per_run, k)}"""
    out = {}
    pending = None
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = re.match(r"\s{2}([A-Z_0-9]+)\((.*?)\):", line)
        if m:
            pending = (m.group(1), m.group(2))
        run = re.search(r"(\d+) runs -\s+([\d.]+) us/run", line)
        if run and pending:
            attrs = pending[1]
            name = re.search(r"name=([^,]*)", attrs)
            ne = re.search(r"ne=\[([^\]]*)\]", attrs)
            src = re.search(r"sources=(\w+)\[([^\]]*)\]", attrs)
            dims = tuple(int(x) for x in ne.group(1).split(",")) if ne else ()
            k = int(src.group(2).split(",")[0]) if src else 0
            qt = src.group(1) if src else ""
            out[(pending[0], name.group(1) if name else "", dims)] = (float(run.group(2)), k, qt)
            pending = None
    return out


def parse_perf(path):
    """ggml-vulkan perf logger -> {(quant, m, n, k): (count, us_each)} for the matmul family.

    Keyed on the weight type as well as the shape: two different tensors in this graph share
    (m, n, k) and differ only in quantisation, so a shape-only key silently merges them.
    """
    out = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        m = re.search(r"MUL_MAT(?:_ID)?_VEC (\S+) m=(\d+) n=(\d+) k=(\d+).*?: (\d+) x ([\d.]+) us", line)
        if m:
            key = (m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)))
            assert key not in out, f"duplicate perf-logger key {key}"
            out[key] = (int(m.group(5)), float(m.group(6)))
    return out


def main():
    hip = parse_replay(REPLAY_HIP)
    vk = parse_replay(REPLAY_VK)

    shared = sorted(set(hip) & set(vk))
    h_sum = sum(hip[k][0] for k in shared)
    v_sum = sum(vk[k][0] for k in shared)

    art = [k for k in shared if (k[0], k[1]) == ARTIFACT]
    v_fixed = v_sum - sum(vk[k][0] for k in art) + ARTIFACT_REAL_US * len(art)

    print(f"{len(shared)} shapes replayed on both backends\n")
    print("unweighted, one reading per distinct shape")
    print(f"  as published                 ROCm {h_sum:7.1f} us   Vulkan {v_sum:7.1f} us   ROCm/Vulkan {h_sum/v_sum:.3f}")
    print(f"  artifact row corrected       ROCm {h_sum:7.1f} us   Vulkan {v_fixed:7.1f} us   ROCm/Vulkan {h_sum/v_fixed:.3f}")

    faster = sum(1 for k in shared if hip[k][0] < vk[k][0])
    ratios = sorted(hip[k][0] / vk[k][0] for k in shared)
    print(f"\n  ROCm faster on {faster} of {len(shared)} shapes, slower on {len(shared)-faster}")
    print(f"  median per-shape ROCm/Vulkan {ratios[len(ratios)//2]:.3f}")

    # split by what the weights are. This needs no occurrence counts: it is the same instrument on
    # both backends, shape by shape.
    import statistics

    mm = [k for k in shared if k[0].startswith("MUL_MAT")]
    quantised, floats = [], []
    for k in mm:
        ht = hip[k][0]
        vt = ARTIFACT_REAL_US if (k[0], k[1]) == ARTIFACT else vk[k][0]
        (floats if hip[k][2] == "f32" else quantised).append((k[1], hip[k][2], ht, vt, ht / vt))

    for label, group, worse in (("quantised-weight", quantised, True), ("f32-weight", floats, False)):
        print(f"\n{label} matmuls, replay against replay, no weighting")
        for n, t, a, b, r in sorted(group, key=lambda x: -x[4]):
            print(f"  {n[:24]:24s} {t:8s} ROCm {a:8.2f} us   Vulkan {b:8.2f} us   {r:6.3f}")
        n_bad = sum(1 for x in group if (x[4] > 1) == worse)
        print(f"  ROCm {'slower' if worse else 'faster'} on {n_bad} of {len(group)}, "
              f"median ratio {statistics.median([x[4] for x in group]):.3f}")

    print("\nwhy this is not weighted by occurrence")
    print("  ROCm issues 437.7 matvec dispatches a token (rocprof trace), Vulkan 511 (perf logger).")
    print("  ROCm fuses pairs Vulkan runs separately: the iq2_xxs expert gate/up is 39.4 dispatches a")
    print("  token at 41.4 us on ROCm against 80 at 19.3 us on Vulkan, and the ROCm kernel is doing more")
    print("  than the Vulkan one (has_fusion folds the GLU in, which Vulkan dispatches 80 times at")
    print("  2.96 us). Per-shape counts are not shared between the backends and the dispatches are not")
    print("  even like for like, so a cross-backend weighted total is not defined for this graph.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
