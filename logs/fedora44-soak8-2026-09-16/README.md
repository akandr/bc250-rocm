# Eight-hour soak on the default Fedora 44 configuration, 2026-09-16

[`scripts/soak_f44.sh`](../../scripts/soak_f44.sh), 11:19 to 19:22 on the default boot: Fedora 44, kernel
7.1.8 with the bc250 amdgpu module, SELinux enforcing, native gfx1013 rocBLAS 7.1.1 and the corrected comgr
from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz, `ollama` stopped (the three-hour soak the day before
lost three gates to it). Each round: qwen2.5-1.5B pp2048, the 1.5B gate, the qwen3-8B gate at the default
fp16 compute type, the MUL_MAT allocation-churn sweep, and every third round the PyTorch training loop on
the gfx1013 build.

## Result

**41 rounds, everything bit-identical, no faults.**

| | |
|---|---|
| 1.5B gate | 8.9442 in all 41 rounds |
| qwen3-8B fp16 gate | 9.1117 in all 41 rounds |
| allocation-churn sweep | 41 of 41 exit 0 |
| PyTorch training, every third round | 13 runs, all `last loss 0.00048`, `maxlossdiff 1.799e-05` |
| prefill pp2048 | 633.83 to 635.99 t/s, a spread of 0.34 percent |
| GPU edge temperature | 66 to 75 C |
| kernel fault lines (page fault, preemption, runlist flush, GPU reset) | 0 |
| governor throttle events | 1, logged at 11:00 during the thermal test that preceded the soak |

The 8B gate matters here beyond correctness: it runs the default fp16 compute type, so every round exercises
the rocBLAS fp16 GEMM path that was returning zeros on Fedora 43 before the toolchain defect was found. Forty-one
identical values is the strongest evidence so far that the Fedora 44 toolchain does not carry it.

The kernel log of this boot holds 4374 `SVM mapping failed` lines, all timestamped 10:51, from the
deep-context experiments that ran before the soak started
([`../fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/)). None are from the soak.

Prefill reads 634 t/s here against 767 in the three-hour soak of the previous night. That is not a
regression: the earlier soak ran while the governor was oscillating up to 2000 MHz
([`../fedora44-soak-2026-09-16/`](../fedora44-soak-2026-09-16/)), and this one is at the pinned 1500 MHz
policy the rest of the current measurements use.

## What this does and does not show

It shows the configuration is stable for a working day under mixed load: inference, a GEMM-heavy churn
sweep, and training, with bit-identical numerics throughout. It does not show that the rare page fault
described in the README is gone; that appeared roughly once in 200 rounds on Fedora 43, and 41 rounds is too
few to say anything about it. Nothing here ran longer than eight hours.
