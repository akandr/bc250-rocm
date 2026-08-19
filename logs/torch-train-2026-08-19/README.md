The 50-step training loop from `patches/pytorch/torch_train.py`, run twice on the
native gfx1013 build against its CPU reference. Kept because the figure it backs
was quoted with no shipped log.

Both runs are identical:

| quantity | value |
|---|---|
| first loss, CPU and GPU | 2.30348 |
| last loss, CPU and GPU | 0.00048 |
| max loss difference across all steps | 1.799e-05 |
| max parameter difference after 50 steps | 9.312e-03 |
| script's own verdict | `agrees with cpu: False` |
| wall time | 20.3 s CPU, 0.26 s GPU |

The quoted per-step agreement of 1.8e-5 is confirmed. So is the identical final
loss, which is what the eight-hour soak's fourteen training rounds were checking.

The last two rows are worth reading together, and the write-up had been quoting
only the favourable one. The script compares parameters as well as losses, with a
1e-3 threshold, and after fifty steps the accumulated parameter difference is
9.3e-3, so its built-in check reports disagreement. That is what fifty steps of
accumulated floating-point difference between two backends looks like rather than
a defect, and the losses staying within 1.8e-5 of each other throughout is the
evidence for that reading. But the failing verdict should be stated rather than
left out.

Determinism is separately clear: the two runs agree to every digit printed,
including the parameter difference.

Run by invoking `patches/pytorch/torch_train.py` directly under the source build, twice, with `HSA_ENABLE_SDMA=0` and the native rocBLAS on `LD_LIBRARY_PATH`.
