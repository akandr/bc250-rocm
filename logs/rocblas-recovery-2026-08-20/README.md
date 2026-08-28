The native gfx1013 rocBLAS was destroyed and then reconstructed. Recorded because
the recovery route is worth knowing and the mistake is worth not repeating.

**What happened.** Preparing to rebuild rocBLAS to test whether the fp16 defect
is systematic in a gfx1013 Tensile build, the existing `build` directory was
renamed to `build-original-2026-08` and a symlink `build -> build-original-2026-08`
put in its place so the canonical `LD_LIBRARY_PATH` would keep resolving. The
build was then started with `rmake.py`, which writes into `./build`, followed the
symlink, and cleared the tree it was supposed to be preserving. The library that
every measurement in this repository depends on was gone within seconds.

**Why it was recoverable.** Two independent partial copies survived:

- `librocblas.so.4.4` itself, in a hybrid library made an hour earlier for a
  different experiment, md5 `2f3fa0222581` matching the original exactly.
- All 56 gfx1013 Tensile kernel files, in the PyTorch wheel at
  `torchvenv/.../torch/lib/rocblas/library/`, where a July session had grafted
  them.

Recombining the two reproduces the original behaviour exactly: the 1.5B gate
returns 8.9442 and the 8B f32 gate returns 9.0975. Both are this repository's
standing reference values and both are reproduced in dozens of captures
elsewhere, so the reconstruction can be checked against them. The reconstruction
is installed at the canonical path and `restored-state.txt` records its size, md5
and file count.

This paragraph also gave the 8B f16 arm as 15.4864, "inside the defect's usual
range". That reading exists in no captured file anywhere, here or elsewhere, so
it has been dropped rather than kept as a bare assertion. It is plausible, since
the range on record for that arm is 14.7398 to 21.4429
([`../fp16-recheck-2026-08-25/`](../fp16-recheck-2026-08-25/)), but a defect whose
value moves by half is not evidence of an exact reconstruction in any case. The
two f32 gates are, and they are what the claim now rests on.

**Two lessons.** A symlink is not a backup, and a build system that writes to a
fixed relative path will follow one. And the accident surfaced something else:
the only reason the Tensile kernels survived is that a wheel believed to be
pristine had them grafted in, which is itself a correction to a figure this
repository had published (see `../torch-probe-2026-08-19/`).
