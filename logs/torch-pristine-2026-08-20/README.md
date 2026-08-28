What a genuinely stock PyTorch ROCm wheel does on this board, measured on a fresh
install rather than on the venv that had been sitting on the machine.

**It aborts.** `torch 2.9.1+rocm6.4`, installed clean into a new virtualenv, dies
at the first rocBLAS-dispatched operation:

    rocBLAS error: Cannot read .../rocblas/library/TensileLibrary.dat:
    No such file or directory for GPU arch : gfx1013

The wheel ships Tensile libraries for gfx908, gfx90a, gfx942, gfx1030, gfx1100,
gfx1101, gfx1102, gfx1200 and gfx1201, and nothing in the gfx101x family at all,
not even the symlinks-to-gfx1010 that the Fedora system rocBLAS carries. The
process aborts rather than raising, so the probe produces no score.

**This corrects two figures, one of which was this repository's own correction.**
The write-up long said the stock wheel manages 1 of 11 operations. That was
revised to 3 of 11 on 2026-08-19 after re-running the probe. Both numbers came
from wheels that were not stock: the venv used for the revision has 56 real
gfx1013 Tensile kernel files grafted into it, against zero in a fresh install.
The correct statement is neither number. A stock wheel cannot dispatch a single
library operation on this board, which is a simpler and stronger reason to build
from source than any partial score.

The revision was wrong in a way worth naming: the wheel's `librocblas.so` was
checked and is genuinely stock, and that was taken as establishing the wheel was
stock. The kernels rocBLAS actually loads live in a directory beside it, and were
not checked. Verifying the library file is not verifying the library.

**Grafting Tensile kernels alone is nearly useless, which was also unclear
before.** Copying the native build's 56 gfx1013 Tensile files into the fresh
wheel stops the abort but yields **1 of 11**: only the host-to-device copy
survives, and every library call fails with `HIPBLAS_STATUS_INTERNAL_ERROR` or
`invalid device function`. So the old "1 of 11" figure describes this
configuration accurately, and the practical conclusion is unchanged and
strengthened: on this board PyTorch has to be built from source, where it scores
11 of 11.

**One discrepancy is unresolved and is left visible rather than tidied.** The
older venv on this machine scores 3 of 11 with what appears to be the same
configuration: same torch version, byte-identical `librocblas.so`, byte-identical
56 gfx1013 Tensile files, identical file listings across the whole of
`torch/lib`, and `lib64` symlinked to `lib` in both. Both results are stable
across alternated repeats, 3 and 1 respectively, and installing the missing numpy
into the fresh venv does not change it. The older venv carries a large number of
additional packages from a source build, one of which presumably matters. What
makes the difference was not identified, and the numbers are reported as measured
rather than reconciled.

`pristine-probe.txt` holds the abort and the file counts on both venvs.

Produced by installing torch into a fresh virtualenv and running `patches/torch_opprobe.py` under it directly, with `TMPDIR` pointed into the home directory because /tmp is a 6 GiB tmpfs that pip overflows.
