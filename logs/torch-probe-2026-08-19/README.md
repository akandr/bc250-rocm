The eleven-operation PyTorch probe, run on both builds, because the figures it
backs appeared in no shipped log.

Same probe in both arms, identical to `patches/torch_opprobe.py` as shipped.

| build | result |
|---|---|
| built from source for gfx1013 | **11 of 11**, `arch list: ['gfx1013']` |
| stock ROCm wheel, pristine | **3 of 11** |

The three that pass on the stock wheel are the host-to-device copy, fp32
`matmul` and fp32 `addmm`. Everything dispatched to one of torch's own compiled
kernels fails with `invalid device function`, the expected result of shipping no
gfx1013 code objects: its `arch list` is gfx900 through gfx1201 with no 1013.

**CORRECTION 2026-08-20, read this before the numbers below.** The wheel used
here is not pristine, so the 3-of-11 figure is not a stock-wheel result. The
check that led to calling it pristine compared the bundled `librocblas.so`, which
is indeed stock: different md5 from the native build, and no gfx1013 in its
strings. It did not look at the external Tensile directory beside it, and that
directory contains 56 real gfx1013 kernel files with genuine gfx1013 code,
written six minutes after the stock files during a July session. A stock ROCm 6.4
wheel ships gfx1013 only as symlinks to gfx1010, as the system library
does; nobody builds gfx1013 Tensile kernels, so rocBLAS PR #8838 is
open. This wheel therefore had the native kernels grafted into it, and that is very
likely why its fp32 `matmul` and `addmm` succeed.

This is the second time in this project that a wheel believed to be stock had
been modified, and the second time a partial verification was taken for a
complete one: the library file was checked and the kernel files beside it were not.

The stock figure, from a fresh install, is in [`../torch-pristine-2026-08-20/`](../torch-pristine-2026-08-20/).

It also corrects how the wheel can be modified. The write-up also gave
4 of 11 for the wheel "with the native rocBLAS grafted in". That figure is not
re-verified here, and it cannot be reached by the direct route: `torch/lib` is built
with `RPATH $ORIGIN`, so the bundled `librocblas.so` takes precedence over anything on
`LD_LIBRARY_PATH`. Running the wheel with the native library on the path loads
the bundled one anyway, confirmed by reading `/proc/self/maps` during a matmul,
and still scores 3 of 11. Grafting means replacing the file inside the wheel,
which is what the earlier session must have done and what this run deliberately
did not do.

This was then resolved. Tracing the wheel with `ROCBLAS_LAYER=1` shows
`rocblas_sgemm` being called for a 512-square fp32 matmul and succeeding, with
no error and a correct result; the failure that follows in the same script is
`torch.isfinite`, one of torch's own kernels. The fp32 library path is therefore
serviced despite the wheel carrying no gfx1013 code objects, which is
why the score is three, not one.

The system library behaves differently. Its
gfx1013 entries are real files in name only: 56 of them, every one a symlink to
its gfx1010 counterpart (`Kernels.so-000-gfx1013.hsaco -> ...gfx1010.hsaco`),
and a 512-square SGEMM against it fails with `rocblas_status_internal_error`.
Same absence of gfx1013 code, different outcome; keep both in mind before
generalising from either.

Run by invoking `patches/torch_opprobe.py` directly under each interpreter, with `HSA_ENABLE_SDMA=0` and the native rocBLAS on `LD_LIBRARY_PATH` for the source build. No harness script; the probe is the experiment.
