The eleven-operation PyTorch probe, run on both builds, because the figures it
backs appeared in no shipped log.

Same probe in both arms, identical to `patches/torch_opprobe.py` as shipped.

| build | result |
|---|---|
| built from source for gfx1013 | **11 of 11**, `arch list: ['gfx1013']` |
| stock ROCm wheel, pristine | **3 of 11** |

The three that pass on the stock wheel are the host-to-device copy, fp32
`matmul` and fp32 `addmm`. Everything dispatched to one of torch's own compiled
kernels fails with `invalid device function`, which is what shipping no gfx1013
code objects looks like: its `arch list` is gfx900 through gfx1201 with no 1013.

**This corrects a figure.** The write-up said the stock wheel manages 1 of 11.
It manages 3, stably across four runs. The wheel here is genuinely pristine: its
bundled rocBLAS has a different md5 from the native build and carries gfx1010
through gfx1035 with no gfx1013.

**And it corrects how the wheel can be modified at all.** The write-up also gave
4 of 11 for the wheel "with the native rocBLAS grafted in". That figure is not
re-verified here, and it cannot be reached the obvious way: `torch/lib` is built
with `RPATH $ORIGIN`, so the bundled `librocblas.so` wins over anything on
`LD_LIBRARY_PATH`. Running the wheel with the native library on the path loads
the bundled one anyway, confirmed by reading `/proc/self/maps` during a matmul,
and still scores 3 of 11. Grafting means replacing the file inside the wheel,
which is what the earlier session must have done and what this run deliberately
did not do.

One loose end worth noting rather than chasing here: fp32 `matmul` and `addmm`
succeed on a wheel whose rocBLAS carries no gfx1013 code, while fp16 `matmul`
fails. Why the fp32 library path survives is not established.

Run by invoking `patches/torch_opprobe.py` directly under each interpreter, with `HSA_ENABLE_SDMA=0` and the native rocBLAS on `LD_LIBRARY_PATH` for the source build. No harness script; the probe is the experiment.
