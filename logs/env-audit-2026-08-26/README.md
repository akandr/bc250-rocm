# Every environment variable, checked against the library that would read it, 2026-08-26

Every environment variable this work relies on was resolved on the board against the library that
would have to read it: each name was looked for in the four binaries that could consult it, the HIP
runtime, the HSA runtime, rocBLAS and `libggml-hip.so`, and counts as read only if it appears in
one of them. `env_vars.txt` records the four paths and versions inspected and the result per
variable; it does not record the command, so the exact invocation is not recoverable from it.

The check is older than this capture. It exists because two variables carried in command
lines here for weeks turned out to do nothing, `GGML_CUDA_FORCE_MMQ` and `GGML_CUDA_FORCE_CUBLAS`
being compile-time options, not environment ones, and one of them put a wrong label on a
defect report. What was missing until now is a run of it: the front page describes the check, and
no output of it was kept anywhere, so the check was itself a claim.

Twelve of the fourteen are read by the library that would have to read them:

| variable | read by |
|---|---|
| `AMD_LOG_LEVEL` | hip |
| `GGML_CUDA_CUBLAS_COMPUTE_TYPE`, `GGML_CUDA_DISABLE_GRAPHS`, `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | ggml |
| `GPU_FORCE_BLIT_COPY_SIZE`, `GPU_STAGING_BUFFER_SIZE` | hip |
| `HSA_ENABLE_INTERRUPT`, `HSA_ENABLE_SDMA`, `HSA_OVERRIDE_GFX_VERSION`, `HSA_XNACK` | hsa |
| `ROCBLAS_LAYER`, `ROCBLAS_TENSILE_LIBPATH` | rocblas |

`GGML_CUDA_FORCE_CUBLAS` and `GGML_CUDA_FORCE_MMQ` report not read by any of them, which is the
expected result and the reason the audit was written.

Two of these matter beyond bookkeeping. `GGML_CUDA_CUBLAS_COMPUTE_TYPE` is the fp16 workaround the
front page recommends, and `HSA_ENABLE_SDMA` is the one it recommends against needing any more;
both are read, so both do what the text says. `HSA_OVERRIDE_GFX_VERSION` being read is what makes
the override trap possible in the first place
([`../override-trap-2026-08-26/`](../override-trap-2026-08-26/)).

The library paths are this board's. On another setup they need adjusting, and the audit refuses to
run instead of reporting fourteen findings that are really an empty search.
