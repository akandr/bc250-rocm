# Where each patch comes from

Every patch file carries a mail-style header; this table is the short form of what the headers say, so
that nobody has to take a `From:` line on trust. "This repository" means written here, on the BC-250, by
Artur Andrzejczak with Anthropic's Claude; each header names the model it was written with, as `Co-authored-by:`
in the GitHub-project patches and as the kernel's `Assisted-by: LLM` in the kernel ones. The
`Signed-off-by` line on any patch here is mine.

| patch | the change is | credit in the header |
|---|---|---|
| `llamacpp/0001-hip-integrated-false.patch` | this repository's one-line revert, found by a bisect; llama.cpp master has since done the same on every platform | upstream's own non-HIP precedent (issue #15034) |
| `llamacpp/0002-kqv-f32-precision.patch` | this repository, one line | |
| `llamacpp/0003-gfx1013-rdna1-macro.patch` | this repository, one line; upstream master still lacks it | |
| `llamacpp/0004-rdna1-fattn-tile-no-spill.patch` | this repository: the RDNA1 tile table, its rows from compile-time sweeps, the dispatch check, the one-token kernel rule | |
| `llamacpp/0005-rdna1-mmvq-table-and-sums.patch` | this repository: the table entry, the `v_sad_u8` sums, the float-activation kernel | the kernel's layout follows ggml-vulkan's `mul_mat_vec_*.comp` shaders |
| `llamacpp/0006-concat-transposed-source.patch` | this repository | |
| `llamacpp/0007-rdna1-gdn-lanes.patch` | this repository: the template parameter and the RDNA1 choice | the 8-lane geometry is ggml-vulkan's for the same op |
| `llamacpp/0008-rdna1-pkf16-prefill-gemm.patch` | this repository: the kernel, its decoders, tile shape and predicate | the tiling is the ordinary shared-memory GEMM shape; the arithmetic-ceiling measurement behind it is in `logs/alu-rates-2026-09-19/` |
| `llamacpp/0009-rdna1-iq-matvec-shmem-codebook.patch` | this repository: the staging, the expert-path exclusion, the 16-bit index loads | staging the codebook in shared memory is what ggml-vulkan's IQ matvec shaders do in `init_iq_shmem` |
| `llamacpp/0010-rdna1-pkf16-halve-column-tile.patch` | this repository: the tile sweep and the shape it settles on | |
| `llamacpp/0011-rdna1-pkf16-q8_0.patch` | this repository: the retest that reverses the earlier exclusion | |
| `llamacpp/0012-rdna1-pkf16-expert-path.patch` | this repository: the expert-path kernel mode, its column tile and the dispatch hook | the compact expert ordering is llama.cpp's own `mm_ids_helper`, which MMQ and MMF also call |
| `llamacpp/0013-rdna1-pkf16-admit-128-token-batches.patch` | this repository: the threshold sweep at the new tile | |
| `hip-graph-capture-null-stream.patch` | this repository; `ROCm/clr` `develop` lacks the check as of 2026-09-19 | |
| `rocclr-hostqueue-thread-release-null-vdev.patch` | **not ours**: a backport of ROCm/clr commit 7d979ab5a0 (Jaydeep, 2026-05-25); the same guard was written here from the crash before that commit was found, and replaced by it | `From:` is the upstream author; the porter note says what was done here |
| `rocr-guard-queue-scratch-release.patch` | **not ours**: a port of rocm-systems PR #2850 (Luna Nova, merged 2026-02-07) to the 7.1.1 source layout, path prefix only | `From:` is the upstream author |
| `amdgpu-flush-pasid-mmio.patch` | **the line is anrp's** (ROCm/ROCm#6313, 2026-05-30), confirmed by ahorek; this repository adds only the correctness A/B | a bare diff with a credit note, no `From:` and no sign-off: nothing to claim |
| `amdgpu-fence-fallback-2ms.patch` | this repository, a probe that did not help, kept for the record | |

The kernel-module changes that the recipe applies through scripts, not patch files
(`scripts/apply_runlist_flush.py`, `scripts/apply_svmflush_generic.py`) are ported from
GabriWar/bc250-rocm-working and extended here, as the front page says, and the 40-CU unlock is
duggasco/bc250-40cu-unlock's.
