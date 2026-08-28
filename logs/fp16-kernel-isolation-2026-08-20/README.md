Trying to separate the two things the gfx1010 override changes at once: the
device identity and the GEMM code objects. Three approaches, all blocked, and the
blocks say something.

The gfx1010 result (`../fp16-arch-2026-08-20/`) makes the fp16 defect vanish, but
it moves identity and library together. Isolating the library required getting
gfx1010-derived code objects onto a device still presenting as gfx1013.

**Hybrid library.** A copy of the native gfx1013 install with all 56 of its
gfx1013 Tensile files replaced by the system's gfx1010 files, renamed. The
substitution worked (the gfx1013-named `Kernels.so` verifiably contains gfx1010
code) and the result aborts with `CUBLAS_STATUS_INTERNAL_ERROR` for both compute
types. So gfx1010 code objects do not execute on gfx1013 hardware, which is
exactly why the native build exists, and is also why the system library fails
here while working under the override.

**Tensile disabled.** Pointing `ROCBLAS_TENSILE_LIBPATH` at an empty directory to
force whatever source-level fallback rocBLAS has: it refuses to start, reporting
it cannot read the library files. There is no non-Tensile path to fall back to.

**What this means.** The two variables are coupled by the hardware rather than by
the experiment: a device presenting as gfx1013 requires gfx1013 code objects, so
"gfx1013 identity with gfx1010 kernels" is not a configuration that exists. The
override result therefore cannot be decomposed by substitution, and the way to
vary the gfx1013 kernels while keeping the identity is to build them differently.

A rebuild of the same rocBLAS source is the next step, which also answers a
question worth answering on its own: whether the defect is systematic in a
gfx1013 Tensile build or an artifact of this particular one. The original install
is preserved as `build-original-2026-08` with the working path kept as a symlink,
verified by md5 and by the gate returning 8.9442 through it afterwards.

That last sentence did not survive the next day, and this note should not be read
without the sequel. The rebuild it proposes was started, `rmake.py` wrote into
`./build`, followed the symlink, and cleared the tree the arrangement was meant to
protect. The library was reconstructed from two partial copies, one of them the
hybrid built for the experiment above, and the recovery is in
[`../rocblas-recovery-2026-08-20/`](../rocblas-recovery-2026-08-20/). A symlink is
not a backup, and a build system that writes to a fixed relative path will follow
one.

Produced by hand rather than by a harness: three ad hoc configurations run directly, each recorded above with the command shape that produced it.
