# The fp16 defect across presented architectures, 2026-08-22

Measured on the dual-architecture llama.cpp build that carries both gfx1010 and
gfx1013 code.

The method line here used to credit `scripts/fp16_arch_and_queue_probes.sh`. It
did not produce this. That script runs the ordinary gfx1013-only build and logs
in a different format, and its run of the same experiment an hour earlier is in
[`../decode-queue-2026-08-22/`](../decode-queue-2026-08-22/), where every override
arm reads FAIL because a gfx1013-only binary has no kernels to load under an
override. The harness that produced the data below was not shipped, so this
directory can be checked but not re-run. Noted 25 August rather than guessed at.

## Result

Three rounds, fp16 compute type, 8B at context 2048 over two chunks:

| round | gfx1013, native rocBLAS | gfx1010 override, system rocBLAS |
|---|---|---|
| 1 | 11.5415 | 9.1117 |
| 2 | 18.2958 | 9.1117 |
| 3 | 17.3015 | 9.1117 |

The gfx1010 arm is not merely correct, it is **identical to four decimals three
times over**. The gfx1013 arm is wrong and differently wrong each time. So the
non-determinism, which is the strangest part of this defect, belongs to the
gfx1013 path specifically and does not follow the model, the graph, the compute
type or the calling code, all of which are the same in both arms.

That sharpens the earlier finding, which had two samples and established only
that the defect disappears.

## What is still confounded

Presenting the device as gfx1010 changes two things at once: the ISA the kernels
target, and which rocBLAS serves them. They cannot be separated on this board,
because the native build carries only gfx1013 code objects and the system library
carries no gfx1013 at all, so no combination isolates one from the other.

Separating them needs a rocBLAS built with both architectures, which is a
multi-hour rebuild and one previous attempt at rebuilding it failed. Until then
"gfx1010 is clean" means the pair, not the ISA alone.

A third arm, presenting as gfx1030, was attempted to ask whether gfx1010 is
special or whether anything but gfx1013 is clean. It could not run: the build
carries gfx1010 and gfx1013 only, so under a gfx1030 override no kernels match
and llama.cpp aborts with `invalid device function`. Answering that needs another
build.

One thing in the log is unaccounted for. It ends with a section headed "f32
control on the same build" whose result line is empty, so a control was run and
returned nothing, and this page never mentioned it. What that control would have
shown, whether the f32 arm on the dual-architecture build returns the usual
9.0975, is therefore not established here. It is established on the ordinary
build in several other directories, so nothing above depends on it, but the empty
line should not have gone unremarked for three days.
