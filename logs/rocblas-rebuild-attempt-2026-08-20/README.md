An attempt to rebuild the native gfx1013 rocBLAS, to test whether the zeroed fp16
GEMM is systematic in a gfx1013 Tensile build or an artifact of the one build
everything has been measured against. **Not completed**, and stopped
deliberately rather than abandoned; this records how far it got so it can be
resumed.

**Why it matters.** Driving the board as gfx1010 makes the defect vanish, but
that changes device identity and library together and the separating cell cannot
be run (`../fp16-kernel-isolation-2026-08-20/`). Varying the gfx1013 kernels
while keeping the identity means building them differently, which makes this the
decisive remaining test.

**Progress.** Four blockers cleared, in an isolated source copy that cannot reach
the working install:

1. `TENSILE_ROCM_ASSEMBLER_PATH` unset, which the repository's own build script
   documents and an abbreviated build environment omitted.
2. The virtualenv Tensile copy is re-fetched unpatched on every configure, so
   the gfx1013 edits have to be reapplied to it each time.
3. A syntax error in the local patched `Tensile/Common.py`
   (`'gfx1012'` had lost its value when the gfx1013 entry was inserted), which
   means that copy could not have built as it stood.
4. `AsmCaps` had no `(10, 1, 3)` entry, giving `KeyError: (10, 1, 3)`; added by
   duplicating the `(10, 1, 2)` block.

**Where it stops.** `Failed to verify all files in manifest`, which is the second
failure mode the build script names, and which needs the gfx1013 additions to the
Tensile and rocBLAS C++ enums (`AMDGPU.hpp`, `PlaceholderLibrary.hpp`,
`handle.hpp`, `handle.cpp`, `tensile_host.cpp`). Those were applied by hand during
the original build and lived only inside the build tree, so they are not
recoverable from the source on the board.

**The Tensile source on the board carries broken gfx1013 edits.** Looking at
`Tensile/Source/lib/include/Tensile/AMDGPU.hpp` in the local Tensile checkout,
the two gfx1013 additions are not valid C++:

    case AMDGPU::Processor::gfx1012:
        return "gfx1012", "gfx1013";        // comma operator: returns "gfx1013"

    else if(deviceString.find("gfx1012", "gfx1013") != std::string::npos)

The first is the comma operator, so every gfx1012 device reports itself as
"gfx1013" and the first string is discarded. The second passes a string literal
where `std::string::find` expects a position, which does not compile at all.
There is no `gfx1013` entry in the `Processor` enum, and
`PlaceholderLibrary.hpp` has no gfx1013 case whatever.

These are not being "fixed" here, because the first line may encode an intent
rather than a slip: mapping gfx1013 onto gfx1012's kernel logic while labelling
the generated files gfx1013 is a coherent hack if you only ever build for this
one target, and the library that resulted does work. Guessing at that intent and
writing a different patch would produce a different library, which is precisely
what this experiment is trying to hold constant. `tensile-broken-edits.txt`
records the lines verbatim.

**Worth stating plainly:** the native rocBLAS on this board is therefore not
currently reproducible from what the board holds. The build needs edits that
were made by hand, and the copy of them that survives does not compile, so the
ones that produced the working library are lost. The library itself is intact
and reconstructed (`../rocblas-recovery-2026-08-20/`), but rebuilding it from
scratch needs the documented manual edits redone.
