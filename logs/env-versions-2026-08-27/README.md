# The environment line, checked against the machine, 2026-08-27

`versions.txt` is one shell invocation on the board, recorded because the front page states an
environment and one item of it was backed by nothing.

| front page says | the board reports |
|---|---|
| Fedora 43 | `Fedora release 43 (Forty Three)` |
| kernel 7.1.8 | `7.1.8-100.fc43.x86_64` |
| LLVM/clang 19 | `clang version 19.0.0git` from `/usr/lib64/rocm/llvm/bin/clang++` |
| Mesa 25.3 | `mesa-vulkan-drivers-25.3.4-7.fc43` |
| the oberon governor | `active` |

The clang version is why this exists. Every other item here appears in dozens of captures already,
and that one appeared in none: the logs record the compiler's path and never its version, while the
`LLVM 21.1.8` they do record is Mesa's radeonsi reporting itself rather than the HIP compiler. Two
different LLVMs on one machine, and the front page named one of them without evidence.

ROCm's own version is not in this capture: `rpm -q rocm-core` reports the package absent, since the
ROCm here is not installed from that package. The 6.4.2 on the front page rests on the thirty-three
captures that carry it, including the runtime's own strings, rather than on this file.
