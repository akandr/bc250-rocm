#!/usr/bin/env python3
# Repair the half-precision soft-float helpers statically linked into ROCm libraries on Fedora 43.
#
# Root cause (logs/fp16-root-cause-2026-09-15/README.md): Fedora 43's rocm-compilersupport
# 19-14.rocm6.4.2 installs a libclang_rt.builtins-x86_64.a whose __extendhfsf2 and __truncsfhf2 were
# built without COMPILER_RT_HAS_FLOAT16, so they pass the 16-bit value in an integer register
# (%edi in, %eax out). The ROCm clang 19 that links against it follows the x86-64 psABI for _Float16
# and passes and expects the value in %xmm0. rocBLAS links that archive (--rtlib=compiler-rt, and
# hip::host/hip::device link it too), so every half<->float conversion in the library converts
# whatever was left in %edi. In rocBLAS that makes `prob.k && *prob.alpha` see alpha as 0 and turns
# fp16 GEMMs into K=0 problems that return zeros, nondeterministically.
#
# This replaces each broken local helper, in a COPY of the library, with the F16C instruction that
# performs the conversion under the psABI convention the callers actually use:
#   __extendhfsf2: endbr64; vcvtph2ps %xmm0,%xmm0; ret
#   __truncsfhf2:  endbr64; vcvtps2ph $4,%xmm0,%xmm0; ret     (4 = round per MXCSR, nearest-even)
# Requires a CPU with F16C (the BC-250's Zen 2 has it). A helper is patched only if its first
# instructions match the integer-register variant; libraries built with a correct archive (the
# PyTorch wheel's bundled rocBLAS, for example) are left alone.
#
# The proper fix is a correct builtins archive or building with -mf16c; this is a stopgap that
# needs no rebuild.
#
# Usage: fix_half_helpers.py <input.so> <output.so>
import subprocess, sys

EXT_BAD = bytes.fromhex("f30f1efa89f9")          # endbr64; mov %edi,%ecx
TRUNC_BAD = bytes.fromhex("f30f1efa660f7ec2")    # endbr64; movd %xmm0,%edx (then returns in %eax)
EXT_NEW = bytes.fromhex("f30f1efac4e27913c0c3")
TRUNC_NEW = bytes.fromhex("f30f1efac4e3791dc004c3")

def symbols(path):
    out = subprocess.run(["readelf", "-Ws", path], capture_output=True, text=True, check=True).stdout
    found = {}
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 8 and f[3] == "FUNC" and f[7] in ("__extendhfsf2", "__truncsfhf2") and f[6] != "UND":
            found[f[7]] = (int(f[1], 16), int(f[2]))
    return found

def vaddr_to_offset(path, vaddr):
    out = subprocess.run(["readelf", "-lW", path], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        f = line.split()
        if f and f[0] == "LOAD":
            off, va, filesz = int(f[1], 16), int(f[2], 16), int(f[4], 16)
            if va <= vaddr < va + filesz:
                return vaddr - va + off
    raise SystemExit(f"address 0x{vaddr:x} not in any LOAD segment")

def main():
    src, dst = sys.argv[1], sys.argv[2]
    if src == dst:
        raise SystemExit("refusing to patch in place; give a different output path")
    data = bytearray(open(src, "rb").read())
    syms = symbols(src)
    if not syms:
        print("no local half helpers: nothing to do")
    patched = 0
    for name, (vaddr, size) in syms.items():
        off = vaddr_to_offset(src, vaddr)
        head = bytes(data[off:off + 8])
        bad, new = (EXT_BAD, EXT_NEW) if name == "__extendhfsf2" else (TRUNC_BAD, TRUNC_NEW)
        if not head.startswith(bad):
            print(f"{name} at 0x{vaddr:x}: does not match the integer-register variant ({head.hex()}), left alone")
            continue
        if size < len(new):
            raise SystemExit(f"{name} is only {size} bytes, cannot fit the replacement")
        data[off:off + len(new)] = new
        patched += 1
        print(f"{name} at 0x{vaddr:x} (file offset 0x{off:x}, {size} bytes): replaced with F16C stub")
    open(dst, "wb").write(data)
    print(f"wrote {dst} with {patched} helper(s) patched")

if __name__ == "__main__":
    main()
