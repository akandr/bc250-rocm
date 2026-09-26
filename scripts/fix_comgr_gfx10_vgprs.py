#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Correct the gfx10 total-VGPR count in a COPY of ROCm 7.0/7.1 libamd_comgr.
#
# ROCm 7.x HIP reads each device's VGPR budget from comgr's ISA metadata table
# (comgr-isa-metadata.def). Through rocm-7.1.1 every gfx10 row says 256 total VGPRs; ROCm 6.4.2
# hard-coded 1024, and ROCm/llvm-project commit 4f5ae331f659 ("[Comgr] Correct total VGPR counts for
# gfx10 devices", first in rocm-7.2.0) changes the same 13 rows to 1024. With 256,
# hipOccupancyMaxActiveBlocksPerMultiprocessor returns 0 for large kernels and llama.cpp's flash
# attention stops on GGML_ASSERT(max_blocks_per_sm > 0).
#
# The compiled row ends with the integers 65536 32 4 40 1024 106 800 106 8 256 256 (uint32, in .def
# order); the row's triple and name pointers are the 16 bytes before the next row's integers. This
# finds every row whose integers match that sequence, checks that its name is one of the 13 rows the
# upstream commit changed, and sets the total (the tenth integer) to 1024.
#
# Usage: fix_comgr_gfx10_vgprs.py <libamd_comgr.so.N> <output>
import struct, subprocess, sys

EXPECTED = {"gfx1010", "gfx1011", "gfx1012", "gfx1013", "gfx1030", "gfx1031", "gfx1032", "gfx1033",
            "gfx1034", "gfx1035", "gfx1036", "gfx10-1-generic", "gfx10-3-generic"}
ROW = 72

src, dst = sys.argv[1], sys.argv[2]
if src == dst:
    raise SystemExit("refusing to patch in place")
d = bytearray(open(src, "rb").read())

loads = []
for line in subprocess.run(["readelf", "-lW", src], capture_output=True, text=True, check=True).stdout.splitlines():
    f = line.split()
    if f and f[0] == "LOAD":
        loads.append((int(f[1], 16), int(f[2], 16), int(f[4], 16)))

def cstr(vaddr):
    for off, va, sz in loads:
        if va <= vaddr < va + sz:
            o = vaddr - va + off
            return bytes(d[o:d.find(b"\0", o)]).decode(errors="replace")
    return None

pat = struct.pack("<11I", 65536, 32, 4, 40, 1024, 106, 800, 106, 8, 256, 256)
hits, i = [], 0
while (i := d.find(pat, i)) >= 0:
    hits.append(i); i += 1

names = []
for h in hits:
    # this row's name pointer: the row started ROW bytes before the end of these integers
    name = cstr(struct.unpack_from("<Q", d, h + 44 - ROW + 8)[0])
    names.append(name)
    print(f"row at file offset 0x{h:x}: {name}")
if set(names) != EXPECTED or len(names) != len(EXPECTED):
    raise SystemExit(f"rows found do not match the 13 expected gfx10 rows: {sorted(n for n in names if n)}")
for h in hits:
    struct.pack_into("<I", d, h + 36, 1024)
open(dst, "wb").write(d)
print(f"wrote {dst}: total VGPRs 256 -> 1024 in {len(hits)} rows")
