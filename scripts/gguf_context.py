#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Print each model's trained context length (<arch>.context_length) and RoPE scaling keys from its GGUF header.
import struct, sys
SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

def rstr(f):
    n, = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")

def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        if et == 8:
            for _ in range(n):
                rstr(f)
        else:
            f.seek(SIZES[et] * n, 1)
        return f"[array of {n}]"
    return struct.unpack(FMT[t], f.read(SIZES[t]))[0]

for path in sys.argv[1:]:
    with open(path, "rb") as f:
        magic, ver, nt, nkv = struct.unpack("<4sIQQ", f.read(24))
        found = {}
        for _ in range(nkv):
            k = rstr(f)
            t, = struct.unpack("<I", f.read(4))
            v = rval(f, t)
            if k == "general.architecture" or k.endswith(".context_length") or "rope.scaling" in k or k == "general.name":
                found[k] = v
        print(path.split("/")[-1], found)
