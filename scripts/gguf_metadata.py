#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""The general.* metadata of every GGUF in a directory, read with llama.cpp's own gguf-py, plus each
file's trained context length, block count and expert count. This is what the model files say about
themselves: name, quantisation, and where they came from if whoever quantised them wrote it in.

    PYTHONPATH=<llama.cpp>/gguf-py python3 gguf_metadata.py [/opt/models]
"""
import glob
import os
import sys

from gguf import GGUFReader, GGUFValueType


def value(f):
    t = f.types[-1] if f.types else None
    if f.types and f.types[0] == GGUFValueType.ARRAY:
        n = len(f.data)
        if t == GGUFValueType.STRING:
            items = [f.parts[i].tobytes().decode("utf-8", "replace") for i in f.data[:8]]
            return items + ([f"... {n} items"] if n > 8 else [])
        return f"array of {n}"
    if t == GGUFValueType.STRING:
        return f.parts[f.data[0]].tobytes().decode("utf-8", "replace")
    return f.parts[f.data[0]].tolist()[0] if f.data else None


d = sys.argv[1] if len(sys.argv) > 1 else "/opt/models"
for p in sorted(glob.glob(os.path.join(d, "*.gguf"))):
    r = GGUFReader(p)
    print(f"=== {os.path.basename(p)}  {os.path.getsize(p)} bytes")
    for k, f in r.fields.items():
        if k.startswith("general.") or k.endswith((".context_length", ".block_count", ".expert_count")):
            v = value(f)
            if isinstance(v, str) and len(v) > 160:
                v = v[:160] + "..."
            print(f"  {k} = {v}")
    print(f"  tensors = {len(r.tensors)}")
