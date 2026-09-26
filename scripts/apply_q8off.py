#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
"""Experiment build only: take Q8_0 out of the RDNA1 float matvec (back to MMVQ) to see whether the
qwen3-8B's 4 percent decode loss since patch 5 is this kernel. Usage: apply_q8off.py <ggml-cuda dir>"""
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "mmvq-rdna1-f32.cu"; s = p.read_text()
old = "        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_0:\n"
assert old in s; s = s.replace(old, "        case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K:\n", 1); p.write_text(s); print("q8_0 off the float kernel")
