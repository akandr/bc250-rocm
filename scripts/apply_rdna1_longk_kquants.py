#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The long-K row gave four warps to every non-simple type with K >= 8192; the IQ types' vec_dot is long
# enough that one wave per row stays better (qwen3.8-27B IQ3_XXS lost 4 percent). Restrict it to the K-quants.
import sys
p = sys.argv[1] + "/ggml/src/ggml-cuda/mmvq.cu"; s = open(p).read()
old = """                default:
                    return long_k ? 4 : 1;   // K >= 8192: four warps split the long row
"""
new = """                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                    return long_k ? 4 : 1;   // K >= 8192: four warps split the long row
                default:
                    return 1;                // IQ types: one wave per row at any K
"""
assert s.count(old) == 1; open(p, "w").write(s.replace(old, new, 1)); print("long_k restricted to the K-quants")
