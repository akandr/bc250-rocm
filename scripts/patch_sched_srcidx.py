#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Diagnostic-only: make GGML_SCHED_DEBUG=2 print each source as a node index as well as a name.
#
# The names it prints are truncated to twenty characters and layers reuse them, so "norm-0" occurs at
# nodes 1, 49 and 59 of a single graph and producer-consumer edges cannot be reconstructed from the
# dump. An index is unambiguous. The lookup is linear per source, which is quadratic over the graph and
# only acceptable because this runs behind a debug environment variable.
#
# Applied, used, and reverted; it is not part of the patch set.
import sys, re
p = sys.argv[1]
s = open(p).read()
old = """                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));"""
new = """                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                int src_idx = -1;
                for (int k = 0; k < graph->n_nodes; k++) {
                    if (graph->nodes[k] == src) { src_idx = k; break; }
                }
                GGML_LOG_DEBUG(" src#%d(%20.20s) (%5.5s) [%5.5s %8.8s]", src_idx, src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));"""
if new in s:
    print("already applied"); sys.exit(0)
if s.count(old) != 1:
    print(f"anchor not found exactly once ({s.count(old)}), refusing", file=sys.stderr); sys.exit(1)
open(p, "w").write(s.replace(old, new))
print("applied")
