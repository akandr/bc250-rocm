#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Adds two switches to ggml-cuda.cu, on top of ggml-org/llama.cpp#27301 (allocation dependencies from graph_optimize):
#   GGML_CUDA_GRAPH_OPT_DEBUG=1       report, for the first captures, every tensor that a concurrent region writes over
#                                     a tensor which another stream of the same region reads from outside the region,
#                                     and dump the first region (nodes, streams, address ranges, execution groups)
#   GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1  keep every tensor a concurrent region writes or reads allocated until the
#                                     region's join, so the allocator cannot recycle it inside the region
# Neither changes anything when unset.
import sys

path = sys.argv[1]
src = open(path).read()


def sub(old, new):
    global src
    n = src.count(old)
    if n != 1:
        sys.exit(f"anchor found {n} times, expected once: {old.strip()[:90]!r}")
    src = src.replace(old, new)


# --- graph_optimize: switches, the dependencies, a short report -------------------------------------------------
sub(r"""    if (!enable_graph_optimization) {
        return;
    }
""", r"""    if (!enable_graph_optimization) {
        return;
    }

    static const bool graphopt_alloc_deps = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT_ALLOC_DEPS");
        return env != nullptr && atoi(env) == 1;
    }();
    static const bool graphopt_debug_opt = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT_DEBUG");
        return env != nullptr && atoi(env) == 1;
    }();
    int graphopt_dep_regions = 0;
""")

sub(r"""                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);
""", r"""                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // keep every tensor of the region, and every tensor it reads, allocated until the join: the streams
                // run the branches in any relative order, so nothing may be recycled before all of them have finished
                if (graphopt_alloc_deps && params != nullptr && params->add_alloc_dep != nullptr) {
                    ggml_tensor * until = const_cast<ggml_tensor *>(join_node);
                    for (int k = fork_node_idx + 1; k < join_node_idx; ++k) {
                        ggml_tensor * n = cgraph->nodes[k];
                        params->add_alloc_dep(params->user_data, n, until);
                        for (int s = 0; s < GGML_MAX_SRC; ++s) {
                            if (n->src[s] != nullptr) {
                                params->add_alloc_dep(params->user_data, n->src[s], until);
                            }
                        }
                    }
                    graphopt_dep_regions++;
                }
""")

sub(r"""                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}
""", r"""                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }

    if (graphopt_debug_opt && graphopt_alloc_deps) {
        static int graphopt_opt_reports = 0;
        if (graphopt_opt_reports++ < 3) {
            fprintf(stderr, "graphopt-allocdeps: %d regions kept allocated until their join\n", graphopt_dep_regions);
        }
    }
}
""")

# --- graph_compute: the check, the dump, the per-capture summary ---------------------------------------------------
sub(r"""    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
""", r"""    // is_valid() compares the branches' outputs with each other and rejects a branch that reads another branch's
    // output; it does not compare a branch's output with what the other branches read from outside the region,
    // such as the fork node's output that all of them read. With GGML_CUDA_GRAPH_OPT_DEBUG=1 that overlap is
    // reported for the first captures.
    static const bool graphopt_debug = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT_DEBUG");
        return env != nullptr && atoi(env) == 1;
    }();
    static int  graphopt_debug_captures = 0;
    static bool graphopt_dumped         = false;
    bool        graphopt_dump           = false;
    bool        graphopt_region_hit     = false;
    int         graphopt_regions_run    = 0;
    int         graphopt_regions_hit    = 0;
    int         graphopt_conflicts      = 0;

    const auto graphopt_check_write = [&](const ggml_cuda_concurrent_event * ev, const ggml_tensor * w, int sw) {
        if (w->data == nullptr || ggml_nbytes(w) == 0) {
            return;
        }
        const char * w0 = (const char *) w->data;
        const char * w1 = w0 + ggml_nbytes(w);
        for (const auto & [r, sr] : ev->stream_mapping) {
            if (sr == sw || ggml_cuda_is_view_or_noop(r)) {
                continue;
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                const ggml_tensor * rs = r->src[s];
                if (rs == nullptr || rs->data == nullptr || ev->stream_mapping.count(rs) != 0) {
                    continue;
                }
                const char * r0 = (const char *) rs->data;
                const char * r1 = r0 + ggml_nbytes(rs);
                if (w0 < r1 && r0 < w1) {
                    graphopt_conflicts++;
                    graphopt_region_hit = true;
                    if (graphopt_conflicts <= 24) {
                        fprintf(stderr, "graphopt-recycle: capture %d: %s (%s, stream %d) writes [%p, +%zu) over %s (%s), "
                                        "which %s (%s, stream %d) reads at [%p, +%zu)\n",
                                graphopt_debug_captures, w->name, ggml_op_name(w->op), sw, w->data, ggml_nbytes(w),
                                rs->name, ggml_op_name(rs->op), r->name, ggml_op_name(r->op), sr, rs->data, ggml_nbytes(rs));
                    }
                }
            }
        }
    };

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
""")

sub(r"""                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };
""", r"""                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }

            if (graphopt_debug && graphopt_debug_captures == 0 && !graphopt_dumped) {
                graphopt_dumped = true;
                graphopt_dump   = true;
                fprintf(stderr, "graphopt-region: fork %s, join %s, %d streams; nodes in their original order:\n",
                        node->name, concurrent_event->join_node->name, concurrent_event->n_streams);
                for (const ggml_tensor * t : concurrent_event->original_order) {
                    const auto it = concurrent_event->stream_mapping.find(t);
                    fprintf(stderr, "graphopt-region:   stream %d %-10s %-28s [%p, +%zu)",
                            it == concurrent_event->stream_mapping.end() ? -1 : it->second,
                            ggml_op_name(t->op), t->name, t->data, ggml_nbytes(t));
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        const ggml_tensor * ts = t->src[s];
                        if (ts != nullptr && concurrent_event->stream_mapping.count(ts) == 0) {
                            fprintf(stderr, " <- %s [%p, +%zu)", ts->name, ts->data, ggml_nbytes(ts));
                        }
                    }
                    fprintf(stderr, "\n");
                }
            }
        }
    };
""")

sub(r"""                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }
""", r"""                    should_launch_concurrent_events = should_launch_concurrent_events && event.is_valid();
                }
            }
            const size_t graphopt_n_regions = stream_ctx.concurrent_events.size();
""")

sub(r"""                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
""", r"""                        if (graphopt_debug && graphopt_debug_captures < 4) {
                            graphopt_regions_run++;
                            if (graphopt_region_hit) {
                                graphopt_regions_hit++;
                                graphopt_region_hit = false;
                            }
                            graphopt_dump = false;
                        }
                        is_concurrent_event_active = false;
                        concurrent_event           = nullptr;
                    } else {
""")

sub(r"""                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
""", r"""                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                if (graphopt_debug && graphopt_debug_captures < 4 && is_concurrent_event_active) {
                    graphopt_check_write(concurrent_event, cgraph->nodes[i + nodes_to_skip], cuda_ctx->curr_stream_no);
                    if (graphopt_dump) {
                        fprintf(stderr, "graphopt-exec: stream %d: %s (%s)%s%s\n", cuda_ctx->curr_stream_no, node->name,
                                ggml_op_name(node->op), nodes_to_skip != 0 ? " fused through " : "",
                                nodes_to_skip != 0 ? cgraph->nodes[i + nodes_to_skip]->name : "");
                    }
                }
""")

sub(r"""                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }
""", r"""                if (!is_concurrent_event_active) {
                    try_launch_concurrent_event(node);
               }
            }

            if (graphopt_debug && graphopt_debug_captures < 4 && graphopt_n_regions > 0) {
                fprintf(stderr, "graphopt-debug: capture %d: %zu regions, %s; %d ran on concurrent streams, "
                                "%d of them writing over %d sources another stream reads\n",
                        graphopt_debug_captures, graphopt_n_regions,
                        should_launch_concurrent_events ? "all valid" : "not all valid, so no streams",
                        graphopt_regions_run, graphopt_regions_hit, graphopt_conflicts);
                graphopt_debug_captures++;
            }
""")

open(path, "w").write(src)
print(f"instrumented {path}")
