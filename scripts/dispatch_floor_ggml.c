// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
// What one graph node costs on each ggml backend when the node does almost nothing.
//
// logs/dispatch-floor-2026-09-22/ measured ROCm's dispatch floor with a bare HIP kernel and left an
// open question on the page: Vulkan's own floor was never measured, so "ROCm's floor is the whole of
// the MoE's decode deficit" was never tested against the alternative that Vulkan pays the same.
//
// This measures both through ggml, which is the comparison that matters for llama.cpp: each backend
// dispatches through its own real path (ggml-cuda on a stream or a captured graph, ggml-vulkan
// recording into a command buffer), so what comes out is the cost of a node in the backend's own
// idiom, not of a hand-written launch.
//
// Method: N elementwise nodes over tensors of S elements, in one of two shapes.
//
//   chain  N nodes each consuming the one before, so the backend cannot overlap any of them. This
//          is the serialized cost of a node.
//   fan    N nodes on N independent inputs, so nothing forbids the backend from overlapping them.
//
// The difference between the two is what a backend gains from having independent work available.
// A backend that dispatches everything onto one serialized stream scores the same in both; one that
// keeps several dispatches in flight is faster in `fan`.
//
// At the smallest S the tensor is 4 KiB and the kernel's own work is negligible, so the per-node
// time there is the floor directly. A least-squares fit over all S is also reported, but it is
// biased: small tensors do not reach peak bandwidth, so the fit's intercept sits below the directly
// measured floor. The directly measured smallest-S number is the one to read.
//
// Build (on the board, against a build tree that has the backend):
//   gcc -O2 -o dispatch_floor_ggml dispatch_floor_ggml.c \
//       -I<tree>/ggml/include -L<build>/bin -lggml -lggml-base -lm
// Run with LD_LIBRARY_PATH=<build>/bin so the backend .so is found.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_NODES 256
#define N_REPS  40
#define N_BEST  5

static const int64_t SIZES[] = { 1024, 16384, 65536, 262144, 1048576 };
#define N_SIZES ((int) (sizeof(SIZES) / sizeof(SIZES[0])))

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

// Best-of-N_BEST per-node time, in microseconds, for N_NODES nodes over n elements.
// fan = 0 builds a dependent chain, fan = 1 builds independent branches.
static double measure(ggml_backend_t backend, int64_t n, int fan, int * out_nodes) {
    const size_t overhead = ggml_tensor_overhead() * (2 * N_NODES + 16) + ggml_graph_overhead_custom(N_NODES + 8, false);

    struct ggml_init_params ip = {
        /*.mem_size   =*/ overhead,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    struct ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        return -1.0;
    }

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, N_NODES + 8, false);

    // ggml_sqrt is elementwise, is in no backend's fusion pattern list, and is a fixed point at 1.0,
    // so a long chain neither overflows nor denormalises.
    struct ggml_tensor * inputs[N_NODES];
    int                  n_inputs = 0;

    if (fan) {
        for (int i = 0; i < N_NODES; i++) {
            struct ggml_tensor * xi = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
            ggml_set_input(xi);
            inputs[n_inputs++] = xi;

            struct ggml_tensor * yi = ggml_sqrt(ctx, xi);
            ggml_set_output(yi);
            ggml_build_forward_expand(gf, yi);
        }
    } else {
        struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        ggml_set_input(x);
        inputs[n_inputs++] = x;

        struct ggml_tensor * cur = x;
        for (int i = 0; i < N_NODES; i++) {
            cur = ggml_sqrt(ctx, cur);
        }
        ggml_set_output(cur);
        ggml_build_forward_expand(gf, cur);
    }

    *out_nodes = ggml_graph_n_nodes(gf);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return -1.0;
    }

    float * host = malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; i++) {
        host[i] = 1.0f;
    }
    for (int i = 0; i < n_inputs; i++) {
        ggml_backend_tensor_set(inputs[i], host, 0, n * sizeof(float));
    }
    free(host);

    // Warm up: first call compiles pipelines, allocates command buffers and, on ggml-cuda, may
    // capture a graph. Several calls so any capture has settled before timing.
    for (int i = 0; i < 3; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    double best = 1e30;
    for (int b = 0; b < N_BEST; b++) {
        const double t0 = now_us();
        for (int r = 0; r < N_REPS; r++) {
            ggml_backend_graph_compute(backend, gf);
        }
        ggml_backend_synchronize(backend);
        const double t1 = now_us();

        const double per_node = (t1 - t0) / ((double) N_REPS * (double) *out_nodes);
        if (per_node < best) {
            best = per_node;
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return best;
}

int main(int argc, char ** argv) {
    const char * want = argc > 1 ? argv[1] : NULL;

    ggml_backend_load_all();

    ggml_backend_dev_t dev = NULL;
    const int ndev = ggml_backend_dev_count();
    for (int i = 0; i < ndev; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        // ggml classifies this APU as an integrated GPU, not a discrete one, so take anything that
        // is not the CPU backend.
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            continue;
        }
        if (want && !strstr(ggml_backend_dev_name(d), want) && !strstr(ggml_backend_dev_description(d), want)) {
            continue;
        }
        dev = d;
        break;
    }

    if (!dev) {
        fprintf(stderr, "no GPU device%s%s found. devices seen:\n", want ? " matching " : "", want ? want : "");
        for (int i = 0; i < ndev; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            fprintf(stderr, "  %s (%s)\n", ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        }
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) {
        fprintf(stderr, "failed to init backend\n");
        return 1;
    }

    printf("device      : %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    printf("nodes       : %d x ggml_sqrt, %d reps a measurement, best of %d\n", N_NODES, N_REPS, N_BEST);

    double floor_direct[2] = { 0, 0 };

    for (int fan = 0; fan <= 1; fan++) {
        printf("\n== %s ==\n", fan ? "fan: independent nodes, overlap permitted"
                                   : "chain: dependent nodes, no overlap possible");
        printf("%12s %14s %16s\n", "elements", "per node (us)", "GB/s implied");

        double xs[N_SIZES], ys[N_SIZES];
        int    nfit = 0;

        for (int i = 0; i < N_SIZES; i++) {
            int    nodes = 0;
            double per   = measure(backend, SIZES[i], fan, &nodes);
            if (per < 0) {
                fprintf(stderr, "measurement failed at n=%lld fan=%d\n", (long long) SIZES[i], fan);
                continue;
            }
            const double gbs = 2.0 * SIZES[i] * sizeof(float) / (per * 1e-6) / 1e9;
            printf("%12lld %14.3f %16.1f\n", (long long) SIZES[i], per, gbs);

            if (nfit == 0) {
                floor_direct[fan] = per;
            }
            xs[nfit] = (double) SIZES[i];
            ys[nfit] = per;
            nfit++;
        }

        if (nfit < 2) {
            continue;
        }

        double sx = 0, sy = 0, sxx = 0, sxy = 0;
        for (int i = 0; i < nfit; i++) {
            sx += xs[i];
            sy += ys[i];
            sxx += xs[i] * xs[i];
            sxy += xs[i] * ys[i];
        }
        const double slope    = (nfit * sxy - sx * sy) / (nfit * sxx - sx * sx);
        const double intercept = (sy - slope * sx) / nfit;

        printf("floor, measured at smallest tensor : %8.3f us per node\n", floor_direct[fan]);
        printf("fit intercept (biased low)         : %8.3f us per node\n", intercept);
        printf("fit throughput                     : %8.1f GB/s (read+write f32)\n",
               2.0 * sizeof(float) / slope / 1e3);
    }

    if (floor_direct[0] > 0 && floor_direct[1] > 0) {
        printf("\nchain %.3f us, fan %.3f us, fan/chain %.3f  (1.0 = no overlap gained)\n",
               floor_direct[0], floor_direct[1], floor_direct[1] / floor_direct[0]);
    }

    ggml_backend_free(backend);
    return 0;
}
