#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Port an ordinary CUDA program to this board with hipify-perl and see what it costs.
# hipify itself is not installed: `dnf install hipify` pulls 153 packages and REMOVES both kernels
# on this machine, so the script is extracted from the RPM with rpm2cpio and run from /tmp.
set -u
cd /tmp
HIPIFY=${HIPIFY:-/tmp/hipify/usr/bin/hipify-perl}
echo "# $(date -Iseconds)  edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
echo
echo "## translate"
perl "$HIPIFY" cuda_demo.cu > cuda_demo.hip.cpp
echo "lines        $(wc -l < cuda_demo.cu) -> $(wc -l < cuda_demo.hip.cpp)"
echo "changed      $(diff cuda_demo.cu cuda_demo.hip.cpp | grep -c '^<') lines, $(diff cuda_demo.cu cuda_demo.hip.cpp | grep '^<' | grep -vcE '^< *//') of them code"
echo "kernel body  untouched: __global__, threadIdx, blockIdx and <<< >>> are shared syntax"
echo "hand edits   0"
echo
echo "## compile, and the two flags Fedora needs that hipify does not know about"
echo "   hipify emits #include <hipblas.h>; Fedora packages it as hipblas/hipblas.h  -> -I/usr/include/hipblas"
echo "   hipcc did not pull in the HIP runtime for this link                          -> -lamdhip64"
hipcc -O2 --offload-arch=gfx1013 -I/usr/include/hipblas cuda_demo.hip.cpp -o cuda_demo \
      -L/usr/lib64 -lhipblas -lamdhip64 2>&1 | grep -viE '^$' | head -4
echo "compiled     $([ -x ./cuda_demo ] && echo yes || echo NO)"
echo
echo "## run, both halves checked against a CPU reference"
LD_LIBRARY_PATH=/opt/bc250-rocm/lib64:/usr/lib64 HSA_ENABLE_SDMA=0 ./cuda_demo
echo
echo "# native rocBLAS SGEMM at the same N=4096 is 4541.1 GFLOP/s (logs/torch-rocblas-bench-2026-09-24/)"
echo "# edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
