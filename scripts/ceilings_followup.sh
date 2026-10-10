#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Two questions the context-ceiling campaign left open, each answered by re-running one of its runs with
# more output, on the same builds and the same patched llama-bench:
#  1. which HIP call fails at the first generated token in the late-failure state: the 8B with a q8_0 cache at
#     65536 tokens, the shortest such run (683 s), where everything fitted with 2 MiB to spare, with
#     llama-bench's log on (-v) and clr's error log (AMD_LOG_LEVEL=1);
#  2. where the MoE with a q4_0 cache segfaults without graphs at 259072 tokens (72 s): under gdb.
set -u
O=~/followup; mkdir -p "$O"
L=/opt/bc250-rocm/lib64; B=~/llama-bench-nosnap/hip/llama-bench
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
kfd () { while :; do echo "$(date +%s) | $(sudo cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null | tr '\n' ' ') | avail=$(free -m | awk '/^Mem:/{print $7}')M"; sleep 5; done; }

log "1: hip qwen3-8b-q8_0 q8_0 d65536 with -v and AMD_LOG_LEVEL=1"
kfd > "$O/latefail_8b_q8_0_d65536.mem" & sp=$!
t0=$(date +%s)
env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L LLAMA_BENCH_NO_DEPTH_CACHE=1 AMD_LOG_LEVEL=1 timeout -k 30 7200 "$B" \
    -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa on -ctk q8_0 -ctv q8_0 -p 0 -n 64 -d 65536 -r 1 -v \
    > "$O/latefail_8b_q8_0_d65536.log" 2>&1 < /dev/null
rc=$?; kill $sp
log "1: rc=$rc after $(( $(date +%s) - t0 ))s; $(grep -aiE 'error|fail' "$O/latefail_8b_q8_0_d65536.log" | grep -v '^#' | head -3 | tr '\n' ' ' | cut -c1-300)"
sleep 30

log "2: hipng qwen3.6-35b-a3b-iq2m q4_0 d259072 under gdb"
kfd > "$O/segv_moe_q4_0_d259072.mem" & sp=$!
t0=$(date +%s)
timeout -k 30 1800 gdb -q -batch -nx \
    -ex "set pagination off" -ex "set confirm off" \
    -ex "handle all nostop noprint pass" -ex "handle SIGSEGV stop print nopass" -ex "handle SIGBUS stop print nopass" \
    -ex "set environment HSA_ENABLE_SDMA=0" -ex "set environment LD_LIBRARY_PATH=$L" \
    -ex "set environment LLAMA_BENCH_NO_DEPTH_CACHE=1" -ex "set environment GGML_CUDA_DISABLE_GRAPHS=1" \
    -ex "set environment AMD_LOG_LEVEL=1" \
    -ex run -ex "bt 40" -ex "info registers" -ex "info sharedlibrary" \
    --args "$B" -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -p 0 -n 64 -d 259072 -r 1 -v \
    > "$O/segv_moe_q4_0_d259072.log" 2>&1 < /dev/null
rc=$?; kill $sp
log "2: rc=$rc after $(( $(date +%s) - t0 ))s; $(grep -aE '^#0|^#1 |SIGSEGV|SIGBUS' "$O/segv_moe_q4_0_d259072.log" | head -4 | tr '\n' ' ' | cut -c1-300)"
log done
