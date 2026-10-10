#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does ggml-cuda say, at debug level, that it could not pin the input buffer and fell back to ordinary memory?
# The 8B with a q4_0 cache at 118784 tokens is a run in that state (campaign run d118784). Start it with
# llama-bench's log on, keep it until the fill has started, stop it.
set -u
O=~/followup; L=/opt/bc250-rocm/lib64; B=~/llama-bench-nosnap/hip/llama-bench
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
log "3: hip qwen3-8b-q8_0 q4_0 d118784 with -v, stopped once the fill has started"
env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L LLAMA_BENCH_NO_DEPTH_CACHE=1 GGML_LOG_LEVEL=debug "$B" \
    -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa on -ctk q4_0 -ctv q4_0 -p 0 -n 64 -d 118784 -r 1 -v \
    > "$O/pinned_8b_q4_0_d118784.log" 2>&1 < /dev/null &
p=$!
for i in $(seq 1 60); do
  sleep 5
  kfdl=$(sudo cat /sys/kernel/debug/kfd/mem_limit | tr '\n' ' ')
  echo "$(date +%s) | $kfdl" >> "$O/pinned_8b_q4_0_d118784.mem"
  grep -qaE "pinned|compute buffer size" "$O/pinned_8b_q4_0_d118784.log" && [ $i -ge 12 ] && break
done
sleep 20; echo "$(date +%s) | $(sudo cat /sys/kernel/debug/kfd/mem_limit | tr '\n' ' ')" >> "$O/pinned_8b_q4_0_d118784.mem"
kill "$p"; wait "$p" 2>/dev/null
log "3: $(grep -aiE 'pinned' "$O/pinned_8b_q4_0_d118784.log" | head -3 | tr '\n' ' ' | cut -c1-300)"
log "3: $(grep -aiE 'host.*buffer size|CPU.*buffer size|compute buffer' "$O/pinned_8b_q4_0_d118784.log" | head -4 | tr '\n' ' ' | cut -c1-300)"
log done3
