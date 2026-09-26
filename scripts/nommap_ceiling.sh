#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The qwen3-8B at a 16128-token depth aborts with a ROCm error on the thirteen-patch build. It ran at
# that exact depth in August (logs/decode-variance-state-2026-08-21, 17.67 t/s) and the front page's
# ceiling table has it generating at 16384. So: where does it start failing, and is it the patches?
set -u
O=~/depth-ceiling; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
VK=~/llama-master/build-vk-f44/bin
M=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "page fault \(src_id|Queue preemption failed"; }

log "start, available $(free -m | awk '/^Mem:/{print $7}') MiB, faults $(faults)"
for d in 4096 8192 12288 14336 15360 16128 16384; do
  out=$(LD_LIBRARY_PATH=$L timeout -k 20 600 "$HIP/llama-bench" -m "$M" -mmp 0 -ngl 99 -fa on \
          -p 0 -n 8 -d $d -r 1 2>&1)
  r=$(echo "$out" | grep -aE "tg8" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
  e=$(echo "$out" | grep -aoE "ROCm error|failed to load model|out of memory" | head -1)
  log "hip  d=$d tg8=${r:-FAIL} ${e:+[$e]} faults=$(faults)"
  sleep 20
done

# Is it the packed-fp16 GEMM? It is a prefill kernel and a 16128-token depth is a prefill.
for d in 15360 16128; do
  out=$(GGML_RDNA1_PKF16=0 LD_LIBRARY_PATH=$L timeout -k 20 600 "$HIP/llama-bench" -m "$M" -mmp 0 \
          -ngl 99 -fa on -p 0 -n 8 -d $d -r 1 2>&1)
  r=$(echo "$out" | grep -aE "tg8" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
  e=$(echo "$out" | grep -aoE "ROCm error|failed to load model|out of memory" | head -1)
  log "hip  d=$d GGML_RDNA1_PKF16=0 tg8=${r:-FAIL} ${e:+[$e]} faults=$(faults)"
  sleep 20
done

# And does flash attention matter? The four-patch tile rows are what made deep context work at all.
for d in 16128; do
  out=$(LD_LIBRARY_PATH=$L timeout -k 20 600 "$HIP/llama-bench" -m "$M" -mmp 0 -ngl 99 -fa off \
          -p 0 -n 8 -d $d -r 1 2>&1)
  r=$(echo "$out" | grep -aE "tg8" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
  e=$(echo "$out" | grep -aoE "ROCm error|failed to load model|out of memory" | head -1)
  log "hip  d=$d -fa off tg8=${r:-FAIL} ${e:+[$e]} faults=$(faults)"
  sleep 20
done

# Vulkan as the control: same model, same depth, other backend.
for d in 16128; do
  out=$(timeout -k 20 600 "$VK/llama-bench" -m "$M" -mmp 0 -ngl 99 -fa on -p 0 -n 8 -d $d -r 1 2>&1)
  r=$(echo "$out" | grep -aE "tg8" | awk -F'|' '{v=$(NF-1); gsub(/ /,"",v); print v}')
  e=$(echo "$out" | grep -aoE "error|failed to load model|out of memory" | head -1)
  log "vk   d=$d tg8=${r:-FAIL} ${e:+[$e]}"
done
log "faults at end: $(faults)"
touch "$O/DONE"; log DONE
