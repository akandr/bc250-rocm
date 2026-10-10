#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# A/B for llama-bench's depth snapshot. After the depth prefill llama-bench stores the sequence state in a host
# vector (llama_state_seq_get_data, upstream PR #16944), also with -r 1 where it is never reused; clr pins that
# vector and it counts against the KFD limit. Build llama-bench with the store behind LLAMA_BENCH_NO_DEPTH_CACHE
# (a separate binary; the campaign's is untouched) and run the stock ROCm arm (graphs and packet capture on) at
# depths where the stock tool fails: with the snapshot, 14B f16 generates at 13312 and fails at 14336.
# The ceiling campaign is frozen while this runs and is resumed on exit, whatever happens.
set -u
O=~/nosnap-check; mkdir -p "$O"; cd ~/llama-bench-nosnap || exit 1
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
trap 'bash ~/resume_campaign.sh >> "$O/log" 2>&1; log "campaign resumed"' EXIT
B=~/llama-master/build-hip-pkf16; S=~/llama-master
log "building llama-bench-nosnap-hip"
c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP -DLLAMA_SHARED -DLLAMA_SUBPROCESS \
    -I. -I$S/common/. -I$S/common/../vendor -I$S/src/../include -I$S/ggml/src/../include -O3 -DNDEBUG \
    -o llama-bench-nosnap-hip main.cpp llama-bench.cpp -Wl,-rpath,$B/bin \
    $B/bin/libllama-common.so.0.0.1 $B/bin/libllama.so.0.0.1 $B/bin/libggml.so.0.19.0 $B/bin/libggml-cpu.so.0.19.0 \
    $B/bin/libggml-hip.so.0.19.0 $B/bin/libggml-base.so.0.19.0 $B/common/libllama-common-base.a >> "$O/log" 2>&1 \
  || { log "build failed"; exit 1; }
log "built; cooling down 120 s before measuring"; sleep 120
sample () { while :; do echo "$(date +%s) | $(sudo -n cat /sys/kernel/debug/kfd/mem_limit | tr "\n" " " | sed "s/  */ /g") | avail=$(free -m | awk "/^Mem:/{print \$7}")M"; sleep 3; done; }
for d in 16384 24576; do
  f=$O/hip_nosnap_qwen3-14b_f16_d$d
  sample > "$f.kfd" & sp=$!
  t0=$(date +%s)
  env LLAMA_BENCH_NO_DEPTH_CACHE=1 HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 timeout -k 30 3600 \
      ./llama-bench-nosnap-hip -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -ctk f16 -ctv f16 -p 0 -n 64 -d $d -r 1 -o jsonl \
      > "$f.jsonl" 2> "$f.err" < /dev/null
  rc=$?; secs=$(( $(date +%s) - t0 )); sleep 4; kill $sp
  ts=$(grep -oE '"avg_ts": [0-9.]+' "$f.jsonl" | head -1)
  sys=$(awk -F"|" '{split($2,a," "); for(i=1;i<=length(a);i++) if(a[i]=="System") print a[i+3]+0}' "$f.kfd" | uniq -c | tail -3 | tr -s " \n" " ")
  log "stock ROCm, no snapshot, 14B f16 d$d: rc=$rc after ${secs}s ${ts:-no result}; System mem used, last phases [samples value]: $sys"
  grep -iE "error|fail|abort" "$f.err" | tail -2 | sed "s/^/    /" >> "$O/log"
done
log done
