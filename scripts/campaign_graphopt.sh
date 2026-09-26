#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The campaign of scripts/campaign_split_f44.sh with a third arm: ROCm with
# GGML_CUDA_GRAPH_OPT=1, ggml-cuda's multi-stream graph optimisation, which is off by default
# (logs/graph-opt-2026-09-24/).
#
# Everything else is deliberately identical to that script so the figures can be compared with the
# campaigns already in logs/: one llama-bench test per invocation, because running -p 512 and -n 64
# together depresses and scatters decode on large models; pp512 and tg64 alternated by backend;
# three rounds; -mmp 0 -ngl 99 -fa on; HSA_ENABLE_SDMA=0.
set -u
exec 9>~/.campaign.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=${1:-~/campaign-graphopt}; mkdir -p "$D"
export HSA_ENABLE_SDMA=0

HIPBIN=${HIPBIN:-~/llama-master/build-hip-pkf16/bin}
VKBIN=${VKBIN:-~/llama-master/build-vk-f44/bin}

log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
samples () { grep -oE '"samples_ts": \[[^]]*\]' "$1" | tr -d '"samples_ts:[] '; }

# arm -> binary dir and the environment that distinguishes it
run_arm () {  # $1 arm  $2 model  $3 flags  $4 outfile
  case $1 in
    hip)    env -u GGML_CUDA_GRAPH_OPT LD_LIBRARY_PATH="$HIPBIN" timeout -k 30 1800 \
              "$HIPBIN/llama-bench" -m /opt/models/$2.gguf -mmp 0 -ngl 99 -fa on $3 -r 3 -o jsonl > "$4" 2>/dev/null ;;
    hipopt) env GGML_CUDA_GRAPH_OPT=1 LD_LIBRARY_PATH="$HIPBIN" timeout -k 30 1800 \
              "$HIPBIN/llama-bench" -m /opt/models/$2.gguf -mmp 0 -ngl 99 -fa on $3 -r 3 -o jsonl > "$4" 2>/dev/null ;;
    vk)     env LD_LIBRARY_PATH="$VKBIN" timeout -k 30 1800 \
              "$VKBIN/llama-bench" -m /opt/models/$2.gguf -mmp 0 -ngl 99 -fa on $3 -r 3 -o jsonl > "$4" 2>/dev/null ;;
  esac
}

log "start  hip=$HIPBIN  vk=$VKBIN  free=$(free -m | awk '/Mem:/{print $7}')MiB"
log "kfd holders: $(fuser -v /dev/kfd 2>&1 | tail -1 | tr -s ' ')"

for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  for round in 1 2 3; do
    for be in hip hipopt vk; do
      run_arm "$be" "$m" "-p 512 -n 0" "$D/${be}_${m}_pp_$round.jsonl"
      run_arm "$be" "$m" "-p 0 -n 64"  "$D/${be}_${m}_tg_$round.jsonl"
      log "$m r$round $be pp512=[$(samples $D/${be}_${m}_pp_$round.jsonl)] tg64=[$(samples $D/${be}_${m}_tg_$round.jsonl)] edge=$(sensors 2>/dev/null | awk '/edge/{print $2}')"
    done
  done
done
log DONE
