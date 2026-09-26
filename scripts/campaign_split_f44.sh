#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Throughput with one llama-bench test per invocation, on the default Fedora 44 configuration.
# Running -p 512 and -n 64 in one invocation depresses and scatters the decode figure on large models
# (qwen3-14B: 21 to 26 t/s combined, 26 to 27 alone), the same artefact as putting several depths in one
# invocation. Here pp512 and tg64 are separate invocations, alternated by backend, three rounds each.
set -u
exec 9>~/.campaign.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=${1:-~/campaign-split-f44}; mkdir -p "$D"
export HSA_ENABLE_SDMA=0
# HIPBIN selects the ROCm build under test; the default is the three-patch build the first
# campaign measured, build-hip-fatile is the same tree with the RDNA1 flash-attention patch.
declare -A BIN=( [hip]=${HIPBIN:-~/llama-master/build-hip-f44/bin} [vk]=~/llama-master/build-vk-f44/bin )
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
samples () { grep -oE '"samples_ts": \[[^]]*\]' "$1" | tr -d '"samples_ts:[] '; }
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  for round in 1 2 3; do
    for be in hip vk; do
      B=${BIN[$be]}
      timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3 -o jsonl > $D/${be}_${m}_pp_$round.jsonl 2>/dev/null
      timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3 -o jsonl > $D/${be}_${m}_tg_$round.jsonl 2>/dev/null
      log "$m r$round $be pp512=[$(samples $D/${be}_${m}_pp_$round.jsonl)] tg64=[$(samples $D/${be}_${m}_tg_$round.jsonl)]"
    done
  done
done
log DONE
