#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# llama.cpp master against the measured base, at the corrected GPU clock.
# An earlier comparison ran while the governor was oscillating, so it is repeated here:
#   base    7ba604f + the three patches (build-hip-f44)
#   master  bfdc321 + 0002 and 0003 (llama-new/build-hip-f44), which needs --load-mode none instead of -mmp 0
# One test per invocation, builds alternated, three rounds of -r 3, plus the two short gates.
set -u
D=${1:-~/master-vs-base-f44}; mkdir -p "$D"
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
samples () { grep -oE '"samples_ts": \[[^]]*\]' "$1" | tr -d '"samples_ts:[] '; }
log "clock under load must stay at one step; config: $(tr -d ' \n' < /etc/oberon-config.yaml)"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b qwen3.6-35b-a3b-iq2m; do
  for round in 1 2 3; do
    for v in base master; do
      B=~/llama-master/build-hip-f44/bin; LM="-mmp 0"
      [ $v = master ] && { B=~/llama-new/build-hip-f44/bin; LM="--load-mode none"; }
      timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf $LM -ngl 99 -fa on -p 512 -n 0 -r 3 -o jsonl > $D/${v}_${m}_pp_$round.jsonl 2>/dev/null
      timeout -k 30 1800 $B/llama-bench -m /opt/models/$m.gguf $LM -ngl 99 -fa on -p 0 -n 64 -r 3 -o jsonl > $D/${v}_${m}_tg_$round.jsonl 2>/dev/null
      log "$m r$round $v pp512=[$(samples $D/${v}_${m}_pp_$round.jsonl)] tg64=[$(samples $D/${v}_${m}_tg_$round.jsonl)]"
    done
  done
done
for v in base master; do
  B=~/llama-master/build-hip-f44/bin; NM="--no-mmap"
  [ $v = master ] && { B=~/llama-new/build-hip-f44/bin; NM="--load-mode none"; }
  g1=$(timeout -k 30 1800 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf $NM -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+")
  g2=$(timeout -k 30 1800 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf $NM -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+")
  log "gates $v: 1.5B $g1 | 8B $g2"
done
log DONE
