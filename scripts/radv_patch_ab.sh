#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# RADV built from Mesa 26.1.8 source, stock against two community patches: the gfx1013 compute-queue
# exposure (tri3gubki-ops) and the signed-dot i24 reassociation (dmorazasanchez). Both loaded as
# side-by-side ICDs, never installed. Same llama.cpp Vulkan binary throughout.
set -u
D=${1:-~/radv-ab}; mkdir -p $D
V=~/llama-master/build-vk-f44/bin
log(){ echo "[$(date +%T)] $*" | tee -a $D/log; sync; }
samples(){ grep -oE '"samples_ts": \[[^]]*\]' "$1" | tr -d '"samples_ts:[] '; }
run(){
  local v=$1 tag=$2 out=$3; shift 3
  local I=$HOME/mesabuild/inst-$v
  env LD_LIBRARY_PATH=$I/lib64 VK_ICD_FILENAMES=$I/share/vulkan/icd.d/radeon_icd.x86_64.json \
    timeout -k 30 1800 $V/llama-bench "$@" -o jsonl > $out 2>/dev/null
  echo "[$(date +%T)] $v $tag [$(samples $out)]" | tee -a $D/log
}
log "system mesa: $(rpm -q mesa-vulkan-drivers); built: 26.1.8 base and patched"
for round in 1 2 3; do
  for v in base patched; do
    run $v "1.5B pp512 r$round" $D/${v}_q15_pp_$round.jsonl -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3
    run $v "1.5B tg64 r$round"  $D/${v}_q15_tg_$round.jsonl -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3
    run $v "8B pp512 r$round"   $D/${v}_q8_pp_$round.jsonl  -m /opt/models/qwen3-8b-q8_0.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3
    run $v "8B tg64 r$round"    $D/${v}_q8_tg_$round.jsonl  -m /opt/models/qwen3-8b-q8_0.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3
  done
done
for v in base patched; do
  I=$HOME/mesabuild/inst-$v
  g=$(env LD_LIBRARY_PATH=$I/lib64 VK_ICD_FILENAMES=$I/share/vulkan/icd.d/radeon_icd.x86_64.json \
      timeout -k 30 1800 $V/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+")
  log "gate $v: $g"
done
log DONE
