#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Vulkan on Fedora 44: system Mesa 26.1.8 against Mesa 25.3.4 loaded privately, same llama.cpp binary, same boot.
# Mesa 25.3.4 comes from the Fedora 43 mesa-vulkan-drivers-25.3.4-7 rpm extracted to ~/mesa2534-f44 (nothing
# installed), with the two libraries it needs that Fedora 44 no longer ships (libLLVM.so.21.1,
# libdisplay-info.so.2) copied from the Fedora 43 subvolume into ~/mesa2534-f44/lib, and selected with
# VK_ICD_FILENAMES. One test per invocation, drivers alternated, three rounds of -r 3.
set -u
D=${1:-~/mesa-ab-f44}; mkdir -p "$D"
M=~/mesa2534-f44
VK=~/llama-master/build-vk-f44/bin
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
samples () { grep -oE '"samples_ts": \[[^]]*\]' "$1" | tr -d '"samples_ts:[] '; }
run () { # run <driver> <outfile> <llama-bench args...>
  local drv=$1 out=$2; shift 2
  if [ $drv = m2534 ]; then
    env LD_LIBRARY_PATH=$M/lib VK_ICD_FILENAMES=$M/usr/share/vulkan/icd.d/radeon_icd.x86_64.json timeout -k 30 1800 $VK/llama-bench "$@" -o jsonl > $out 2>$out.err
  else
    timeout -k 30 1800 $VK/llama-bench "$@" -o jsonl > $out 2>$out.err
  fi
}
log "system: $(rpm -q mesa-vulkan-drivers); private: $(LD_LIBRARY_PATH=$M/lib VK_ICD_FILENAMES=$M/usr/share/vulkan/icd.d/radeon_icd.x86_64.json vulkaninfo --summary 2>/dev/null | grep -m1 driverInfo | tr -s ' ')"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b qwen3.8-27b-iq3xxs; do
  for round in 1 2 3; do
    for drv in m2618 m2534; do
      run $drv $D/${drv}_${m}_pp_$round.jsonl -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 0 -r 3
      run $drv $D/${drv}_${m}_tg_$round.jsonl -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3
      grep -m1 -oE "Mesa [0-9.]+" $D/${drv}_${m}_pp_$round.jsonl.err > /dev/null 2>&1 || true
      log "$m r$round $drv pp512=[$(samples $D/${drv}_${m}_pp_$round.jsonl)] tg64=[$(samples $D/${drv}_${m}_tg_$round.jsonl)]"
    done
  done
done
for drv in m2618 m2534; do
  if [ $drv = m2534 ]; then E="env LD_LIBRARY_PATH=$M/lib VK_ICD_FILENAMES=$M/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"; else E=env; fi
  g=$($E timeout -k 30 1800 $VK/llama-perplexity -m /opt/models/qwen3-14b.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+")
  log "gate qwen3-14B $drv: $g"
done
log DONE
