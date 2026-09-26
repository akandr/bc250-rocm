#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Benchmark campaign on the default Fedora 44 configuration: ROCm 7.1.1 with the native gfx1013
# rocBLAS and corrected comgr installed through scripts/install_rocm711_overrides.sh (no environment
# variables), and the llama.cpp Vulkan backend on Mesa 26.1.8, same source tree (7ba604f with the
# three patches), same kernel 7.1.8 with the bc250 amdgpu module.
# Same models and flags as scripts/campaign_current_config.sh, plus the 27B and a depth ladder.
set -u
exec 9>~/.campaign.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=${1:-~/campaign-f44}; mkdir -p "$D"
HIP=~/llama-master/build-hip-f44/bin
VK=~/llama-master/build-vk-f44/bin
WIKI=~/wiki.test.raw
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }
val () { grep -aoE "$1 +\| +[0-9.]+" "$2" | grep -oE "[0-9.]+$" | head -1; }

log "=== $(grep -o "Fedora release [0-9]*" /etc/fedora-release), kernel $(uname -r), SELinux $(getenforce)"
log "    $(rpm -q rocm-hip rocblas rocm-comgr mesa-vulkan-drivers | tr '\n' ' ')"
log "    rocblas: $(ldd $HIP/libggml-hip.so.0 | grep -oE "/[^ ]*librocblas[^ ]*") comgr: $(ldconfig -p | grep -m1 -oE "/[^ ]*libamd_comgr.so.3$")"
log "    cmdline: $(tr ' ' '\n' < /proc/cmdline | grep -E 'amdgpu|ttm' | tr '\n' ' ')"

MODELS="qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs"
log "=== throughput, pp512 and tg64, -r 3, flash attention on"
for m in $MODELS; do
  timeout -k 30 2400 $HIP/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/hip_$m.log 2>&1
  timeout -k 30 2400 $VK/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/vk_$m.log 2>&1
  log "  $m: HIP pp512=$(val pp512 $D/hip_$m.log) tg64=$(val tg64 $D/hip_$m.log) | VK pp512=$(val pp512 $D/vk_$m.log) tg64=$(val tg64 $D/vk_$m.log)"
done

log "=== correctness gates, wikitext perplexity, context 2048 over eight chunks"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b deepseek-r1-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  timeout -k 30 3600 $HIP/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 8 > $D/gate_hip_$m.log 2>&1
  timeout -k 30 3600 $VK/llama-perplexity -m /opt/models/$m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 8 > $D/gate_vk_$m.log 2>&1
  log "  $m: HIP $(grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" $D/gate_hip_$m.log | cut -d= -f2) | VK $(grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" $D/gate_vk_$m.log | cut -d= -f2)"
done

log "=== decode at depth, qwen2.5-1.5B, tg64 after d tokens"
for be in hip vk; do
  B=$HIP; [ $be = vk ] && B=$VK
  timeout -k 30 3600 $B/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa on -p 0 -n 64 -d 0,4096,8192,16384,24576,30720 -r 2 > $D/depth_$be.log 2>&1
  log "  $be: $(grep -aoE "tg64 @ d[0-9]+ +\| +[0-9.]+" $D/depth_$be.log | sed -E 's/tg64 @ d([0-9]+) +\| +/\1:/' | tr '\n' ' ')"
done
log DONE
