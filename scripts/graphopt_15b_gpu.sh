#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# graphopt_check.sh's 1.5B runs on 2 October found no ROCm device and ran on the CPU, so they say nothing about
# GGML_CUDA_GRAPH_OPT=1 on that model. The same comparison on the GPU, every other GPU user stopped, campaign
# build: three replies each, default, the option, graphs disabled.
set -u
O=~/graphopt-15b-gpu; mkdir -p "$O"
H=~/llama-master/build-hip-pkf16/bin; L=/opt/bc250-rocm/lib64
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
log "services: $(systemctl is-active ollama hw-watcher.timer crond | tr '\n' ' ')"
gen () {  # tag env...
  local tag=$1 m=qwen2.5-1.5b-q4km; shift
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 600 "$H/llama-cli" -m /opt/models/$m.gguf -ngl 99 -fa on -st \
      -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$O/${m}_$tag.raw" 2> "$O/${m}_$tag.err" < /dev/null
  local rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$O/${m}_$tag.raw" > "$O/${m}_$tag.txt"
  local gpu=$(grep -c "no ROCm-capable device" "$O/${m}_$tag.err") off=$(grep -oE "offloaded [0-9]+/[0-9]+ layers to GPU" "$O/${m}_$tag.err" | head -1)
  log "$tag: rc=$rc md5=$(md5sum < "$O/${m}_$tag.txt" | cut -c1-8) $(wc -c < "$O/${m}_$tag.txt")B${off:+, $off}$( [ "$gpu" != 0 ] && echo ", NO GPU") | $(tr '\n' ' ' < "$O/${m}_$tag.txt" | cut -c1-90)"
}
for i in 1 2 3; do gen default$i X=1; done
for i in 1 2 3; do gen graphopt$i GGML_CUDA_GRAPH_OPT=1; done
for i in 1 2 3; do gen nographs$i GGML_CUDA_DISABLE_GRAPHS=1; done
log done
