#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Two things the packet-capture validation could not settle. (1) Does GGML_CUDA_GRAPH_OPT=1 break single-token
# decode? The validation (GRAPH_OPT=1 in both arms) gave word salad for the 1.5B, an earlier smoke test without it
# was coherent, and the perplexity gates are prefill, where graphs never run. Greedy, 128 tokens, same prompt:
# default, GRAPH_OPT=1 twice (run-to-run determinism), graphs disabled, GRAPH_OPT=1 with packet capture off,
# Vulkan as the reference, on the 1.5B and the 8B. (2) Packet capture on and off with GRAPH_OPT=1 on all six
# models, the comparison the validation meant to make (its llama-cli runs had no -c and crashed on ROCm).
# Freezes the running throughput pass between two of its runs and resumes it at the end, whatever happens.
set -u
O=~/graphopt-check; mkdir -p "$O"
H=~/llama-master/build-hip-pkf16/bin; V=~/llama-master/build-vk-f44/bin; L=/opt/bc250-rocm/lib64
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
until c=$(ps -eo pid,args | awk '$2 == "bash" && $3 ~ /campaign_packet_capture\.sh$/ {print $1; exit}'); [ -n "$c" ]; do sleep 2; done
kill -STOP "$c"; trap 'kill -CONT "$c"; log "throughput pass resumed"' EXIT
while pgrep -x llama-bench >/dev/null; do sleep 2; done
log "throughput pass $c frozen between runs"
gen () {  # tag bindir model env...
  local tag=$1 b=$2 m=$3; shift 3
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 600 "$b/llama-cli" -m /opt/models/$m.gguf -ngl 99 -fa on -st \
      -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$O/${m}_$tag.raw" 2> "$O/${m}_$tag.err" < /dev/null
  local rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$O/${m}_$tag.raw" > "$O/${m}_$tag.txt"
  log "$m $tag (rc=$rc, $(wc -c < "$O/${m}_$tag.txt") bytes): $(tr '\n' ' ' < "$O/${m}_$tag.txt" | cut -c1-140)"
}
same () { if [ ! -s "$O/$1.txt" ] || [ ! -s "$O/$2.txt" ]; then echo EMPTY; elif cmp -s "$O/$1.txt" "$O/$2.txt"; then echo identical; else echo DIFFERENT; fi; }
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0; do
  gen default "$H" "$m" X=1
  gen graphopt "$H" "$m" GGML_CUDA_GRAPH_OPT=1
  gen graphopt2 "$H" "$m" GGML_CUDA_GRAPH_OPT=1
  gen nographs "$H" "$m" GGML_CUDA_DISABLE_GRAPHS=1
  gen graphopt_nopc "$H" "$m" GGML_CUDA_GRAPH_OPT=1 DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
  gen vulkan "$V" "$m" X=1
  log "$m: GRAPH_OPT vs default $(same ${m}_graphopt ${m}_default); GRAPH_OPT twice $(same ${m}_graphopt ${m}_graphopt2); default vs no graphs $(same ${m}_default ${m}_nographs); default vs Vulkan $(same ${m}_default ${m}_vulkan)"
done
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  [ -f "$O/${m}_graphopt.txt" ] || gen graphopt "$H" "$m" GGML_CUDA_GRAPH_OPT=1
  [ -f "$O/${m}_graphopt_nopc.txt" ] || gen graphopt_nopc "$H" "$m" GGML_CUDA_GRAPH_OPT=1 DEBUG_CLR_GRAPH_PACKET_CAPTURE=0
  log "generate $m: packet capture on vs off $(same ${m}_graphopt ${m}_graphopt_nopc)"
done
log done
