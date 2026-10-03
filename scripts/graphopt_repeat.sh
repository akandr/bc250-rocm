#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does GGML_CUDA_GRAPH_OPT=1 corrupt decode on the dense qwen3 models? In graphopt_check.sh both qwen3-14b runs with
# GRAPH_OPT=1 and two of three qwen3-8b runs gave word salad, while the 1.5B, the MoE, the 27B and deepseek-r1-14b
# stayed coherent and qwen3-8b with graphs disabled was coherent. Repeat each configuration several times, services
# stopped: default (graphs on, no GRAPH_OPT), GRAPH_OPT=1, graphs disabled. Greedy, 128 tokens, -c 4096.
# Freezes the running throughput pass between two of its runs and resumes it at the end, whatever happens.
set -u
O=~/graphopt-repeat; mkdir -p "$O"
H=~/llama-master/build-hip-pkf16/bin; L=/opt/bc250-rocm/lib64
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
until c=$(ps -eo pid,args | awk '$2 == "bash" && $3 ~ /campaign_packet_capture\.sh$/ {print $1; exit}'); [ -n "$c" ]; do sleep 2; done
kill -STOP "$c"; trap 'kill -CONT "$c"; log "throughput pass resumed"' EXIT
while pgrep -x llama-bench >/dev/null; do sleep 2; done
log "throughput pass $c frozen between runs; services: $(systemctl is-active ollama hw-watcher.timer crond | tr "\n" " ")"
gen () {  # tag model env...
  local tag=$1 m=$2; shift 2
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 600 "$H/llama-cli" -m /opt/models/$m.gguf -ngl 99 -fa on -st \
      -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$O/${m}_$tag.raw" 2> "$O/${m}_$tag.err" < /dev/null
  local rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$O/${m}_$tag.raw" > "$O/${m}_$tag.txt"
  local gpu=$(grep -c "no ROCm-capable device" "$O/${m}_$tag.err")
  log "$m $tag (rc=$rc, $(wc -c < "$O/${m}_$tag.txt") bytes$( [ "$gpu" != 0 ] && echo ", NO GPU")): $(tr '\n' ' ' < "$O/${m}_$tag.txt" | cut -c1-120)"
}
for m in qwen3-14b qwen3-8b-q8_0; do
  for i in 1 2 3; do gen default$i "$m" X=1; done
  for i in 1 2 3; do gen graphopt$i "$m" GGML_CUDA_GRAPH_OPT=1; done
  for i in 1 2; do gen nographs$i "$m" GGML_CUDA_DISABLE_GRAPHS=1; done
  log "$m: md5 of each reply: $(cd "$O" && md5sum ${m}_*.txt | awk "{printf \"%s=%s \", substr(\$2, length(\"$m\")+2), substr(\$1,1,8)}")"
done
log done
