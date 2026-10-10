#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The README tells a reader to apply logs/graphopt-correctness-2026-10-02/alloc-deps.diff on top of the thirteen
# patches. That combination was never built: the fix was tested on a copy of the campaign's working tree. Build
# it from the clean tree of rebuild_check.sh and compare greedy replies, default against the option with the fix,
# on the four models where the option launches streams.
set -u
export LC_ALL=C
O=${1:-~/graphopt-fix-check}; S=~/llama-fixcheck; D=${DIFF:-~/alloc-deps.diff}
mkdir -p "$O"
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
L=/opt/bc250-rocm/lib64
rm -rf "$S"; git -C ~/llama-rebuild worktree prune
git -C ~/llama-rebuild worktree add -q --detach "$S" HEAD || { log "worktree failed"; exit 1; }
git -C "$S" apply "$D" > "$O/apply.log" 2>&1 || { log "alloc-deps.diff does not apply: $(tail -2 "$O/apply.log")"; exit 1; }
log "thirteen patches ($(git -C "$S" log --oneline -1 | cut -c1-60)) + alloc-deps.diff applied"
cmake -S "$S" -B "$S/build-hip" -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1013 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
      -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/x86_64-redhat-linux/16/libgomp.so -DOpenMP_pthread_LIBRARY=/usr/lib64/libpthread.a \
      > "$O/configure.log" 2>&1 || { log "configure failed"; exit 1; }
t0=$(date +%s)
cmake --build "$S/build-hip" -j6 --target llama-cli > "$O/build.log" 2>&1 || { log "build failed"; exit 1; }
log "built in $(( $(date +%s) - t0 )) s"
was=$(systemctl is-active ollama); [ "$was" = active ] && sudo systemctl stop ollama
log "ollama was $was, stopped for the test"
mkdir -p "$O/replies"
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
gen () {  # tag model env...
  local tag=$1 m=$2; shift 2; local f=$O/replies/${m}_$tag
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 900 "$S/build-hip/bin/llama-cli" -m /opt/models/$m.gguf \
      -ngl 99 -fa on -st -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$f.raw" 2> "$f.err" < /dev/null
  local rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$f.raw" > "$f.txt"
  log "$m $tag: rc=$rc md5=$(md5sum < "$f.txt" | cut -c1-8) $(wc -c < "$f.txt")B$(grep -q 'no ROCm-capable device' "$f.err" && echo ' NO GPU')"
}
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3-14b deepseek-r1-14b; do
  gen default "$m" X=1
  gen fix1 "$m" GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1
  gen fix2 "$m" GGML_CUDA_GRAPH_OPT=1 GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1
  gen optonly "$m" GGML_CUDA_GRAPH_OPT=1
done
[ "$was" = active ] && sudo systemctl start ollama
log "ollama $(systemctl is-active ollama)"
log done
