#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Why does GGML_CUDA_GRAPH_OPT=1 corrupt greedy decode on the dense qwen3 models (qwen3-8b-q8_0, qwen3-14b), while
# the default and graphs-disabled runs are byte-identical and coherent (~/graphopt-repeat)? Hypothesis: inside a
# concurrent Q/K/V region the graph allocator recycles a tensor that the other streams still read from outside the
# region (the fork node's output, inp_pos, ...) for a tensor one stream writes, and is_valid() never compares those.
#
# Freezes phase C of the ceiling campaign in the 20 s pause after a run (no run in progress, so no recorded duration
# includes the freeze), builds a copy of the campaign's source tree with ggml-org/llama.cpp#27301 (allocation
# dependencies from graph_optimize) and graphopt_instrument.py (GGML_CUDA_GRAPH_OPT_DEBUG=1 reports the overlaps,
# GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1 keeps every tensor of a region allocated until its join), runs greedy generations
# and decode benchmarks, and resumes phase C whatever happens. The campaign's own build is not touched.
set -u
O=~/graphopt-mechanism; mkdir -p "$O"
SRC=~/llama-graphopt; B=$SRC/build-hip-pkf16; H=$B/bin; L=/opt/bc250-rocm/lib64
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
note () { echo "[$(date '+%F %T')] NOTE: $*" >> ~/ceilings-nosnap/log; sync; }

bp=$(ps -eo pid=,args= | awk '$2 == "bash" && $3 ~ /ceilings_bisect\.sh$/ {print $1; exit}')
[ -n "$bp" ] || { log "no ceilings_bisect.sh running, nothing to freeze"; exit 1; }
log "waiting for phase C (bisect $bp) to reach the 20 s pause after a run"
until ps -eo ppid=,args= | awk -v p="$bp" '$1 == p && $2 == "sleep" && $3 == "20" {f = 1} END {exit !f}'; do
  kill -0 "$bp" 2>/dev/null || { log "phase C ended before a pause was reached; nothing frozen"; exit 1; }
  sleep 0.5
done
kill -STOP "$bp"
# never leave phase C frozen: this script and a watchdog that outlives it (even a SIGKILL) both resume it
me=$$
sudo -n sh -c "echo -1000 > /proc/$me/oom_score_adj"
( trap '' HUP; while kill -0 "$me" 2>/dev/null; do sleep 10; done; kill -CONT "$bp" 2>/dev/null ) &
sudo -n sh -c "echo -1000 > /proc/$!/oom_score_adj"
resume () {
  sync; sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches'
  kill -CONT "$bp"; note "phase C resumed after the GRAPH_OPT mechanism check"; log "phase C resumed"
}
trap resume EXIT
note "phase C frozen in the pause after a run, for the GRAPH_OPT mechanism check (~/graphopt-mechanism); no run was in progress, so no recorded duration includes the freeze"
log "phase C frozen; last result: $(grep -v NOTE ~/ceilings-nosnap/log | tail -1 | cut -c1-150)"

# --- build -----------------------------------------------------------------------------------------------------------
log "copy the source tree (no build directories, no .git)"
rsync -a --delete --exclude='/build*' --exclude='/.git' ~/llama-master/ "$SRC/" || { log "rsync failed"; exit 1; }
cd "$SRC" || exit 1
[ -d "$SRC/.git" ] && rm -rf "$SRC/.git"
git init -q && git add -A &&
  git -c user.name=bc250 -c user.email=bc250@localhost commit -qm "llama-master working tree of build-hip-pkf16" || exit 1
git apply "$O/pr27301.diff" || { log "ggml-org/llama.cpp#27301 does not apply"; exit 1; }
python3 "$O/graphopt_instrument.py" ggml/src/ggml-cuda/ggml-cuda.cu >> "$O/log" || { log "instrumentation does not apply"; exit 1; }
git diff > "$O/source.diff"; log "patched: $(git diff --stat | tail -1)"
cmake -S "$SRC" -B "$B" -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON \
      -DGGML_HIP_GRAPHS=ON > "$O/configure.log" 2>&1 || { log "configure failed, see configure.log"; exit 1; }
diff <(grep -E '^GGML_[A-Z0-9_]+:' ~/llama-master/build-hip-pkf16/CMakeCache.txt | sort) \
     <(grep -E '^GGML_[A-Z0-9_]+:' "$B/CMakeCache.txt" | sort) > "$O/cache-vs-pkf16.diff"
log "GGML options differing from build-hip-pkf16: $(grep -c '^[<>]' "$O/cache-vs-pkf16.diff")"
t0=$(date +%s)
log "build llama-cli and llama-bench, -j 7"
timeout 7200 cmake --build "$B" -j 7 --target llama-cli llama-bench > "$O/build.log" 2>&1 ||
  { log "build failed: $(grep -m3 -E 'error' "$O/build.log" | cut -c1-200)"; exit 1; }
log "built in $(( $(date +%s) - t0 )) s"

# --- greedy generations ----------------------------------------------------------------------------------------------
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
gen () {  # tag model env...
  local tag=$1 m=$2 rc; shift 2
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 600 "$H/llama-cli" -m /opt/models/$m.gguf -ngl 99 -fa on -st \
      -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$O/${m}_$tag.raw" 2> "$O/${m}_$tag.err" < /dev/null
  rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$O/${m}_$tag.raw" > "$O/${m}_$tag.txt"
  log "$m $tag: rc=$rc md5=$(md5sum < "$O/${m}_$tag.txt" | cut -c1-8) $(wc -c < "$O/${m}_$tag.txt")B" \
      "overlaps=$(grep -c '^graphopt-recycle' "$O/${m}_$tag.err")" \
      "$(grep -m1 -E '^graphopt-(debug|allocdeps)' "$O/${m}_$tag.err" | cut -c1-110)" \
      "| $(tr '\n' ' ' < "$O/${m}_$tag.txt" | cut -c1-70)"
  # a default run has to work; if it does not, the GPU or the build is broken and phase C should get the board back
  case $tag in default*) [ "$rc" = 0 ] || { log "a default run failed, stopping the check"; exit 1; } ;; esac
}
G=GGML_CUDA_GRAPH_OPT=1; D=GGML_CUDA_GRAPH_OPT_DEBUG=1; A=GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1
log "expected default md5 from the campaign build: qwen3-8b-q8_0 90b056d9, qwen3-14b 679a50d9"
m=qwen3-8b-q8_0
gen default    $m X=1
gen graphopt1  $m $G $D
gen graphopt2  $m $G $D
gen allocdeps1 $m $G $A $D
gen allocdeps2 $m $G $A
gen allocdeps3 $m $G $A
m=qwen3-14b
gen default    $m X=1
gen graphopt1  $m $G $D
gen allocdeps1 $m $G $A $D
gen allocdeps2 $m $G $A
gen allocdeps3 $m $G $A
m=qwen2.5-1.5b-q4km
gen default    $m X=1
gen graphopt1  $m $G $D
gen allocdeps1 $m $G $A $D
m=deepseek-r1-14b
gen default1   $m X=1
gen default2   $m X=1
gen graphopt1  $m $G $D
gen graphopt2  $m $G $D
gen allocdeps1 $m $G $A $D
gen allocdeps2 $m $G $A

# --- does the fix keep the speed? same flags as the throughput passes, decode only (graphs only run for decode) ------
bench () {  # tag model env...
  local tag=$1 m=$2; shift 2
  env "$@" HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 900 "$H/llama-bench" -m /opt/models/$m.gguf -mmp 0 -ngl 99 \
      -fa on -p 0 -n 64 -r 3 -o jsonl > "$O/bench_${m}_$tag.jsonl" 2> "$O/bench_${m}_$tag.err" < /dev/null
  log "bench $m $tag: tg64=[$(grep -oE '"samples_ts": \[[^]]*\]' "$O/bench_${m}_$tag.jsonl" | tr -d '"samples_ts:[] ')]"
}
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b; do
  for r in 1 2; do
    bench default$r   $m X=1
    bench graphopt$r  $m $G
    bench allocdeps$r $m $G $A
  done
done
log done
