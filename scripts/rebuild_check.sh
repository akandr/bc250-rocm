#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Does the published recipe rebuild the build every figure here was measured on? Clones llama.cpp at 7ba604f
# from the board's own tree, applies the thirteen patches of patches/llamacpp/ with git am, builds as README
# step 7 says, and compares the result with build-hip-pkf16 (21 September, built from the working tree of
# ~/llama-master): the GGML options of both CMake caches, the two source trees file by file, 128 greedy tokens
# from every benchmark model on both builds, hashed, and prefill and decode speed with the builds alternated.
set -u
export LC_ALL=C
O=${1:-~/rebuild-check}; S=~/llama-rebuild; P=${PATCHES:-~/repro-patches}
mkdir -p "$O"
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log"; sync; }
L=/opt/bc250-rocm/lib64; C=~/llama-master/build-hip-pkf16/bin; R=$S/build-hip/bin
MODELS="qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs"
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }

log "clone 7ba604f from ~/llama-master and apply $(ls "$P"/*.patch | wc -l) patches with git am"
rm -rf "$S"
git clone -q --no-checkout ~/llama-master "$S" && git -C "$S" checkout -q 7ba604f || { log "clone failed"; exit 1; }
git -C "$S" -c user.name=bc250 -c user.email=bc250@localhost am "$P"/*.patch > "$O/am.log" 2>&1 ||
  { log "git am stopped: $(grep -m2 -E 'error|Patch failed' "$O/am.log" | tr '\n' ' ')"; exit 1; }
git -C "$S" log --reverse --format='%h %s' 7ba604f..HEAD > "$O/commits.txt"
log "applied: $(grep -c '' "$O/commits.txt") commits"

# the clean tree against ~/llama-master's working tree today, which build-hip-pkf16 was built from
diff -rqN -x .git -x 'build*' -x __pycache__ "$S" ~/llama-master | sed "s|$HOME/||g" > "$O/tree-diff-summary.txt"
diff -ruN -x .git -x 'build*' -x __pycache__ "$S" ~/llama-master | sed "s|$HOME/||g" > "$O/tree-diff.patch"
log "files differing from ~/llama-master's working tree: $(grep -c '' "$O/tree-diff-summary.txt")"

log "configure and build as README step 7 says (llama-cli and llama-bench)"
cmake -S "$S" -B "$S/build-hip" -DGGML_HIP=ON -DGGML_HIP_NO_VMM=ON -DAMDGPU_TARGETS=gfx1013 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_COMPILER=/usr/lib64/rocm/llvm/bin/clang++ \
      -DOpenMP_gomp_LIBRARY=/usr/lib/gcc/x86_64-redhat-linux/16/libgomp.so -DOpenMP_pthread_LIBRARY=/usr/lib64/libpthread.a \
      > "$O/configure.log" 2>&1 || { log "configure failed"; exit 1; }
diff <(grep -E '^GGML_[A-Z0-9_]+:' ~/llama-master/build-hip-pkf16/CMakeCache.txt | sort) \
     <(grep -E '^GGML_[A-Z0-9_]+:' "$S/build-hip/CMakeCache.txt" | sort) > "$O/ggml-options-diff.txt"
log "GGML options differing from build-hip-pkf16: $(grep -c '^[<>]' "$O/ggml-options-diff.txt")"
t0=$(date +%s)
cmake --build "$S/build-hip" -j6 --target llama-cli llama-bench > "$O/build.log" 2>&1 ||
  { log "build failed: $(grep -m3 error "$O/build.log" | cut -c1-200)"; exit 1; }
log "built in $(( $(date +%s) - t0 )) s"

mkdir -p "$O/replies"
PROMPT="The BC-250 is a mining board built around an AMD APU. Explain in plain words what a GPU compute queue is and why a driver bug in it matters:"
gen () {  # tag bin-dir model
  local tag=$1 H=$2 m=$3 f
  f=$O/replies/${m}_$tag
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 900 "$H/llama-cli" -m /opt/models/$m.gguf -ngl 99 -fa on -st \
      -c 4096 --temp 0 -s 1 -n 128 -p "$PROMPT" > "$f.raw" 2> "$f.err" < /dev/null
  local rc=$?
  awk '/^> /{on=1; next} /^\[ Prompt:/{on=0} on' "$f.raw" > "$f.txt"
  local nodev=$(grep -c "no ROCm-capable device" "$f.err")
  log "$m $tag: rc=$rc md5=$(md5sum < "$f.txt" | cut -c1-8) $(wc -c < "$f.txt")B$( [ "$nodev" != 0 ] && echo ' NO GPU') | $(tr '\n' ' ' < "$f.txt" | cut -c1-70)"
}
log "greedy replies, every other GPU user stopped: $(systemctl is-active ollama hw-watcher.timer crond | tr '\n' ' ')"
for m in $MODELS; do
  gen campaign "$C" "$m"; gen rebuild "$R" "$m"; gen campaign-again "$C" "$m"
done

# speed: prefill and decode in separate invocations, three rounds of three repetitions, the order of the
# builds swapped every round; samples from llama-bench's JSON output
J=$O/bench.jsonl; : > "$J"
bench () {  # tag bin model args...
  local tag=$1 b=$2 m=$3; shift 3
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L timeout -k 30 1800 "$b/llama-bench" -m /opt/models/$m.gguf \
      -mmp 0 -ngl 99 -fa on -r 3 "$@" -o jsonl 2>/dev/null | sed "s/^{/{\"arm\": \"$tag\", /" >> "$J"
}
log "speed, edge $(temp)"
for m in $MODELS; do
  for round in 1 2 3; do
    if [ $((round % 2)) -eq 1 ]; then order="campaign rebuild"; else order="rebuild campaign"; fi
    for tag in $order; do
      b=$C; [ "$tag" = rebuild ] && b=$R
      bench "$tag" "$b" "$m" -p 512 -n 0
      bench "$tag" "$b" "$m" -p 0 -n 64
    done
  done
  log "$m speed done, edge $(temp)"
done
python3 - "$J" <<'PY' | tee -a "$O/log"
import json, sys, statistics as st
rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    test = f"pp{r['n_prompt']}" if r['n_prompt'] else f"tg{r['n_gen']}"
    m = r['model_filename'].rsplit('/', 1)[-1].removesuffix('.gguf')
    rows.setdefault((m, test), {}).setdefault(r['arm'], []).extend(r['samples_ts'])
print("model test campaign rebuild rebuild/campaign samples")
for (m, test), a in rows.items():
    c, b = st.median(a.get('campaign', [0])), st.median(a.get('rebuild', [0]))
    print(f"{m} {test} {c:.2f} {b:.2f} {b / c if c else 0:.4f} {len(a.get('campaign', []))}/{len(a.get('rebuild', []))}")
PY
log done
