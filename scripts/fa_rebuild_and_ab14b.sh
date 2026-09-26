#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Apply the D=128 row changes (row 64 as chosen by the sweep, rows 16 and 8 as found), rebuild the
# four-patch build, let the package cool, then run the 14B A/B. One launch, nothing else on the board.
# Usage: rebuild_and_ab14b.sh "<row 64 spec ncols:nthreads:occ:nbatch_fa:nbatch_K>"
set -u
ROW64=${1:?row 64 spec}
M=~/llama-master; O=~/ab14b
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
cp /tmp/fattn-tile.cuh.patched5 ggml/src/ggml-cuda/fattn-tile.cuh
python3 ~/apply_rdna1_rows.py $M 32:512:2:32:128 16:512:2:32:128 --d128 "$ROW64" 16:512:3:32:64 8:256:4:32:64 | tee -a $O/log
git diff --stat ggml/src/ggml-cuda/fattn-tile.cuh | tail -1 | tee -a $O/log
# dispatcher: on RDNA1 the 64-column ncols2 = 1 tile spills where the 32-column one is clean, so
# keep ncols2 = 1 dispatch at 32 columns there (host code, gated on the compute-capability check)
python3 - <<'PY' | tee -a $O/log
p="ggml/src/ggml-cuda/fattn-tile.cuh"; s=open(p).read()
old = """#ifdef GGML_USE_HIP
    if constexpr (DKQ <= 128) {
        if (Q->ne[1] > 32/ncols2) {
            constexpr int cols_per_block = 64;"""
new = """#ifdef GGML_USE_HIP
    if constexpr (DKQ <= 128) {
        // On RDNA1 the 64-column tile with ncols2 == 1 spills registers while the 32-column tile does not,
        // so models whose head ratio has no power-of-two divisor stay at 32 columns there.
        const bool rdna1_no_gqa = GGML_CUDA_CC_IS_RDNA1(cc) && ncols2 == 1;
        if (!rdna1_no_gqa && Q->ne[1] > 32/ncols2) {
            constexpr int cols_per_block = 64;"""
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new, 1)); print("dispatcher: RDNA1 ncols2 == 1 capped at 32 columns")
PY
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.patched6
log "rebuild start"
nice -n 5 cmake --build build-hip-fatile -j 7 --target test-backend-ops test-export-graph-ops llama-bench llama-perplexity llama-cli > $O/rebuild.log 2>&1 \
  && log "rebuild ok" || { log "rebuild FAILED (see rebuild.log)"; exit 1; }
sleep 150   # cool the package after the build before measuring
log "sanity 1.5B pp2048 fa=1 new build: $(LD_LIBRARY_PATH=/opt/bc250-rocm/lib64 timeout -k 20 600 $M/build-hip-fatile/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 2048 -n 0 -r 3 2>/dev/null | grep -a pp2048 | awk -F'|' '{print $(NF-1)}' | tr -d ' ')"
L=/opt/bc250-rocm/lib64
# row 64 also serves the D=128 GQA models, so their gates and the 8B prefill are re-checked first
log "1.5B gate new build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/build-hip-fatile/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (patch-5 build read 8.9306, three-patch 8.9442, Vulkan 8.9734)"
log "8B pp2048 fa=1 new build: $(LD_LIBRARY_PATH=$L timeout -k 20 900 $M/build-hip-fatile/bin/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 2048 -n 0 -r 3 2>/dev/null | grep -a pp2048 | awk -F'|' '{print $(NF-1)}' | tr -d ' ')  (patch-5 build 256.3 to 257.6)"
log "8B gate new build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/build-hip-fatile/bin/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')  (patch-5 build 9.1248, three-patch 9.1117)"
BASE=build-hip-f44 NEW=build-hip-fatile OUT=$O ~/fa_ab14b.sh   # scripts/fa_ab14b.sh on the board
