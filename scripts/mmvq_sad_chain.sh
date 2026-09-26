#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment B: apply the v_sad_u8 activation sums to the tree (on top of whatever mmvq.cu state the
# tree holds), build build-hip-sad, cool, then the decode A/B of the previous build against it, plus the
# 1.5B gate and the 8B gate, which should be bit-identical to the previous build's since the change is
# exact integer arithmetic.
# Usage: sad_chain.sh <previous build dir name>   (build-hip-mmvq or build-hip-fatile)
set -u
PREV=${1:?previous build}
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/sad-ab
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
cp ggml/src/ggml-cuda/vecdotq.cuh /tmp/vecdotq.cuh.orig
python3 ~/apply_rdna1_sad.py $M/ggml/src/ggml-cuda | tee -a $O/log
git diff --stat ggml/src/ggml-cuda/vecdotq.cuh ggml/src/ggml-cuda/mmvq.cu | tee -a $O/log
cmake -S . -B build-hip-sad -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1 && log "configured" || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build build-hip-sad -j 7 --target test-backend-ops llama-bench llama-perplexity llama-cli > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; exit 1; }
sleep 150
BASE=$PREV NEW=build-hip-sad OUT=$O ~/mmvq_ab.sh
log "8B gate build-hip-sad: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/build-hip-sad/bin/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "8B gate $PREV: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$PREV/bin/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "1.5B gate $PREV: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$PREV/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "CHAIN DONE"
