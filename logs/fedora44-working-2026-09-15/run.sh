#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Fedora 44 + ROCm 7.1.1, production 7.1.8 kernel. Native gfx1013 rocBLAS 7.1.1 (rb711) and comgr with
# corrected gfx10 VGPR totals (comgr-fixed), each switched in through LD_LIBRARY_PATH.
export PATH=/usr/bin:/usr/sbin HSA_ENABLE_SDMA=0
H=~/llama-master/build-hip-f44/bin; D=~/s0915/f44b/log; M=/opt/models
RB=$HOME/rb711/install/lib64; CG=$HOME/rb711/comgr-fixed
both="LD_LIBRARY_PATH=$CG:$RB"
ppl() { label=$1 model=$2 c=$3 ch=$4 fa=$5; shift 5
  env "$@" timeout -k 30 1800 $H/llama-perplexity -m $model --no-mmap -ngl 99 -fa $fa -c $c -f ~/wiki.test.raw --chunks $ch > $D/$label.log 2>&1
  echo "$label: rc=$? $(grep -aoE "Final estimate: PPL = [0-9.]+" $D/$label.log) $(grep -aE "GGML_ASSERT|ROCm error: " $D/$label.log | head -1 | cut -c1-90)"; }
echo "== which libraries a run maps"
env $both AMD_LOG_LEVEL=4 timeout 120 $H/llama-perplexity -m $M/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 512 -f ~/wiki.test.raw --chunks 1 > $D/maps_probe.log 2>&1 &
pid=$!; sleep 25; for p in $(/usr/bin/pgrep -f "llama-perplexity -m $M/qwen2.5-1.5b"); do grep -oE "/[^ ]*(librocblas|libamd_comgr)[^ ]*" /proc/$p/maps | sort -u; done; wait $pid
grep -aoE "totalNumVGPRs=[0-9]+|vgprsPerSimd[^,]*" $D/maps_probe.log | sort | uniq -c | head -3
echo "== comgr unpatched, native rocBLAS (expect the flash-attention assert)"
ppl gate_rbonly $M/qwen2.5-1.5b-q4km.gguf 4096 8 on LD_LIBRARY_PATH=$RB
echo "== both fixes"
ppl gate $M/qwen2.5-1.5b-q4km.gguf 4096 8 on $both
ppl q8_default_1 $M/qwen3-8b-q8_0.gguf 2048 2 on $both
ppl q8_default_2 $M/qwen3-8b-q8_0.gguf 2048 2 on $both
ppl q8_f16_tcache1 $M/qwen3-8b-q8_0.gguf 2048 2 on $both GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 GLIBC_TUNABLES=glibc.malloc.tcache_count=1
ppl q8_f32 $M/qwen3-8b-q8_0.gguf 2048 2 on $both GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
ppl q14_default $M/qwen3-14b.gguf 2048 2 on $both
ppl q15_faoff_1024 $M/qwen2.5-1.5b-q4km.gguf 1024 2 off $both GGML_CUDA_NO_VMM=1
echo "== throughput"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0; do
  env $both timeout -k 20 1200 $H/llama-bench -m $M/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > $D/bench_$m.log 2>&1
  echo "$m: $(grep -aE "pp512|tg64" $D/bench_$m.log | awk -F"|" "{print \$(NF-1)}" | tr -s " " | tr "\n" " ")"
done
echo "== test-backend-ops"
env $both timeout 5400 $H/test-backend-ops -b ROCm0 > $D/tbo_full.log 2>&1; rc=$?
echo "rc=$rc ok=$(sed "s/\x1b\[[0-9;]*m//g" $D/tbo_full.log | grep -acE ": OK") fail=$(sed "s/\x1b\[[0-9;]*m//g" $D/tbo_full.log | grep -a FAIL | grep -vc "Backend ROCm0") $(sed "s/\x1b\[[0-9;]*m//g" $D/tbo_full.log | grep -aE "backends passed")"
echo DONE
