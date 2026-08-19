#!/usr/bin/env bash
# bench_fixed_stack.sh - full re-benchmark on the FIXED llama.cpp stack
# (master 7ba604f + integrated=false + kqv PREC_F32 + gfx1013 RDNA1 macro).
# Board: kernel 7.1.5, 40 CU, bc250_flush_pasid_kiq=0, bc250_flush_by_runlist=1.
# This is the harness that produced logs/bench-fixed-2026-08/.
# HIP phases run first; Vulkan phases wait for build-vk to finish.
set -u
D=~/bench-fixed-2026-08-12; mkdir -p $D
HIP=~/llama-master/build-hip/bin
VK=~/llama-master/build-vk/bin
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
Q8B=/opt/models/qwen3-8b-q8_0.gguf
Q9B=/opt/models/qwen3.5-9b.gguf
DS14=/opt/models/deepseek-r1-14b.gguf
Q14=/opt/models/qwen3-14b.gguf
MOE=/opt/models/qwen3.6-35b-a3b-iq2m.gguf
WIKI=~/wiki.test.raw
export HSA_ENABLE_SDMA=0

phase () { echo "=== [$(date +%H:%M:%S)] $1 ===" | tee -a $D/campaign.log; sync; }
run () { # run <tag> <timeout> <cmd...>
  local tag=$1 to=$2; shift 2
  timeout -k 20 "$to" "$@" > $D/$tag.log 2>&1
  echo "[$(date +%H:%M:%S)] $tag rc=$?" | tee -a $D/campaign.log; sync
}

# thermal logger (30 s cadence) for the whole campaign
( while true; do
    t=$(sensors 2>/dev/null | grep -m1 edge | grep -oE '[0-9]+\.[0-9]' | head -1)
    c=$(cat /sys/class/drm/card*/device/pp_dpm_sclk 2>/dev/null | grep '\*' | head -1)
    echo -e "$(date +%s)\t${t:-NA}\t${c:-NA}" >> $D/thermals.tsv
    sleep 30
  done ) & THERM=$!
trap 'kill $THERM 2>/dev/null' EXIT

phase "A qwen1.5b HIP bench: prefill + decode depth ladder (fa on)"
run a_q15_pp 2400 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512,2048,4096 -n 0 -r 3
run a_q15_tg 3600 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 0 -n 64 -d 0,4096,8192,16384,24576,30720 -r 2
run a_q15_faoff 1200 $HIP/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa off -p 512 -n 64 -r 3

phase "B qwen1.5b HIP perplexity gates"
run b_ppl_faon 3600 $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f $WIKI --chunks 8
run b_ppl_faoff 3600 $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa off -c 4096 -f $WIKI --chunks 8

phase "C large models HIP bench (fa on, pp512 + tg64 at d0/d4096)"
run c_8b   3600 $HIP/llama-bench -m $Q8B  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run c_9b   3600 $HIP/llama-bench -m $Q9B  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run c_ds14 4800 $HIP/llama-bench -m $DS14 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run c_q14  4800 $HIP/llama-bench -m $Q14  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run c_moe  4800 $HIP/llama-bench -m $MOE  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2

phase "D large models HIP perplexity gates (chunks 2, ctx 2048, fa on)"
run d_ppl_8b   3600 $HIP/llama-perplexity -m $Q8B  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run d_ppl_9b   3600 $HIP/llama-perplexity -m $Q9B  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run d_ppl_ds14 3600 $HIP/llama-perplexity -m $DS14 --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run d_ppl_q14  3600 $HIP/llama-perplexity -m $Q14  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run d_ppl_moe  3600 $HIP/llama-perplexity -m $MOE  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2

phase "E wait for Vulkan build"
for i in $(seq 1 120); do
  [ -x $VK/llama-bench ] && [ -x $VK/llama-perplexity ] && break
  sleep 30
done
if [ ! -x $VK/llama-bench ]; then
  echo "VULKAN BUILD MISSING - skipping Vulkan phases" | tee -a $D/campaign.log
else
phase "F Vulkan bench, same build, same boot"
run f_q15_pp 2400 $VK/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512,2048,4096 -n 0 -r 3
run f_q15_tg 3600 $VK/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 0 -n 64 -d 0,4096,8192,16384,24576,30720 -r 2
run f_8b   3600 $VK/llama-bench -m $Q8B  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run f_9b   3600 $VK/llama-bench -m $Q9B  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run f_ds14 4800 $VK/llama-bench -m $DS14 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run f_q14  4800 $VK/llama-bench -m $Q14  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2
run f_moe  4800 $VK/llama-bench -m $MOE  -mmp 0 -ngl 99 -fa on -p 512 -n 64 -d 0,4096 -r 2

phase "G Vulkan perplexity anchors"
run g_ppl_q15  3600 $VK/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f $WIKI --chunks 8
run g_ppl_8b   3600 $VK/llama-perplexity -m $Q8B  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run g_ppl_9b   3600 $VK/llama-perplexity -m $Q9B  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run g_ppl_ds14 3600 $VK/llama-perplexity -m $DS14 --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run g_ppl_q14  3600 $VK/llama-perplexity -m $Q14  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
run g_ppl_moe  3600 $VK/llama-perplexity -m $MOE  --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2
fi

phase "CAMPAIGN DONE"
grep -H "Final estimate" $D/*_ppl_*.log $D/b_ppl_*.log 2>/dev/null | tee -a $D/campaign.log
echo done > $D/DONE
