#!/usr/bin/env bash
# SDMA works now. Is it worth enabling?
#
# Swapping the cyan_skillfish2 SDMA firmware for navi12 (GabriWar tip) makes
# SDMA complete at every size from 4 KiB to 2 GiB, and the full stack gates
# bit-identically with it enabled. That closes the defect. It does not settle
# whether the recipe should enable it: the blit path the workaround forced
# reaches 152 GB/s at 2 GiB from pinned memory where SDMA reaches 47.8, so for
# bulk transfer the workaround was faster. What SDMA buys is a separate engine,
# which should show up in model load time and possibly in inference.
#
# Arms alternated, because this board has already produced two opposite answers
# from blocked designs.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv78; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

bench () { # bench <tag> <sdma>
  local tag=$1 sdma=$2
  local t0=$(date +%s.%N)
  env HSA_ENABLE_SDMA=$sdma GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 900 $HIP/llama-bench -m $M8 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 > "$D/$tag.log" 2>&1
  local t1=$(date +%s.%N)
  log "  $tag sdma=$sdma  wall=$(echo "$t1 - $t0" | bc)s  pp512=$(grep -aoE "pp512 \| +[0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$")  tg64=$(grep -aoE "tg64 \| +[0-9.]+" "$D/$tag.log" | grep -oE "[0-9.]+$")"
}

log "=== throughput and load time, SDMA on against off, ABBA over three rounds"
for r in 1 2 3; do
  if [ $((r % 2)) -eq 1 ]; then o="1 0 0 1"; else o="0 1 1 0"; fi
  i=0
  for s in $o; do i=$((i+1)); bench "r${r}_${i}_sdma$s" "$s"; done
done

log "=== gate stability with SDMA enabled, three repeats (reference 8.9442)"
for i in 1 2 3; do
  v=$(env HSA_ENABLE_SDMA=1 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      timeout -k 30 900 $HIP/llama-perplexity -m $Q15 --no-mmap -ngl 99 -fa on -c 4096 -f $WIKI --chunks 8 2>&1 \
      | grep -aoE "Final estimate: PPL = [0-9.]+" | grep -oE "[0-9.]+$")
  log "  gate $i: ${v:-FAIL}"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
