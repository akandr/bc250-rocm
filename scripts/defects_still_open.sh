#!/usr/bin/env bash
# Do the documented open defects still reproduce on the current configuration?
#
# Three things about this board changed after those defects were characterised:
# the navi12 SDMA microcode, kernel 7.1.8, and amdgpu.gpu_recovery=0. Any of them
# could in principle have moved something. A README that lists a defect as open
# when it has quietly stopped happening is as wrong as one that misses a defect,
# and nothing has re-checked them as a set.
#
# Also re-measures the SGEMM figures on the front page, which come from the same
# pre-change era.
set -u
exec 9>~/.defects.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv116; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== configuration: kernel $(uname -r), gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery)"

log "=== defect 1: the fp16 cuBLAS path should still be wrong, f32 should still be right"
for i in 1 2 3; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 30 1800 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f "$WIKI" --chunks 2 > "$D/f32_$i.log" 2>&1
  a=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/f32_$i.log" | grep -oE '[0-9.]+$')
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
    timeout -k 30 1800 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f "$WIKI" --chunks 2 > "$D/f16_$i.log" 2>&1
  b=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/f16_$i.log" | grep -oE '[0-9.]+$')
  log "  round $i: f32=${a:-FAIL} f16=${b:-FAIL}"
done

log "=== defect 2: SDMA above 16384 bytes, which the navi12 microcode fixed"
if [ -x ~/sdma_probe_bytes ]; then
  HSA_ENABLE_SDMA=1 timeout -k 10 120 ~/sdma_probe_bytes 20 > "$D/sdma.log" 2>&1
  log "  probe rc=$? : $(grep -aoE 'ALL SIZES COMPLETED|HUNG:.*' "$D/sdma.log" | tail -1)"
else
  log "  sdma_probe_bytes not present, skipped"
fi

log "=== front page SGEMM figures, same probe as the original"
if [ -x ~/rocblas_probe ]; then
  for n in 512 1024 2048 4096; do
    r=$(LD_LIBRARY_PATH=$L timeout -k 20 900 ~/rocblas_probe "$n" 2>&1 | grep -aoE '[0-9.]+ ms|[0-9.]+ GFLOP' | tr '\n' ' ')
    log "  N=$n: ${r:-no output}"
  done
else
  log "  rocblas_probe not present, listing candidates: $(ls ~ | grep -iE 'sgemm|rocblas' | tr '\n' ' ')"
fi

log "=== faults during this run: $(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE 'page fault \(src_id|GPU reset begin')"
touch "$D/DONE"; log done
