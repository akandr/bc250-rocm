#!/usr/bin/env bash
# Close three gaps named while auditing the investigation.
#
# 1. The documented mechanism, "the first fp16 cuBLAS call of each graph returns
#    exactly zero", has not been re-verified on the current stack. Everything
#    lately is perplexity, which is far downstream. BC250_TEMP already sums the
#    GEMM output before the f16 to f32 conversion, so the claim can be checked
#    directly rather than assumed still true.
#
# 2. I asserted that the native rocBLAS cannot serve a gfx1010 override, because
#    it carries only gfx1013 code objects, without testing it. Assertions of that
#    shape have been wrong here before.
#
# 3. Decode variance lives in per-process setup. CPU pinning, memory state and
#    queue placement are eliminated. Address space randomisation is per-process
#    by definition and has never been varied.
set -u
exec 9>~/.threegaps.lock; flock -n 9 || { echo "locked"; exit 1; }
D=~/inv122; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
MB=~/llama-master/build-multi/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== 1. does the first fp16 GEMM of each graph still return zero?"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_TEMP=1 LD_LIBRARY_PATH=$L \
  timeout -k 30 900 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
  -f ~/wiki.test.raw --chunks 1 > "$D/temp_f16.log" 2>&1
tot=$(grep -ac "BC250TEMP" "$D/temp_f16.log")
zer=$(grep -a "BC250TEMP" "$D/temp_f16.log" | grep -c "abs_sum=0 ")
log "  fp16: $tot GEMM outputs recorded, $zer of them exactly zero"
log "  first six: $(grep -a 'BC250TEMP' "$D/temp_f16.log" | head -6 | grep -oE 'abs_sum=[0-9.e+-]+' | tr '\n' ' ')"
log "  positions of the zeros: $(grep -a 'BC250TEMP' "$D/temp_f16.log" | grep -n 'abs_sum=0 ' | cut -d: -f1 | head -12 | tr '\n' ' ')"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 BC250_TEMP=1 LD_LIBRARY_PATH=$L \
  timeout -k 30 900 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
  -f ~/wiki.test.raw --chunks 1 > "$D/temp_f32.log" 2>&1
log "  f32 control: $(grep -ac 'BC250TEMP' "$D/temp_f32.log") recorded, $(grep -a 'BC250TEMP' "$D/temp_f32.log" | grep -c 'abs_sum=0 ') zero"

log "=== 2. can the native rocBLAS serve a gfx1010 override?"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 HSA_OVERRIDE_GFX_VERSION=10.1.0 LD_LIBRARY_PATH=$L \
  timeout -k 30 900 "$MB/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
  -f ~/wiki.test.raw --chunks 2 > "$D/native_override.log" 2>&1
r=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/native_override.log" | grep -oE '[0-9.]+$')
log "  native library + gfx1010 override: ${r:-FAILED}"
[ -z "$r" ] && log "    reason: $(grep -aiE 'error|abort|no kernel|not found' "$D/native_override.log" | head -1 | cut -c1-100)"

log "=== 3. decode variance with address space randomisation disabled"
for i in 1 2 3 4 5 6; do
  if [ $((i % 2)) -eq 1 ]; then
    v=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
        setarch -R timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
        | grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" | grep -oE "[0-9.]+$"); arm="ASLR-off"
  else
    v=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
        timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
        | grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" | grep -oE "[0-9.]+$"); arm="ASLR-on"
  fi
  log "  run $i $arm: ${v:-FAIL}"
done
touch "$D/DONE"; log done
