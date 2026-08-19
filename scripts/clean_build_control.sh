#!/usr/bin/env bash
# Is the 8B decode drop a real regression, or the instrumentation in the tree?
#
# The campaign of 2026-08-12 recorded 39.20 t/s for 8B decode; the same
# measurement now gives 35.4 to 36.3, while prefill is unchanged (241.0 then,
# 242.4 now) and Vulkan on the same board is unchanged. A decode-only change
# with prefill flat points at per-token overhead rather than at the board.
#
# The working tree has since acquired debug instrumentation for the fp16 and
# allocation-reuse investigations, including getenv() calls inside the CUDA
# memory pool alloc and free paths. Those run per pool operation, which decode
# does per token and prefill barely does. This builds a clean tree carrying only
# the three shipped patches and measures the same thing.
set -u
D=~/inv68; mkdir -p "$D"
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
M8=/opt/models/qwen3-8b-q8_0.gguf
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
CLEAN=~/llama-clean
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the previous job"
while [ ! -f ~/inv67/DONE ]; do sleep 60; done

log "=== preparing a clean tree at the campaign commit"
rm -rf "$CLEAN"
git -C ~/llama-master worktree prune 2>/dev/null
git clone --shared --no-checkout ~/llama-master "$CLEAN" > "$D/clone.log" 2>&1
git -C "$CLEAN" checkout 7ba604f >> "$D/clone.log" 2>&1
log "  HEAD: $(git -C "$CLEAN" rev-parse --short HEAD)  dirty: $(git -C "$CLEAN" status --porcelain | wc -l)"

log "=== applying only the three shipped patches"
cd "$CLEAN"
sed -i "s/info.devices\[id\].integrated = prop.integrated;/info.devices[id].integrated = false;/" ggml/src/ggml-cuda/ggml-cuda.cu
log "  integrated=false: $(grep -c "integrated = false" ggml/src/ggml-cuda/ggml-cuda.cu)"
sed -i "s/#if defined(__gfx1010__) || defined(__gfx1012__)$/#if defined(__gfx1010__) || defined(__gfx1012__) || defined(__gfx1013__)/" ggml/src/ggml-cuda/vendors/hip.h
log "  rdna1 macro: $(grep -c "gfx1013__)" ggml/src/ggml-cuda/vendors/hip.h)"
perl -0pi -e "s/(ggml_tensor \* kqv = ggml_mul_mat\(ctx0, v, kq\);\n        cb\(kqv, \"kqv\", il\);\n)/\$1        ggml_mul_mat_set_prec(kqv, GGML_PREC_F32);\n/" src/llama-graph.cpp
log "  kqv precision: $(grep -c "set_prec(kqv" src/llama-graph.cpp)"

log "=== building (this takes a while)"
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_FLAGS="-O3" > "$D/cmake.log" 2>&1
log "  configure rc=$?"
cmake --build build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? errors=$(grep -c "error:" "$D/build.log")"

if [ ! -x "$CLEAN/build-hip/bin/llama-bench" ]; then log "BUILD FAILED, stopping"; touch "$D/DONE"; exit 1; fi

log "=== clean build: the campaign measurement (2026-08-12 gave pp512 241.01, tg64 39.20)"
for i in 1 2; do
  "${E[@]}" timeout -k 20 1800 "$CLEAN/build-hip/bin/llama-bench" -m $M8 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/clean_8b_$i.log" 2>&1
  log "  clean 8B pp512 $(grep -aoE "pp512 \| +[0-9.]+" "$D/clean_8b_$i.log" | grep -oE "[0-9.]+$")  tg64 $(grep -aoE "tg64 \| +[0-9.]+" "$D/clean_8b_$i.log" | grep -oE "[0-9.]+$")"
done
log "=== instrumented build, same measurement, same boot"
for i in 1 2; do
  "${E[@]}" timeout -k 20 1800 ~/llama-master/build-hip/bin/llama-bench -m $M8 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/instr_8b_$i.log" 2>&1
  log "  instr 8B pp512 $(grep -aoE "pp512 \| +[0-9.]+" "$D/instr_8b_$i.log" | grep -oE "[0-9.]+$")  tg64 $(grep -aoE "tg64 \| +[0-9.]+" "$D/instr_8b_$i.log" | grep -oE "[0-9.]+$")"
done
log "=== small model both ways (campaign: pp512 806, tg64 113.5)"
"${E[@]}" timeout -k 20 1800 "$CLEAN/build-hip/bin/llama-bench" -m $Q15 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/clean_q15.log" 2>&1
log "  clean q15 pp512 $(grep -aoE "pp512 \| +[0-9.]+" "$D/clean_q15.log" | grep -oE "[0-9.]+$")  tg64 $(grep -aoE "tg64 \| +[0-9.]+" "$D/clean_q15.log" | grep -oE "[0-9.]+$")"
"${E[@]}" timeout -k 20 1800 ~/llama-master/build-hip/bin/llama-bench -m $Q15 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 3 > "$D/instr_q15.log" 2>&1
log "  instr q15 pp512 $(grep -aoE "pp512 \| +[0-9.]+" "$D/instr_q15.log" | grep -oE "[0-9.]+$")  tg64 $(grep -aoE "tg64 \| +[0-9.]+" "$D/instr_q15.log" | grep -oE "[0-9.]+$")"
touch "$D/DONE"; log done
