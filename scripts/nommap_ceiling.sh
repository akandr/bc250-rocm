#!/usr/bin/env bash
# Does loading without mmap lower the usable context, and if so why?
#
# A decode-variance experiment used --load-mode none for two of three arms, and
# every run of those arms aborted while the mmap arm succeeded. This walks the
# depth ladder in both load modes to find where the boundary sits, then samples
# memory through one run of each mode at the failing depth, so the explanation
# is measured rather than assumed.
#
# Fault counts here would come from dmesg, which sees only the current boot; see
# scripts/fault_count.sh. This harness does not count faults, since the failure
# is a host-side abort with nothing logged by the kernel.
set -u
D=~/inv92; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
E=(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L)
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== depth ladder, load mode none"
"${E[@]}" timeout -k 20 900 $HIP/llama-bench -m $M8 -lm none -ngl 99 -fa 1 -p 0 -n 8 -r 1 \
  > "$D/lm_none_nodepth.log" 2>&1
log "  no depth: $(grep -aoE 'tg8 \| +[0-9.]+' "$D/lm_none_nodepth.log" | tail -1)"
for d in 8192 12288 14336 16128; do
  "${E[@]}" timeout -k 20 900 $HIP/llama-bench -m $M8 -lm none -ngl 99 -fa 1 -p 0 -n 8 -d $d -r 1 \
    > "$D/lm_none_d$d.log" 2>&1
  log "  depth $d: rc=$? $(grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" "$D/lm_none_d$d.log" | tail -1)"
done

log "=== control: the same depth under mmap"
"${E[@]}" timeout -k 20 900 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
  > "$D/lm_mmap_d16128.log" 2>&1
log "  mmap 16128: $(grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" "$D/lm_mmap_d16128.log" | tail -1)"

# Is it memory? Sample free through one run of each mode at the failing depth.
sample () { # sample <mode> <outfile>
  local mode=$1 out=$2 samp
  ( while true; do echo "$(date +%s) $(free -m | awk '/^Mem:/{print $3, $7}')"; sleep 2; done > "$out" ) &
  samp=$!
  "${E[@]}" timeout -k 20 900 $HIP/llama-bench -m $M8 -lm "$mode" -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/rerun_$mode.log" 2>&1
  local rc=$?
  kill $samp 2>/dev/null
  log "  $mode: rc=$rc $(awk '{if($2>mu)mu=$2; if(min==0||$3<min)min=$3} END{print "peak used", mu, "MiB; min available", min, "MiB"}' "$out")"
}
log "=== memory through the failure"
sample none "$D/mem_none_16128.txt"
sample mmap "$D/mem_mmap_16128.txt"
touch "$D/DONE"; log done
