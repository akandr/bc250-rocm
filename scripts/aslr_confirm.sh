#!/usr/bin/env bash
# Confirmation: does disabling address space randomisation reduce decode spread?
#
# A first pass at three runs per arm gave spreads of 0.37 with ASLR off and 2.74
# with it on, a variance ratio near 60. That is not a result. Two effects of
# similar size have already evaporated on this board when resampled: a
# graph-capture comparison that reversed sign, and a memory-compaction effect
# that went from a variance ratio of 7.32 at six runs per arm to 1.16 at twelve.
#
# Prediction, written before these runs: if the effect is real, the ASLR-off arm
# will again show a materially smaller spread than the ASLR-on arm across eight
# runs each. If it is the same artefact as before, the two spreads will converge.
set -u
exec 9>~/.aslr.lock; flock -n 9 || { echo "locked"; exit 1; }
D=~/inv123; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

bench () { # bench <aslr-off?>
  if [ "$1" = off ]; then
    env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L setarch -R \
      timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1
  else
    env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
      timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1
  fi
}

log "=== eight rounds, arms alternated ABBA so drift cannot masquerade as an arm effect"
for r in $(seq 1 8); do
  if [ $((r % 2)) -eq 1 ]; then order="off on"; else order="on off"; fi
  for a in $order; do
    v=$(bench "$a" | grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" | grep -oE "[0-9.]+$")
    log "  r$r aslr=$a: ${v:-FAIL}"
  done
done
log "=== summary"
for a in off on; do
  grep -oE "aslr=$a: [0-9.]+" "$D/log" | awk '{print $2}' | \
    awk -v a="$a" 'NF{n++;s+=$1;q[n]=$1} END{if(n<2){print "  "a": n="n; exit}
      mu=s/n; for(i=1;i<=n;i++) v+=(q[i]-mu)^2; sd=sqrt(v/(n-1));
      lo=q[1]; hi=q[1]; for(i=1;i<=n;i++){if(q[i]<lo)lo=q[i]; if(q[i]>hi)hi=q[i]}
      printf "  aslr %s: n=%d mean=%.2f sd=%.2f spread=%.2f cv=%.1f%%\n", a, n, mu, sd, hi-lo, 100*sd/mu}' | tee -a "$D/log"
done
touch "$D/DONE"; log done
