#!/usr/bin/env bash
# Confirmation of the memory-state effect on decode variance.
#
# A first run found that dropping the page cache and compacting memory before
# each invocation cuts the spread of 8B decode at depth from a coefficient of
# variation near 8 percent to 2.9 percent, while leaving the mean alone. The
# variance ratio was 7.32 against the default arm, two-tailed p 0.048 at six
# runs per arm. That is exactly the regime where an earlier comparison on this
# board reversed sign between two careful designs, so it needs more samples
# before it means anything.
#
# Two arms only, since CPU pinning was shown to do nothing and dropping it
# doubles the samples per arm for the same wall time. Twelve rounds, alternating
# ABBA so a drift over the four hours cannot masquerade as an arm effect.
set -u
D=~/inv95; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv94/DONE ]; do sleep 120; done

bench () { env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 20 1800 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
  | grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" | grep -oE "[0-9.]+$"; }

run () { # run <arm> <tag>
  local arm=$1 tag=$2 v
  if [ "$arm" = C ]; then
    sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
    echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1; sleep 5
  fi
  v=$(bench)
  log "  $tag arm=$arm: ${v:-FAIL}"
}

log "=== A default against C dropped-and-compacted, ABBA, twelve rounds"
for r in $(seq 1 12); do
  if [ $((r % 2)) -eq 1 ]; then order="A C"; else order="C A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== summary"
for a in A C; do
  grep -oE "arm=$a: [0-9.]+" "$D/log" | cut -d" " -f2 | \
    awk -v a="$a" 'NF{n++;s+=$1;q[n]=$1} END{if(n<2){print "  arm "a": n="n; exit}
      mu=s/n; for(i=1;i<=n;i++) v+=(q[i]-mu)^2; sd=sqrt(v/(n-1));
      printf "  arm %s: n=%d mean=%.2f sd=%.2f cv=%.1f%%\n", a, n, mu, sd, 100*sd/mu}' | tee -a "$D/log"
done
touch "$D/DONE"; log done
