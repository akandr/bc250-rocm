#!/usr/bin/env bash
# What does the f32 compute-type workaround cost? Second attempt.
#
# The first attempt was invalid through my own error: two copies of the script
# were launched and ran concurrently, interleaving into one log and competing
# for the GPU, which is why most of its cells read FAIL and the rest cannot be
# trusted. The lock below is so that cannot happen again.
#
# The claim under test is load-bearing: the write-up says the workaround "costs
# nothing measurable", and it is the recommended fix for the fp16 defect. The
# figures behind that sentence survive in no shipped log and their harness is
# not among the shipped scripts, so this measures the claim directly rather than
# trying to reproduce numbers whose conditions are unknown.
set -u
exec 9>~/.inv97.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv97; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv96/DONE ]; do sleep 120; done
log "=== f32 compute type on (A) against off (B), qwen3-8B Q8_0, pp512 and tg64"

run () { # run <arm> <tag>
  local arm=$1 tag=$2 out pp tg
  if [ "$arm" = A ]; then
    out=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
          timeout -k 20 1800 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa on -p 512 -n 64 -r 1 2>&1)
  else
    out=$(env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L \
          timeout -k 20 1800 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa on -p 512 -n 64 -r 1 2>&1)
  fi
  echo "$out" > "$D/${tag}_$arm.log"
  pp=$(echo "$out" | grep -aoE "pp512 \| +[0-9]+\.[0-9]+" | grep -oE "[0-9.]+$")
  tg=$(echo "$out" | grep -aoE "tg64 \| +[0-9]+\.[0-9]+"  | grep -oE "[0-9.]+$")
  log "  $tag arm=$arm pp512=${pp:-FAIL} tg64=${tg:-FAIL}"
}

for r in $(seq 1 8); do
  if [ $((r % 2)) -eq 1 ]; then order="A B"; else order="B A"; fi
  for a in $order; do run $a "r$r"; done
done

log "=== summary"
for a in A B; do
  for m in pp512 tg64; do
    grep -oE "arm=$a .*$m=[0-9]+\.[0-9]+" "$D/log" | grep -oE "$m=[0-9.]+" | cut -d= -f2 | \
      awk -v a="$a" -v m="$m" 'NF{n++;s+=$1;q[n]=$1} END{if(n<2){print "  arm "a" "m": n="n; exit}
        mu=s/n; for(i=1;i<=n;i++) v+=(q[i]-mu)^2; sd=sqrt(v/(n-1));
        printf "  arm %s %s: n=%d mean=%.2f sd=%.2f\n", a, m, n, mu, sd}' | tee -a "$D/log"
  done
done
touch "$D/DONE"; log done
