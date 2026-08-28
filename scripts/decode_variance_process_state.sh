#!/usr/bin/env bash
# What makes decode at depth vary between processes?
#
# The clue: llama-bench repetitions inside one invocation spread by about 0.43,
# while separate invocations of the same command spread 1.20. The variance lives
# in whatever differs per process rather than in the decode loop.
#
# A first version of this used a non-mmap arm and every run of it aborted, which
# turned out to be a memory ceiling worth its own write-up but measured nothing
# about variance. All three arms here load the same way, so the comparison is
# about process state rather than load path:
#
#   A  default, as every decode measurement in this repository is taken
#   B  pinned to a fixed set of CPUs, in case scheduling or migration matters
#   C  page cache dropped and memory compacted first, so the weights land in a
#      freshly allocated and less fragmented physical layout
#
# Six rounds, arms rotated, since blocked designs on this board have repeatedly
# produced differences that are not there.
set -u
D=~/inv93; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

bench () { "$@" env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 20 1800 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 2>&1 \
  | grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+" | grep -oE "[0-9.]+$"; }

run () { # run <arm> <tag>
  local arm=$1 tag=$2 v
  case $arm in
    A) v=$(bench) ;;
    B) v=$(bench taskset -c 0-5) ;;
    C) sync; echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1
       echo 1 | sudo tee /proc/sys/vm/compact_memory > /dev/null 2>&1; sleep 5
       v=$(bench) ;;
  esac
  log "  $tag arm=$arm: ${v:-FAIL}"
}

log "=== decode at depth 16128, three arms, rotated, all mmapped"
for r in 1 2 3 4 5 6; do
  case $((r % 3)) in
    1) order="A B C" ;;
    2) order="B C A" ;;
    0) order="C A B" ;;
  esac
  for a in $order; do run $a "r$r"; done
done

log "=== within-process spread for comparison: one invocation, eight repetitions"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
  timeout -k 20 3600 $HIP/llama-bench -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 8 \
  > "$D/within.log" 2>&1
log "  $(grep -aoE "tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+ . +[0-9.]+" "$D/within.log" | tail -1)"

log "=== summary"
for a in A B C; do
  grep -oE "arm=$a: [0-9.]+" "$D/log" | cut -d" " -f2 | \
    awk -v a="$a" 'NF{n++;s+=$1;q[n]=$1} END{if(n<2){print "  arm "a": n="n; exit}
      mu=s/n; for(i=1;i<=n;i++) v+=(q[i]-mu)^2; sd=sqrt(v/(n-1));
      printf "  arm %s: n=%d mean=%.2f sd=%.2f cv=%.1f%%\n", a, n, mu, sd, 100*sd/mu}' | tee -a "$D/log"
done
touch "$D/DONE"; log done
