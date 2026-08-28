#!/usr/bin/env bash
# Two more open questions, both cheap.
#
# A. A tension in the record. The zeroed fp16 GEMMs land on the first call of
#    every graph execution after the first, which reads as capture behaving
#    differently from replay. But this repository also records, from 18 August,
#    that HIP graph capture is not involved: six runs per arm with capture off
#    were as wrong as with it on. Both cannot be the whole story. Counting the
#    zeros with capture disabled says which reading survives.
#
# B. Decode variance lives in per-process setup and four candidates are
#    eliminated. Nothing has asked whether the variance depends on depth. If it
#    grows with the primed cache, that points at where the cache lands; if it is
#    flat, the cache is not involved and the search narrows the other way.
set -u
exec 9>~/.gaps2.lock; flock -n 9 || { echo locked; exit 1; }
D=~/inv124; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== A. do the zeroed GEMMs survive with graph capture disabled?"
for arm in on off; do
  if [ "$arm" = off ]; then extra=(GGML_CUDA_DISABLE_GRAPHS=1); else extra=(); fi
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 BC250_TEMP=1 LD_LIBRARY_PATH=$L "${extra[@]}" \
    timeout -k 30 900 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f ~/wiki.test.raw --chunks 1 > "$D/graphs_$arm.log" 2>&1
  tot=$(grep -ac BC250TEMP "$D/graphs_$arm.log")
  zer=$(grep -a BC250TEMP "$D/graphs_$arm.log" | grep -c "abs_sum=0 ")
  pos=$(grep -a BC250TEMP "$D/graphs_$arm.log" | grep -n "abs_sum=0 " | cut -d: -f1 | tr '\n' ' ')
  ppl=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/graphs_$arm.log" | grep -oE '[0-9.]+$')
  log "  graph capture $arm: $tot GEMM outputs, $zer zero at positions [$pos] ppl=${ppl:-FAIL}"
done

log "=== B. does decode variance depend on depth?"
for d in 2048 8192 16128; do
  vals=""
  for i in 1 2 3 4 5; do
    v=$(env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
        timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d $d -r 1 2>&1 \
        | grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" | grep -oE "[0-9.]+$")
    vals="$vals ${v:-FAIL}"
  done
  log "  depth $d:$vals"
  echo "$vals" | tr ' ' '\n' | grep -E "^[0-9.]+$" | \
    awk -v d="$d" 'NF{n++;s+=$1;q[n]=$1} END{if(n<2)exit; mu=s/n; for(i=1;i<=n;i++)v+=(q[i]-mu)^2;
      sd=sqrt(v/(n-1)); printf "    depth %s: n=%d mean=%.2f sd=%.2f cv=%.1f%%\n", d, n, mu, sd, 100*sd/mu}' | tee -a "$D/log"
done
touch "$D/DONE"; log done
