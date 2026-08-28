#!/usr/bin/env bash
# What is the correct perplexity when fusion is refused?
#
# Disabling fusion left the KQV defect present, but one of its eight values was
# 9.9968 rather than the 10.0103 that is correct on the normal path. If fusion
# changes the arithmetic slightly, then 9.9968 is that arm's correct answer and
# the pass criterion used to score it was wrong.
#
# The patched build settles it: with the KQV precision line in place the defect
# cannot occur, so whatever that arm returns repeatedly is its correct value.
set -u
exec 9>~/.inv106.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv106; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
WIKI=~/wiki.test.raw
SRC=~/llama-master
C=$SRC/ggml/src/ggml-cuda/ggml-cuda.cu
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cp "$C" "$D/ggml-cuda.cu.orig"
log "=== adding the fusion kill switch to the PATCHED tree (KQV line left in place)"
python3 - "$C" <<'PYEOF'
import sys
p=sys.argv[1]; s=open(p).read()
anchor = """                               std::initializer_list<enum ggml_unary_op> unary_ops) {"""
add = """                               std::initializer_list<enum ggml_unary_op> unary_ops) {
    {   // BC-250 probe: refuse every fusion when asked
        static const bool bc250_nofuse = getenv("BC250_NO_FUSION") != nullptr;
        static unsigned long bc250_calls = 0;
        if (bc250_nofuse) {
            if (++bc250_calls % 50000 == 1) {
                fprintf(stderr, "BC250NOFUSE refusals=%lu\\n", bc250_calls);
            }
            return false;
        }
    }"""
assert s.count(anchor)==1
open(p,"w").write(s.replace(anchor, add, 1))
PYEOF
cmake --build $SRC/build-hip -j6 > "$D/build.log" 2>&1
log "  build rc=$? set_prec present: $(grep -c 'set_prec(kqv' $SRC/src/llama-graph.cpp)"

ppl () { # ppl <tag> <extra env...>
  local tag=$1; shift
  env HSA_ENABLE_SDMA=0 LD_LIBRARY_PATH=$L GGML_CUDA_NO_VMM=1 "$@" \
    timeout -k 30 1800 $SRC/build-hip/bin/llama-perplexity -m $Q15 --no-mmap -ngl 99 \
    -fa off -c 2048 -f $WIKI --chunks 2 > "$D/$tag.log" 2>&1
  log "  $tag: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/$tag.log" | grep -oE '[0-9.]+$' || echo FAIL) refusals=$(grep -aoE 'refusals=[0-9]+' "$D/$tag.log" | tail -1 | grep -oE '[0-9]+$' || echo 0)"
}

log "=== patched build, fusion ON, four runs (expect 10.0103)"
for i in 1 2 3 4; do ppl "fuse_on_$i"; done
log "=== patched build, fusion REFUSED, four runs"
for i in 1 2 3 4; do ppl "fuse_off_$i" BC250_NO_FUSION=1; done

log "=== restoring"
cp "$D/ggml-cuda.cu.orig" "$C"
cmake --build $SRC/build-hip -j6 > "$D/rebuild.log" 2>&1
log "  rebuild rc=$? probe removed: $(grep -c BC250NOFUSE "$C")"
log "=== summary"
log "  fusion on:  $(grep -oE 'fuse_on_[0-9]: [0-9.]+' "$D/log" | awk '{print $2}' | sort -u | tr '\n' ' ')"
log "  fusion off: $(grep -oE 'fuse_off_[0-9]: [0-9.]+' "$D/log" | awk '{print $2}' | sort -u | tr '\n' ' ')"
touch "$D/DONE"; log done
