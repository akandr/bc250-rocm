#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Correctness at depth on qwen2.5-1.5B Q4_K_M, ROCm against Vulkan: a needle in a haystack, one fact at the
# start of a 32k, 64k and 128k prompt and a question about it at the end, greedy, 32 tokens, on both backends,
# with the two answers compared. llama-completion, the classic CLI, because the server-based llama-cli caps
# its slot at the model's training context (32768) and llama-perplexity cannot hold the logits of a 64k window.
#
# llama-completion for each backend was compiled out of tree against that build's libraries, so that the
# measured builds are not rebuilt:
#   c++ -std=c++17 -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU -DGGML_USE_CUDA -DGGML_USE_HIP \
#       -DLLAMA_SHARED -DLLAMA_SUBPROCESS -I. -Icommon -Ivendor -Iinclude -Iggml/include -O2 -DNDEBUG \
#       -o ~/llama-completion-hip tools/completion/main.cpp tools/completion/completion.cpp \
#       -Wl,-rpath,$B/bin $B/bin/libllama-common.so.0.0.1 $B/bin/libllama.so.0.0.1 $B/bin/libggml.so.0.19.0 \
#       $B/bin/libggml-cpu.so.0.19.0 $B/bin/libggml-hip.so.0.19.0 $B/bin/libggml-base.so.0.19.0 \
#       $B/common/libllama-common-base.a        # B=build-hip-pkf16; Vulkan: -DGGML_USE_VULKAN, libggml-vulkan
set -u
O=${1:-$HOME/depthcheck-2026-10-16}; mkdir -p "$O"
TXT=$HOME/wiki.test.raw
L=/opt/bc250-rocm/lib64
HIP=$HOME/llama-master/build-hip-pkf16/bin
VK=$HOME/llama-master/build-vk-f44/bin
M=/opt/models/qwen2.5-1.5b-q4km.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
STOPPED=""
for u in ollama hw-watcher.timer crond; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; STOPPED="$STOPPED $u"; fi
done
restore () { for u in $STOPPED; do sudo systemctl start $u; done; log "restored:$STOPPED"; }
trap restore EXIT
sleep 5
log "start: stopped$STOPPED; edge $(temp); text $(wc -c < $TXT) bytes"
# the prompts: the fact, filler from the text (which does not contain it), the question
python3 - "$TXT" "$O" <<'PYGEN'
import sys, re
txt, out = sys.argv[1], sys.argv[2]
body = re.sub(r"\s+", " ", open(txt, encoding="utf-8", errors="replace").read())
fact = "Remember this: the secret code word is lantern-742. "
q = "\n\nQuestion: what is the secret code word mentioned at the very beginning? Answer with the code word only.\nAnswer:"
for k in (32, 64, 128):
    n_chars = int(k * 1024 * 3.9)   # about 3.9 characters per token on this text with Qwen's tokenizer
    filler = (body * (n_chars // len(body) + 1))[:n_chars]
    open(f"{out}/needle_{k}k.txt", "w").write(fact + filler + q)
PYGEN
for k in 32 64 128; do
  for be in hip vk; do
    b=$HIP; [ $be = vk ] && b=$VK
    C=$(( k * 1024 + 2048 ))
    t0=$(date +%s)
    LD_LIBRARY_PATH=$L timeout -k 30 7200 "$b/llama-completion" -m "$M" -ngl 99 -fa on -c $C -b 2048 -ub 512 --temp 0 -n 32 \
        -no-cnv --no-warmup --no-display-prompt -f "$O/needle_${k}k.txt" > "$O/needle2_${k}k_$be.out" 2> "$O/needle2_${k}k_$be.err"
    ans=$(tr '\n' ' ' < "$O/needle2_${k}k_$be.out" | sed -E 's/^ +//' | cut -c1-160)
    ntok=$(grep -aoE "prompt eval time = +[0-9.]+ ms / +[0-9]+ tokens" "$O/needle2_${k}k_$be.err" | tail -1 | grep -oE "[0-9]+ tokens")
    pp=$(grep -aoE "prompt eval time =.*tokens per second" "$O/needle2_${k}k_$be.err" | tail -1 | grep -oE "[0-9.]+ tokens per second")
    log "needle ${k}k $be: ${ntok:-?} prompt at ${pp:-?}, $(( $(date +%s) - t0 )) s, answer: ${ans:-<none>} edge=$(temp)"
    sleep 20
  done
  if cmp -s "$O/needle2_${k}k_hip.out" "$O/needle2_${k}k_vk.out"; then log "needle ${k}k: hip and vk outputs IDENTICAL"; else log "needle ${k}k: hip and vk outputs DIFFER"; fi
done
touch "$O/DONE"; log DONE
