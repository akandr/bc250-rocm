#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# llama-bench with llama-bench-no-depth-cache.patch, built against the existing HIP and Vulkan builds of the
# same llama.cpp tree without touching them: one binary per backend in its own directory, each linked to
# its build's libraries (rpath). With LLAMA_BENCH_NO_DEPTH_CACHE=1 it skips the host copy of the sequence
# state that llama-bench otherwise makes after every depth fill, even with -r 1.
set -eu
S=${S:-~/llama-master}
O=${O:-~/llama-bench-nosnap}
P=$(cd "$(dirname "$0")" && pwd)/llama-bench-no-depth-cache.patch
mkdir -p "$O/hip" "$O/vk"; cd "$O"
cp "$S/tools/llama-bench/llama-bench.cpp" "$S/tools/llama-bench/main.cpp" .
patch llama-bench.cpp < "$P"
for be in hip vk; do
  case $be in
    hip) B=$S/build-hip-pkf16; DEFS="-DGGML_USE_CUDA -DGGML_USE_HIP"; G=libggml-hip.so.0.19.0 ;;
    vk)  B=$S/build-vk-f44;    DEFS="-DGGML_USE_VULKAN";             G=libggml-vulkan.so.0.19.0 ;;
  esac
  # the flags of the tree's own llama-bench target (compile_commands.json), compiled as one executable
  c++ -DGGML_BACKEND_SHARED -DGGML_SHARED -DGGML_USE_CPU $DEFS -DLLAMA_SHARED -DLLAMA_SUBPROCESS \
      -I. -I"$S/common" -I"$S/vendor" -I"$S/include" -I"$S/ggml/include" -O3 -DNDEBUG \
      -o "$be/llama-bench" main.cpp llama-bench.cpp -Wl,-rpath,"$B/bin" \
      "$B/bin/libllama-common.so.0.0.1" "$B/bin/libllama.so.0.0.1" "$B/bin/libggml.so.0.19.0" \
      "$B/bin/libggml-cpu.so.0.19.0" "$B/bin/$G" "$B/bin/libggml-base.so.0.19.0" "$B/common/libllama-common-base.a"
  echo "built $O/$be/llama-bench"
done
