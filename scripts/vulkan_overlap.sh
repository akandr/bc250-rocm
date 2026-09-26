#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The default perf logger syncs after EVERY node, so its total is a fully serialized GPU span.
# CONCURRENT mode writes timestamps only where a sync is already required, so it measures the
# real overlapped execution. The difference is what Vulkan gains from overlap.
set -u
VKB=${VKB:-$HOME/llama-new/build-vk/bin}
M=${M:-/opt/models/qwen3.6-35b-a3b-iq2m.gguf}
cd /tmp
for mode in serial concurrent; do
  if [ "$mode" = concurrent ]; then export GGML_VK_PERF_LOGGER_CONCURRENT=1; else unset GGML_VK_PERF_LOGGER_CONCURRENT; fi
  echo "===== $mode"
  LD_LIBRARY_PATH=$VKB GGML_VK_PERF_LOGGER=1 $VKB/llama-bench -m "$M" -p 0 -n 8 -r 1 -ngl 99 \
    > /tmp/vkconc-$mode.out 2> /tmp/vkconc-$mode.err
  grep -E "tg8" /tmp/vkconc-$mode.out | tail -1
  echo "perf blocks: $(grep -c '^Vulkan Timings:' /tmp/vkconc-$mode.err)"
  grep "^Total time:" /tmp/vkconc-$mode.err | tail -3
done
echo "===== baseline, no logger"
LD_LIBRARY_PATH=$VKB $VKB/llama-bench -m "$M" -p 0 -n 8 -r 1 -ngl 99 2>/dev/null | grep tg8
