#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# ROCm's dispatch floor against Vulkan's, measured the same way on the same tree.
#
# logs/dispatch-floor-2026-09-22/ measured ROCm's floor alone and said on the page that Vulkan's was
# not measured, so it could not claim ROCm's was the larger. This measures both.
#
# Needs scripts/dispatch_floor_ggml.c built against two build trees of the same llama.cpp checkout,
# one with the HIP backend and one with Vulkan, so the only difference is the backend:
#
#   gcc -O2 -o /tmp/df_hip scripts/dispatch_floor_ggml.c -I$TREE/ggml/include -L$HIPB -lggml -lggml-base -lm
#   gcc -O2 -o /tmp/df_vk  scripts/dispatch_floor_ggml.c -I$TREE/ggml/include -L$VKB  -lggml -lggml-base -lm
#
# Run under flock on an otherwise idle board.

set -u

TREE=${TREE:-$HOME/llama-new}
HIPB=${HIPB:-$TREE/build-hip-f44/bin}
VKB=${VKB:-$TREE/build-vk/bin}
PAIRS=${PAIRS:-3}

cd /tmp || exit 1

# per-node microseconds at the smallest tensor, for the chain arm and the fan arm
vals() { grep -E "^ +1024 " | awk '{print $2}' | tr '\n' ' '; }

echo "tree $(git -C "$TREE" log -1 --format='%h %ad' --date=short 2>/dev/null)"
echo "ggml $(basename "$(ls "$HIPB"/libggml-base.so.0.* 2>/dev/null | head -1)")"
echo "start $(date -Iseconds) edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
echo
printf '%-5s %-22s %8s %8s %8s\n' pair arm chain fan edge

for p in $(seq 1 "$PAIRS"); do
    h_on=$(LD_LIBRARY_PATH=$HIPB ./df_hip 2>/dev/null | vals)
    h_off=$(LD_LIBRARY_PATH=$HIPB GGML_CUDA_DISABLE_GRAPHS=1 ./df_hip 2>/dev/null | vals)
    v=$(LD_LIBRARY_PATH=$VKB ./df_vk 2>/dev/null | vals)
    e=$(sensors 2>/dev/null | awk '/edge/{print $2}')

    printf '%-5s %-22s %8s %8s %8s\n' "p$p" "ROCm, HIP graphs"   $h_on  "$e"
    printf '%-5s %-22s %8s %8s %8s\n' "p$p" "ROCm, no graphs"    $h_off "$e"
    printf '%-5s %-22s %8s %8s %8s\n' "p$p" "Vulkan"             $v     "$e"
done

echo
echo "chain = dependent nodes, no overlap possible. fan = independent nodes, overlap permitted."
echo "end $(date -Iseconds) edge $(sensors 2>/dev/null | awk '/edge/{print $2}')"
