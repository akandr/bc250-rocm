#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The context ceilings again, without llama-bench's depth snapshot. llama-bench as shipped copies the
# sequence state into a host buffer after the depth fill; on ROCm clr pins that buffer against the KFD
# limit and on Vulkan it costs the same in RAM, so the ceilings of the earlier passes
# (logs/context-ceilings-2026-09-30/earlier-passes/) are llama-bench's rather than the backends'. Same
# harness (ceilings_bisect.sh), same four arms, all five models and three caches, with the copy switched
# off (build_llama_bench_nosnap.sh).
#
# Every depth that generated with the copy also generates without it, so the earlier passes seed the
# search as lower bounds; their failures do not carry over. Order: the four larger models, the 1.5B with
# an f16 cache, a validation and throughput campaign of its own (nothing in
# logs/context-ceilings-2026-09-30/ comes from it), and last the 1.5B with quantised caches, whose
# ceilings lie near a million tokens and past it: one bisection step after the first failure, and a week
# per run before calling it a timeout.
set -u
D=~/ceilings-nosnap; mkdir -p "$D"
export LLAMA_BENCH_NO_DEPTH_CACHE=1 HIP=~/llama-bench-nosnap/hip VK=~/llama-bench-nosnap/vk
mark () { echo "[$(date '+%F %T')] $*" >> "$D/log"; sync; }
S=$D/seed-ok.tsv
awk -F'\t' '$5 == "ok"' ~/ceilings-compare/summary.tsv ~/ceilings-bisect/summary.tsv > "$S"
mark "chain start: $(grep -c "" "$S") earlier passes as lower bounds"
SEED=$S MODELS="qwen3-14b qwen3-8b-q8_0 qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m" bash ~/ceilings_bisect.sh "$D" ||
  { mark "chain stopped in phase A"; exit 1; }
mark PHASE-A-DONE
SEED=$S MODELS=qwen2.5-1.5b-q4km KVS=f16 bash ~/ceilings_bisect.sh "$D" || { mark "chain stopped in phase B"; exit 1; }
mark PHASE-B-DONE
bash ~/packet_capture_validation.sh ~/nosnap-validation
bash ~/campaign_packet_capture.sh ~/nosnap-throughput-1
bash ~/campaign_packet_capture.sh ~/nosnap-throughput-2
mark VALIDATION-DONE
SEED=$S MODELS=qwen2.5-1.5b-q4km KVS="q8_0 q4_0" RESDIV=2 TMO=604800 bash ~/ceilings_bisect.sh "$D" ||
  { mark "chain stopped in phase C"; exit 1; }
mark ALL-DONE
