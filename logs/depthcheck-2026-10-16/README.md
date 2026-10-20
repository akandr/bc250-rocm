# Correctness at depth: a needle in a 32k, 64k and 128k haystack, 2026-10-16

## What was open

The context ceilings say how deep each model can still generate, and the perplexity gates and the greedy
reply checks were all run at ordinary context lengths. Whether what comes out at depth is right, and
whether ROCm and Vulkan still agree there, had not been checked. The 1.5B's trained length is 32768.

## Method

[`scripts/depthcheck.sh`](../../scripts/depthcheck.sh): qwen2.5-1.5B Q4_K_M, a prompt that opens with
"Remember this: the secret code word is lantern-742.", continues with wikitext (which does not contain the
word) to about 32k, 64k and 128k tokens, and ends with a question asking for the code word. Greedy, 32
tokens, `-fa on`, `-c` 2048 above the prompt, on ROCm `build-hip-pkf16` and Vulkan `build-vk-f44`, both at
7ba604f; the two outputs are compared byte for byte.

The tool is `llama-completion`, the classic CLI, built out of tree against each build's own libraries. Two
traps on the way: the server-based `llama-cli` of this tree caps its slot at the model's training context
("the slot context (67584) exceeds the training context of the model (32768) - capping"), so a 64k prompt
is rejected whatever `-c` says, and `llama-perplexity -c 65536` dies with `std::bad_alloc`, since it keeps
the logits of the whole window (65536 × 151936 floats). Those first attempts are the `needle_*_*.err` and
`ppl_*_*.err` files; the results below are `needle2_*`.

The HIP libraries this links against had been relinked from the same source tree earlier that day; before
use, the build reproduced the campaign's 1.5B greedy reply verbatim and its speed within a percent
(pp512 1788.5 against 1798.6, tg64 197.5 against 196.7).

## Result

| prompt | ROCm answer | Vulkan answer | outputs |
|---|---|---|---|
| 32k (above the trained length) | lantern-742 | lantern-742 | identical |
| 64k | lantern-742 | lantern-742 | identical |
| 128k | garbage | garbage, different | differ |

**Up to twice its trained length the 1.5B still finds the fact at the start of the prompt, and ROCm and
Vulkan produce the same reply byte for byte** (`lantern-742`, then end of text; the prompts were 30940
and 60158 tokens, and 32 was only the generation cap). At four times, both backends produce unrelated text,
and different text, which is what two numerically different implementations do once the model itself has
left the range its positional encoding was trained for. So the context ceilings are capacity figures: the
1080k the 1.5B can hold on ROCm is not 1080k it can use, and the useful range ends somewhere between 64k
and 128k for this model, on either backend.

## What this does not say

One model, one fact, one position. No perplexity at these windows (see the trap above), no YaRN or other
rope scaling, which is how a 32k model is meant to be run beyond 32k. The 128k row says nothing about
which backend is "more right"; neither is.

## Files

- `log`: both phases, every run with its wall time and answer.
- `needle2_{32,64,128}k_{hip,vk}.out`, `.err`: the second phase's generated text and llama-completion's
  stderr (timings). The prompt files are not kept; the script regenerates them from `wiki.test.raw`.
