# The six benchmark models, as their files describe them, 2026-10-10

What each GGUF under `/opt/models` says about itself, read with llama.cpp's own `gguf-py`
([`scripts/gguf_metadata.py`](../../scripts/gguf_metadata.py)), and the files' SHA-256 sums, so a
reader can tell whether a copy is the one measured here (`metadata.txt`, `sha256.txt`; the directory
also holds two files no table on the front page uses, `gemma4.gguf` and `qwen3.5-9b.gguf`).

| model, as the tables name it | file | the file records | nominal type | trained context | sha256 |
|---|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | `qwen2.5-1.5b-q4km.gguf` | `qwen2.5-1.5b-instruct`, version v0.1; no quantiser, no repository | Q4_K_M | 32768 | `6a1a2eb6d156` |
| qwen3-8B Q8_0 | `qwen3-8b-q8_0.gguf` | `Qwen3 8B`, Apache-2.0; no quantiser, no repository | Q8_0 | 40960 | `d87f4a5a2f1a` |
| deepseek-r1-14B Q4_K_M | `deepseek-r1-14b.gguf` | `DeepSeek R1 Distill Qwen 14B`; no quantiser, no repository | Q4_K_M | 131072 | `6e9f90f02bb3` |
| qwen3-14B Q4_K_M | `qwen3-14b.gguf` | `Qwen3 14B`; no quantiser, no repository | Q4_K_M | 40960 | `a8cc1361f314` |
| qwen3.6-35B-A3B MoE IQ2_M | `qwen3.6-35b-a3b-iq2m.gguf` | `Qwen3.6-35B-A3B`, quantised by Unsloth (`huggingface.co/unsloth`) from `Qwen/Qwen3.6-35B-A3B`, 256 experts | IQ2_M, a dynamic mix: 181 Q5_K, 80 IQ2_XXS, 70 Q6_K, 37 IQ3_XXS, 3 IQ4_XS tensors beside 361 F32 | 262144 | `2be7ef1ed7e1` |
| qwen3.8-27B UD-IQ3_XXS | `qwen3.8-27b-iq3xxs.gguf` | `Qwen3.8-27B`, quantised by Unsloth from `Qwen/Qwen3.8-27B` | file type Q4_K_S, a dynamic mix: 195 IQ3_S, 149 IQ4_XS, 112 IQ3_XXS, 48 IQ1_M, 1 Q5_K tensors beside 360 F32 | 262144 | `0a6129dcbbbe` |

Four of the six files carry a model name and a quantisation type and nothing about where they came
from; those four are reproducible only by checksum. The two Unsloth files are dynamic quantisations,
so the type in their name is nominal and the tensors are the mix listed, which is why the prefill GEMM
of patch 8 covers IQ2_XXS, IQ3_XXS, IQ3_S and IQ4_XS and the matvec of patch 5 the IQ types.

The sums were taken with `sha256sum` on the board; the metadata with `gguf-py` from llama.cpp 7ba604f.
