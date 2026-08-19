Backing logs for figures quoted in the write-up whose original run directories
were never shipped.

These were recovered from the board during an audit that checked every
distinctive figure in the document against the logs here (see
`scripts/audit_figures.py`). Each file is the original run, copied unchanged;
nothing here was re-measured.

| file | figure it backs |
|---|---|
| `inv39/ds14_16k_ppl.log` | 4.5489, the 14B at a 16384-token context |
| `inv22/vboot_*/faon_ppl.log` | 9.9148, flash attention on, across three fresh boots |
| `inv22/vboot_*/off_ub8_ppl.log` | 9.8574, flash attention off at micro-batch 8, same three boots |
| `inv25/vk4096.log` | 11.0279, the Vulkan reference at context 4096 |
| `inv25/f32compute.log`, `inv25/kqvf32.log` | 11.0634, the f32-compute and KQV-precision arms |
| `bench-2026-08/clean_ppl_vk8.log` | 11.1859, the Vulkan side of the wikitext-2 comparison |
| `inv2/ppl_oldhip_ub8.log` | 11.2071, the ROCm side of the same comparison |
| `inv32/a_ub8.log` | 9.1309, the micro-batch 8 vector path under fp16 compute |
| `inv25/nkvo.log`, `inv25/kvf32.log`, `inv25/knobs.log` | 11.0467 and 11.0571, the KV-on-host and f32-KV-cache arms |

The last three were found only when the figure audit was widened past the README
to cover patch headers and the notes prepared for upstream, which is where they
are cited. Figures that leave the repository deserve the same check as the ones
that stay in it.

Two further figures had no surviving artifact at all and were re-measured
instead rather than left cited from memory: see `../macro-remeasure-2026-08-18/`.
