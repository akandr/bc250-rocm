# The per-op replay faults the GPU, and it is the harness, 2026-09-21

A soak started on 21 September reported ten kernel fault lines already present in the current boot. They
were not the rare fault this repository has been chasing. They were one event, all ten lines at the same
second on 20 September, and they came from the tool used to compare backends operation by operation.

## The event

    amdgpu 0000:01:00.0: [gfxhub] page fault (src_id:0 ring:24 vmid:8 pasid:7373)
    amdgpu 0000:01:00.0:  Process test-backend-op pid 287125
    amdgpu 0000:01:00.0:   in page starting at address 0x00007f800f3c9000 from client 0x1b (UTCL2)
    amdgpu 0000:01:00.0:          Faulty UTCL2 client ID: TCP (0x8)
    amdgpu 0000:01:00.0:          PERMISSION_FAULTS: 0x3

Ten of those at 18:31:07, one second before a `test-backend-ops perf --test-file` replay of the
qwen3.6-35B MoE's pp512 graph finished. The host and the GPU both carried on; every measurement since
has been taken on the same boot.

## Reproducing it, and what it is not

Six replays of the same file, three with the packed-fp16 GEMM enabled and three with
`GGML_RDNA1_PKF16=0` ([`repro-log`](repro-log), [`scripts/opreplay_fault_repro.sh`](../../scripts/opreplay_fault_repro.sh)):

| arm | runs | exit | fault lines added |
|---|---|---|---|
| GEMM on | 3 | 134 each | 10 each |
| GEMM off | 3 | 134 each | 10 each |

Reproducible six times out of six, and identical with the kernel under suspicion switched off, so it is
not that kernel. It is also not the board and not the operation: `test-backend-ops -o GET_ROWS` passes
111 of 111 correctness cases, and the models themselves run for hours.

The replay names the operation as it dies
([`replay-gemm-on.log`](replay-gemm-on.log)):

    GET_ROWS(name=ffn_moe_weights-0, type=f32, ne=[1,8,1,1],
             sources=f32[1,256,1,1], i32[8,1,1,1]): Memory access fault by GPU node-1 on address 0

That is the mixture-of-experts router gather: eight indices into a 256-row table. In the correctness
path the harness fills an index tensor with values that are in range. In the perf path it does not, and
`get_rows` does not bound them, because in a real graph they are always valid. So the shader reads from
wherever the random index points. Vulkan's replay of the same file runs to the end, presumably because
its out-of-range reads happen to land in mapped memory; that is luck, not a difference in correctness.

## What to do about it

Nothing in llama.cpp: the operation is right and the graphs it is given are right. The lesson is about
the measurement. **A graph containing `GET_ROWS` or `MUL_MAT_ID` cannot be replayed to the end on ROCm
with `test-backend-ops perf --test-file`,** so check that a replay finished before reading its numbers.
The pp512 replay of this model measures 75 operations before it aborts where Vulkan measures 119; the
one-token replay in [`logs/moe-decode-2026-09-20/`](../moe-decode-2026-09-20/) runs to the end and its
comparison is complete.

The practical consequence for this repository is smaller than it looks, because the fault lines it
leaves in the journal are indistinguishable at a glance from the ones the real defect leaves. A soak
that counts fault lines will count these too. Any such count taken on a boot where a per-op replay has
run needs this subtracted first.

## Files

`repro-log` is the six-run reproduction, `replay-gemm-{on,off}.log` one replay from each arm, and
`dmesg-first-event.txt` the original event of 20 September.
