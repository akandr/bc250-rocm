# umr captures of a stalled board, July 2026

Two captures taken while the board was stalled, holding the kernel-side view of
the waiting threads, which show the process parked in `kfd_wait_on_events`.

These are kept as archival. They belong to an event-latency line of
investigation from that period, and the load-hang conclusion that thread fed
into was later retracted: the apparent hangs were a harness mistake, a newer
llama.cpp CLI ignoring `-no-cnv` and spinning on a closed stdin, and every
"hung" load had in fact completed. The stacks themselves are real captures of a
real stall; it is the interpretation built on them that did not survive.

What the per-wave dumps also show, read later: in `fresh_stall_155145.txt` the three waves have
`STATUS=0x00012441`, with HALT set, and `TRAPSTS.EXCP=0x100`, the memory-violation bit; the bit is set in
`stall_capture_154607.txt` too. Neither file records its kernel or boot arguments. Both were taken on
14 July, minutes after the fresh-boot loop in [`../stock/`](../stock/) ended, in the same session.
