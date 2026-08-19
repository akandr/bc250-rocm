# umr captures of a stalled board, July 2026

Two captures taken while the board was stalled, holding the kernel-side view of
the waiting threads, which show the process parked in `kfd_wait_on_events`.

These are kept as archival. They belong to an event-latency line of
investigation from that period, and the load-hang conclusion that thread fed
into was later retracted: the apparent hangs were a harness mistake, a newer
llama.cpp CLI ignoring `-no-cnv` and spinning on a closed stdin, and every
"hung" load had in fact completed. The stacks themselves are real captures of a
real stall; it is the interpretation built on them that did not survive.
