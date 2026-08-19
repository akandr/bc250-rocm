`GPU_FORCE_BLIT_COPY_SIZE` moving the SDMA copy boundary, which is the second,
independent way of showing that the failure is the async copy path rather than
the size of the transfer.

ROCclr uses a blit compute kernel below a threshold and hands larger copies to
an SDMA engine. With SDMA left enabled, a 1 MiB copy hangs at every setting up
to 1023 and completes at 1024 and above, so the boundary tracks the knob
exactly. Reaching the same result by a second mechanism is why this is kept:
`HSA_ENABLE_SDMA=0` and this knob are different levers on the same choice.

`knob.txt` is the sweep. The staging-buffer knob, which does not move the
boundary, is in `../../loose-ends-2026-08-18/sdma-staging/`.
