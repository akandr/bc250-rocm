"""Does the zeroed fp16 GEMM reproduce outside llama.cpp?

In llama.cpp the first fp16 GEMM of a graph, after unrelated work, sometimes
returns all zeros with both operands verifiably intact. A standalone HIP
reproducer never triggered it, which left open whether the defect belongs to
that program or to the stack underneath it. PyTorch is an independent
implementation on the same rocBLAS and the same board, so the same pattern is
worth trying here: unrelated work, then a half-precision matmul, repeatedly,
with every result checked for a zeroed output.

Checks stay on the host so a failed check cannot be confused with a failed GEMM.
"""
import torch, time, sys

dev = torch.device("cuda")
cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 200

# shapes near the llama.cpp case: tall-thin activations against square weights
SHAPES = [(2048, 2048, 512), (4096, 4096, 128), (1536, 1536, 256), (8192, 1024, 64)]

zeroed = []
wrong = []
t0 = time.time()

for c in range(cycles):
    m, k, n = SHAPES[c % len(SHAPES)]

    # unrelated work first: allocation churn plus elementwise, the context in
    # which the llama.cpp case appears
    junk = torch.randn(1024, 1024, device=dev)
    junk = (junk * 1.5).relu().sum()
    torch.cuda.synchronize()
    scratch = [torch.empty(64 * 1024 * (1 + (i % 7)), device=dev) for i in range(6)]
    del scratch

    a = torch.randn(m, k, device=dev, dtype=torch.float16)
    b = torch.randn(k, n, device=dev, dtype=torch.float16)
    out = a @ b
    torch.cuda.synchronize()

    oc = out.float().cpu()
    if bool((oc == 0).all()):
        # confirm the inputs were not themselves zero before blaming the GEMM
        ain = float(a.float().cpu().abs().sum())
        bin_ = float(b.float().cpu().abs().sum())
        zeroed.append((c, m, k, n, ain, bin_))
        print("ZEROED at cycle %d shape %dx%dx%d  input sums %.1f %.1f"
              % (c, m, k, n, ain, bin_), flush=True)
    else:
        ref = a.float().cpu() @ b.float().cpu()
        rel = float((oc - ref).abs().max()) / (float(ref.abs().max()) + 1e-9)
        if rel > 5e-2:
            wrong.append((c, rel))
            print("WRONG at cycle %d rel=%.3e" % (c, rel), flush=True)

    if (c + 1) % 50 == 0:
        print("  %d/%d cycles, %d zeroed, %d wrong (%.1fs)"
              % (c + 1, cycles, len(zeroed), len(wrong), time.time() - t0), flush=True)

print("\n%d cycles: %d zeroed outputs, %d wrong outputs" % (cycles, len(zeroed), len(wrong)))
print("VERDICT:", "REPRODUCES outside llama.cpp" if zeroed else "does not reproduce here")
