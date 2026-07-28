import torch, sys
dev = torch.device("cuda"); torch.manual_seed(0)
def test(n, iters):
    a = torch.randn(n, n); b = torch.randn(n, n)
    c_cpu = a @ b
    ag = a.to(dev); bg = b.to(dev)
    for k in range(iters):
        cg = ag @ bg; torch.cuda.synchronize()
    err = (cg.cpu() - c_cpu).abs().max().item()
    rel = err / (c_cpu.abs().max().item() + 1e-9)
    print("N=%d x%d: rel_err=%.2e %s" % (n, iters, rel, "OK" if rel < 1e-2 else "WRONG"), flush=True)
for (n,it) in [(1024,1),(2048,1),(4096,1),(4096,50),(8192,1)]:
    test(n,it)
