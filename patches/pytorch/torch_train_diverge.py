"""Is the GPU/CPU parameter difference after training numerical drift or a defect?

Two checks that tell those apart:

  1. Growth. Ordinary floating-point divergence starts near zero and compounds
     step by step, especially under Adam, which renormalises the step size and
     so does not damp an early difference. A wrong kernel shows a large
     difference at the very first step.
  2. Determinism. The same run twice on the GPU must give bit-identical
     parameters. If it does not, the problem is not CPU-versus-GPU arithmetic.

A single-step gradient comparison is included as the cleanest signal: one
forward and one backward, no optimiser state, straight against the CPU.
"""
import torch, torch.nn as nn

def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256),
                         nn.ReLU(), nn.Linear(256, 10))

torch.manual_seed(0)
x = torch.randn(512, 256)
y = torch.randint(0, 10, (512,))
lossfn = nn.CrossEntropyLoss()

# 1. one step, gradients only
grads = {}
for devname in ["cpu", "cuda"]:
    dev = torch.device(devname)
    m = build().to(dev)
    out = m(x.to(dev))
    loss = lossfn(out, y.to(dev))
    loss.backward()
    grads[devname] = [p.grad.detach().cpu().clone() for p in m.parameters()]
    print("%s single-step loss %.8f" % (devname, float(loss.detach().cpu())))

gmax = max(float((a - b).abs().max()) for a, b in zip(grads["cpu"], grads["cuda"]))
grel = gmax / max(float(a.abs().max()) for a in grads["cpu"])
print("single-step max gradient difference: %.3e (relative %.3e)" % (gmax, grel))

# 2. divergence growth over steps
def run(dev, steps):
    d = torch.device(dev)
    m = build().to(d)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    xd, yd = x.to(d), y.to(d)
    snaps = {}
    for s in range(steps):
        opt.zero_grad()
        lossfn(m(xd), yd).backward()
        opt.step()
        if s + 1 in (1, 5, 10, 25, 50):
            snaps[s + 1] = [p.detach().cpu().clone() for p in m.parameters()]
    return snaps

cpu_s = run("cpu", 50)
gpu_s = run("cuda", 50)
print("\nstep  max parameter difference vs cpu")
for k in sorted(cpu_s):
    d = max(float((a - b).abs().max()) for a, b in zip(cpu_s[k], gpu_s[k]))
    print("%4d  %.3e" % (k, d))

# 3. gpu determinism
g2 = run("cuda", 50)
same = all(bool(torch.equal(a, b)) for a, b in zip(gpu_s[50], g2[50]))
print("\ngpu run reproduces bit-identically:", same)
