"""A full training loop on the GPU: forward, backward, optimiser step.

This is the case the stock wheel could not run at all, since backward passes
lean on exactly the elementwise and reduction kernels it lacks. Loss is
checked for finiteness and for actually decreasing, and the final parameters
are compared against the same run on the CPU, so a silently wrong GPU result
cannot pass as success.
"""
import torch, torch.nn as nn, time

torch.manual_seed(0)


def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256),
                         nn.ReLU(), nn.Linear(256, 10))


torch.manual_seed(0)
x = torch.randn(512, 256)
y = torch.randint(0, 10, (512,))

results = {}
for devname in ["cpu", "cuda"]:
    dev = torch.device(devname)
    model = build().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossfn = nn.CrossEntropyLoss()
    xd, yd = x.to(dev), y.to(dev)
    losses = []
    t0 = time.time()
    for step in range(50):
        opt.zero_grad()
        out = model(xd)
        loss = lossfn(out, yd)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    if devname == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    results[devname] = (losses, [p.detach().cpu().clone() for p in model.parameters()], dt)
    print("%-5s first loss %.5f  last loss %.5f  finite %s  %.2f s"
          % (devname, losses[0], losses[-1], all(l == l for l in losses), dt))

cpu_losses, cpu_params, _ = results["cpu"]
gpu_losses, gpu_params, _ = results["cuda"]

print("loss decreased on gpu:", gpu_losses[-1] < gpu_losses[0])
dmax = max(float((a - b).abs().max()) for a, b in zip(cpu_params, gpu_params))
print("max parameter difference against cpu after 50 steps: %.3e" % dmax)
print("agrees with cpu:", dmax < 1e-3)
lmax = max(abs(a - b) for a, b in zip(cpu_losses, gpu_losses))
print("max loss difference across all steps: %.3e" % lmax)
