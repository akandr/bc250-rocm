"""Which PyTorch operations can run on this board with the stock ROCm wheel.

Separates the two code paths: operations dispatched into a math library
(rocBLAS/hipBLASLt, shipped with its own code objects) from operations that
launch PyTorch's own compiled kernels (built for the wheel's arch list).
Each case is run in isolation so one failure does not mask the rest.
"""
import torch, traceback

dev = torch.device("cuda")
results = []


def case(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        results.append((name, "OK", out))
    except Exception as e:
        msg = str(e).split("\n")[0][:70]
        results.append((name, "FAIL", msg))


# host-built inputs, copied to device: no PyTorch kernel needed to create them
a = torch.randn(512, 512)
b = torch.randn(512, 512)
ad, bd = a.to(dev), b.to(dev)
ref = a @ b

case("h2d copy + d2h copy", lambda: float((ad.cpu() - a).abs().max()))
case("matmul (library)", lambda: round(float((ad @ bd).cpu().sub(ref).abs().max()), 6))
case("matmul fp16 (library)",
     lambda: round(float((ad.half() @ bd.half()).float().cpu().sub(ref).abs().max()), 4))
case("addmm (library)", lambda: round(float(torch.addmm(bd, ad, bd).cpu().sub(ref + b).abs().max()), 5))

# PyTorch's own kernels
case("randn on device", lambda: tuple(torch.randn(512, 512, device=dev).shape))
case("elementwise add", lambda: round(float((ad + bd).cpu().sub(a + b).abs().max()), 6))
case("elementwise mul", lambda: round(float((ad * bd).cpu().sub(a * b).abs().max()), 6))
case("sum reduction", lambda: round(float(ad.sum().cpu() - a.sum()), 3))
case("relu", lambda: round(float(torch.relu(ad).cpu().sub(torch.relu(a)).abs().max()), 6))
case("softmax", lambda: round(float(torch.softmax(ad, -1).cpu().sub(torch.softmax(a, -1)).abs().max()), 6))
case("zeros + fill", lambda: float(torch.zeros(16, device=dev).fill_(3.0).sum().cpu()))

print("%-26s %-5s %s" % ("case", "state", "value or error"))
for n, s, v in results:
    print("%-26s %-5s %s" % (n, s, v))

ok = sum(1 for _, s, _ in results if s == "OK")
print("\n%d of %d cases ran" % (ok, len(results)))
print("arch list:", torch.cuda.get_arch_list())
