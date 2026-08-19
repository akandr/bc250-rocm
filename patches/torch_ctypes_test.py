"""Run a gfx1013-compiled kernel on tensors allocated by the stock torch wheel.

The wheel's own elementwise kernels fail with `invalid device function` because
it ships no gfx1013 code objects. If a kernel built for gfx1013 computes
correctly on the wheel's own device memory, in the wheel's process and on its
stream, then the runtime, driver, allocator and hardware are all fine, and the
missing code objects are the whole of the problem.
"""
import ctypes, torch

lib = ctypes.CDLL("/home/akandr/bc250_ext.so")
lib.bc250_add_scale.restype = ctypes.c_int
lib.bc250_add_scale.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                ctypes.c_float, ctypes.c_int]

dev = torch.device("cuda")
n = 1 << 20
a = torch.randn(n)
b = torch.randn(n)
ad, bd = a.to(dev), b.to(dev)
out = torch.empty_like(ad)          # allocated by torch's caching allocator

rc = lib.bc250_add_scale(ctypes.c_void_p(ad.data_ptr()),
                         ctypes.c_void_p(bd.data_ptr()),
                         ctypes.c_void_p(out.data_ptr()), 2.5, n)
torch.cuda.synchronize()
ref = (a + b) * 2.5
err = float((out.cpu() - ref).abs().max())
print("launch rc:", rc)
oc = out.cpu()
print("gfx1013 kernel on torch memory: max abs err %.3e  correct: %s  all zero: %s"
      % (err, err < 1e-4, bool((oc == 0).all().item())))

# larger and repeated, to be sure it is not a one-shot fluke
for trial in range(3):
    big = 1 << 24
    x = torch.randn(big).to(dev)
    y = torch.randn(big).to(dev)
    o = torch.empty_like(x)
    lib.bc250_add_scale(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()),
                        ctypes.c_void_p(o.data_ptr()), 1.5, big)
    torch.cuda.synchronize()
    e = float((o.cpu() - (x.cpu() + y.cpu()) * 1.5).abs().max())
    print("  trial %d at 16.7M elements: max abs err %.3e" % (trial, e))

# and the wheel's own equivalent, in the same process, for contrast
try:
    w = (ad + bd) * 2.5
    torch.cuda.synchronize()
    print("wheel's own kernel: ran, err %.3e" % float((w.cpu() - ref).abs().max()))
except Exception as e:
    print("wheel's own kernel:", str(e).split("\n")[0][:60])
