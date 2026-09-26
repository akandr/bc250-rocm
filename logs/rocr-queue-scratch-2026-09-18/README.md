# The segfault at the memory limit: ROCr's queue scratch guard, and what is behind it, 2026-09-18

When a deep context pushes pinned system memory past the KFD limit (13422 MiB here), the kernel returns
`-ENOMEM` and the process dies with `segfault at 20 ... in libhsa-runtime64.so.1.18.0` instead of an
error ([`logs/fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/)). That page matched the
symptom to rocm-systems PR #2850, merged February 2026 and absent from ROCm 7.1.1, and said so was a match
on the symptom, not a proof. This is the proof, the fix ported and tested, and the second bug it
uncovers.

## The crash, named

Fedora ships debuginfo for both runtimes; downloaded and extracted, not installed, and read with
`eu-addr2line` against the addresses the kernel logs (`repro-14b-d16384.txt`):

| library | fault | function | source |
|---|---|---|---|
| stock `libhsa-runtime64.so.1.18.0` | read at `0x20`, ip offset `0x3e4bf` | `rocr::AMD::GpuAgent::ReleaseQueueMainScratch(ScratchCache::ScratchInfo &)` | `runtime/hsa-runtime/core/inc/scratch_cache.h:177` |

That is the release the PR guards: `GpuAgent::QueueCreate` sets up a scope guard that releases the queue's
main scratch on any early exit, whether or not the scratch was ever acquired, and when the acquisition
fails at the memory limit the guard releases a null base.

## Reproducing it exactly

`rocr_limit_probe.cpp`, which fills memory with `hipMalloc` in 512 MiB steps and then creates a stream,
does **not** reach this path: with the stock runtime the stream still comes up (`probe-stock.log`), so
the queue's scratch is not what fails first when only plain allocations are outstanding. The September
command does: `llama-bench -m qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2`, the 14B with a
16384-token context, which fills the context and then dies (`14b-d16384-stock.log`, exit 139, `segfault
at 20 ... in libhsa-runtime64.so.1.18.0`).

## The fix, ported and built

`0002-rocr-guard-queue-scratch-release.patch` is PR #2850's hunk for `QueueCreate`, rewritten for the
`-p3` prefix Fedora's spec uses. `build_rocr_fixed.sh` rebuilds Fedora's `rocm-runtime-7.1.1-6` source
package with it (the spec's out-of-order `%changelog` has to be dropped, since current `rpm` treats it
as an error) and extracts `libhsa-runtime64.so.1.18.0` without installing anything; the tests below run
it through `LD_LIBRARY_PATH`, and the last section is what it took to install it.

## What the fix exposes

With the patched runtime the same command no longer crashes in ROCr. Some of the time it crashes in HIP
(`14b-d16384-patched.log`): `segfault at 8 ... error 6 in libamdhip64.so.7.1.52802`, a write through a
null pointer plus eight. `eu-addr2line` on that address named `amd::ReleaseExtObjectsCommand`'s
destructor at `command.hpp:1465`, and the first version of this page reported that; it was wrong. The
address is the target of identical-code folding and the symbol it carries is whichever of the folded
bodies the linker kept. A debugger with the runtime's debuginfo loaded gives the real frames
(`relink-r1.log` to `r6.log`, six runs out of six):

    #0 amd::HostQueue::Thread::Release()          rocclr/platform/commandqueue.hpp:187
    #1 amd::HostQueue::terminate()                rocclr/platform/commandqueue.cpp:96
    #2 hip::Stream::terminate()                   hipamd/src/hip_stream.cpp:85
    #3 amd::ReferenceCountedObject::release()
    #4 hip::Stream::Destroy()                     hipamd/src/hip_stream.cpp:79
    #5 hip::Device::NullStream(wait=false)        hipamd/src/hip_device.cpp:42
    #6 hip::GraphNode::CaptureAndFormPacket()
    #7 hip::GraphExec::CaptureAndFormPacketsForGraph()
    #9 hip::GraphExec::Init()  <- hipGraphInstantiate  <- ggml_cuda_graph_evaluate_and_capture

The mechanism, from the 7.1.1 source: a HIP stream is a `HostQueue`, whose `Thread::Init` asks the device
for a virtual device; that calls `Device::acquireQueue`, which tries `hsa_queue_create` at halving sizes
down to 64 and gives up, so at the memory limit `virtualDevice_` stays null. `Stream::Create` fails and
the stream is destroyed on the spot, and `HostQueue::terminate` calls `thread_.Release()`
unconditionally, which is `virtualDevice_->release()`: the reference-count decrement on a null object,
eight bytes in. `ROCm/clr` `develop` has the null check in `Thread::Release`, so this was fixed upstream
after 7.1.1; `0001-rocclr-hostqueue-thread-release-null-vdev.patch` is that guard for the 7.1.1 source,
and `build_clr_fixed.sh` rebuilds Fedora's `rocclr` package with it.

## Why it is intermittent

The stream that fails is the device's null stream, created lazily by `hipGraphInstantiate` when llama.cpp
first instantiates a HIP graph after the context is full. The reading that fits the two outcomes is that
it depends on which request is the first to hit the limit: a kernel launch on an existing stream returns
the error and llama.cpp aborts with `ROCm error: out of memory`; a null-stream creation inside the graph
instantiation fails and the runtime faults in the teardown above. The tally on this page is eleven clean
runs and two faults with the runtime loaded through `LD_LIBRARY_PATH` or installed and no debugger, four
clean under `gdb` with `LD_LIBRARY_PATH`, and six faults out of six under `gdb` with the installed copy
(`relink.log`; re-copying the library before alternate runs changed nothing), so the debugger and the
loader path move the order in which the two requests reach the limit. Which request goes first was not
instrumented; the bug itself does not depend on it.

## The HIP side, rebuilt

`build_clr_fixed2.sh` rebuilds Fedora's `rocclr` source package (which produces `libamdhip64`) with the
`Thread::Release` guard; `rpmbuild --nodeps` with the spec's `%fdupes` line dropped, because the three
build tools it wanted are packaging helpers and `dnf` on this board is blocked by an unrelated broken
`kernel-devel` entry. The library is staged in the home directory and run through `LD_LIBRARY_PATH`,
not installed. With it the 1.5B gate reads 8.9498 under either library and pp512 / tg64 are the same,
911 to 914 and 180 to 181 (`clrverify.log`).

The deep-context command still segfaults with it, three runs of three (`clrfix-r1.log` to `r3.log`), one
frame further along the same path: `amd::Command::Command(HostQueue &, ...)` called from
`hip::ihipLaunchKernelCommand` from `hip::GraphKernelNode::CreateCommand(hip::Stream *)` from
`GraphNode::CaptureAndFormPacket`. `Device::NullStream` already handles a failed stream, it destroys it
(cleanly, now) and returns null with `Cannot create new Stream object` in the log; the caller in
`hip_graph_internal.hpp` does not look:

    auto capture_stream = hip::getNullStream(g_devices[dev_id_]->devices()[0]->context(), false);
    hipError_t status = CreateCommand(capture_stream);

and the kernel command's constructor reads the queue through the null reference. `ROCm/clr` `develop`
has the identical two lines as of this date, so unlike the first null this one is not fixed upstream.
`0002-hip-graph-capture-null-stream.patch` returns `hipErrorOutOfMemory` there, which is what a failed
stream creation reports elsewhere in HIP and what makes `hipGraphInstantiate` return an error llama.cpp
can print; `build_clr_fixed3.sh` builds the library with both patches (`hiplib3`), and `clrfix3-r1.log` to
`r3.log` are the same three-run check on it: **three runs of three end in `ggml-cuda.cu:110: ROCm error`
and `abort()`**, the outcome the ROCr fix was for, in the configuration (gdb, installed ROCr) that had
faulted six times of six. Two null checks, one already upstream and one not, and the crash at the
memory limit is a reported error on this board.

## Installed

Both outcomes kill the process, and the stock runtime segfaults every time, so the patched one is not
worse on this path and is the clean error most of the time. Before it joined the rocBLAS and comgr copies
in `/opt/bc250-rocm/lib64` (`rocr_validate_install.sh`, `rocr_gates.sh`): the 1.5B's pp512 / tg64 read
914.7 / 180.1 and 914.5 / 180.3 under the stock library against 914.6 / 179.3 and 914.9 / 180.8 under the
patched one (`validate.log`); the perplexity gates are identical under both, 8.9498 for the 1.5B and
9.1273 for the 8B (`gates.log`, the stock library reached through a directory holding only a symlink to
it, so rocBLAS and comgr stayed overridden). Installed 2026-09-18 21:49, `md5 3f5bfa2a848e` against the
stock `08c62b88257b`; `ldconfig -p` lists the `/opt/bc250-rocm` entry first. The reproducible form is
[`scripts/build_rocr711_scratch_fix.sh`](../../scripts/build_rocr711_scratch_fix.sh) with
[`patches/rocr-guard-queue-scratch-release.patch`](../../patches/rocr-guard-queue-scratch-release.patch).
