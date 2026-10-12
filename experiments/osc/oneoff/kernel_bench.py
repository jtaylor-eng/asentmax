"""Old (pre-dc550e7, kept as oneoff/triton_stieltjes_old_dc550e7.py) vs new Triton Stieltjes kernel:
fwd+bwd wall at the Table-1 training shapes, plus tile-size ablation for the new kernel.
Run from synthetic/ with the oneoff dir on sys.path (see kernel_bench.sbatch)."""
import sys, time, torch, triton
sys.path.insert(0, ".")
import triton_stieltjes_old_dc550e7 as old
import src.kernels.adasplash.triton_stieltjes as newmod
from src.models.architectures.sparse_gemma import get_nape_slopes

dev = "cuda"
print(torch.cuda.get_device_name(), "smem", triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"])

def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters * 1e3

# training shapes: sort/copy/reverse/mqmtar batches are (B, 8 heads, N<=64..128, D=32) bf16
for (B, H, N, D) in [(128, 8, 64, 32), (128, 8, 128, 32), (64, 8, 256, 32), (8, 8, 2048, 32), (1, 8, 16384, 32)]:
    slopes = get_nape_slopes(H, H // 2).to(dev)
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True); v = torch.randn_like(q, requires_grad=True)
    do = torch.randn_like(q)
    def f_old():
        old.stieltjes_attention(q, k, v, causal=True, sm_scale=D ** -0.5, stieltjes_q=4.0, num_iter=30,
                                normalize=True, ift_grad=True, solver="nr", alibi_slopes=slopes).backward(do)
    def f_new():
        newmod.stieltjes_attention(q, k, v, causal=True, sm_scale=D ** -0.5, stieltjes_q=4.0, num_iter=30,
                                   alibi_slopes=slopes).backward(do)
    orig_pick = newmod._pick_blocks
    res = {"old": bench(f_old), "new": bench(f_new)}
    for bm, bn in [(32, 32), (64, 64), (128, 64), (64, 32)]:
        newmod._pick_blocks = lambda D_, e_, d_, bm=bm, bn=bn: (bm, bn)
        try:
            res[f"new{bm}x{bn}"] = bench(f_new)
        except Exception as e:
            res[f"new{bm}x{bn}"] = float("nan")
    newmod._pick_blocks = orig_pick
    print(f"B={B:3d} N={N:5d}: " + "  ".join(f"{k} {v:7.2f}ms" for k, v in res.items()), flush=True)
