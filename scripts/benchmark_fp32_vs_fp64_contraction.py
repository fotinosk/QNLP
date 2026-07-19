"""Benchmark opt_einsum contraction speed: float32 vs float64, batched vs per-sample.

Isolates whether the float64 contraction (added to fix the overflow/underflow
bug in EinsumModel) is responsible for epochs not speeding up despite the
batched same-topology fast path. Run directly on a cluster GPU node:

    python scripts/benchmark_fp32_vs_fp64_contraction.py
"""

import time

import opt_einsum
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

BOND = 10
EMB = 512
N_POSITIONS = 15  # realistic sentence length
BATCH = 256
N_ITERS = 30

# Chain: a,ab,bc,...,→ single embedding leg, matches real MPS-style diagrams.
expr = "a," + ",".join(f"{chr(97+i)}{chr(98+i)}" for i in range(N_POSITIONS - 2)) + f"->{chr(97+N_POSITIONS-2)}"
shapes = [(BOND,)] + [(BOND, BOND)] * (N_POSITIONS - 3) + [(BOND, EMB)]

batched_expr = "z" + expr.replace(",", ",z").replace("->", "->z")
batched_shapes = [(BATCH, *s) for s in shapes]


def bench(fn, *args, n=N_ITERS):
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn(*args)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


for dtype, name in [(torch.float32, "float32"), (torch.float64, "float64")]:
    tensors = [torch.randn(*s, device=device, dtype=dtype) for s in shapes]
    expr_obj = opt_einsum.contract_expression(expr, *shapes)
    t_single = bench(lambda: expr_obj(*tensors))
    print(f"[{name}] single-sample contraction: {t_single*1e3:.4f} ms/call")

    b_tensors = [torch.randn(*s, device=device, dtype=dtype) for s in batched_shapes]
    b_expr_obj = opt_einsum.contract_expression(batched_expr, *batched_shapes)
    t_batched = bench(lambda: b_expr_obj(*b_tensors))
    print(
        f"[{name}] batched (B={BATCH}) contraction: {t_batched*1e3:.4f} ms/call "
        f"({t_batched/BATCH*1e6:.2f} us/sample-equivalent)"
    )
    print(
        f"[{name}] speedup of batched vs {BATCH}x sequential single-sample calls: "
        f"{(t_single * BATCH) / t_batched:.1f}x"
    )
    print()

print("If float64's batched-vs-single speedup is much smaller than float32's,")
print("fp64 throughput on this GPU is likely the bottleneck cancelling out the")
print("batching win.")
