# g5-infer — custom inference stack for MiMo-V2.6-Distill-9B on Apple M5

Target model (only): `MiMo-V2.6-Distill-Qwen-9B` (dense 8.95B, Qwen3.5 arch).
Weights in use: GGUF `Q2_K` (3.55 GiB). Machine: M5, 24 GB unified, ~150 GB/s.

## Honest ceilings (measured, not marketed)

- Single-stream decode is bandwidth-bound: 3.55 GiB/token ÷ ~150 GB/s ≈ **42 tok/s max**.
  Measured llama.cpp: **28.6**. Gap to close with custom Metal work: ~1.4x.
- **100 tok/s single-stream is physically impossible** for this 9B on this chip.
  100 tok/s **system throughput** (parallel jobs) is achievable and is the goal.

## What we steal from each engine

- **llama.cpp**: GGUF zero-reformat load, full Metal offload (`-ngl 99`), flash attention,
  fused dequant+matvec, `mlock` residency. Baseline we must beat, not rewrite on day one.
- **MLX**: unified-memory zero-copy discipline (never duplicate weights CPU↔GPU),
  persistent command buffers, lazy eval. Lesson: keep weights resident, reuse buffers.
- **SGLang**: radix prefix cache (shared system prompts evaluated once), continuous batching.
- **vLLM**: PagedAttention (KV in fixed blocks; matters once 4+ slots share the cache).

## Phases

1. **Gateway** (`gateway.py`): OpenAI-compatible front door. Exact-prefix response cache,
   parallel fan-out across server slots, per-request timings log. Delivers aggregate 100 now.
2. **Profile**: find where 28.6 → 42 goes (Metal timings, thread/batch sweeps, kernel selection).
3. **Custom Metal decode**: M5-AMX-tuned Q2_K vec-dot kernels, fused sampling, resident weights.
   Target: 38–42 single-stream (the ceiling), then stack batching on top.

## Layout

- `gateway.py` — phase 1 gateway (stdlib only).
- `metal/` — profiling notes + custom kernel work (phase 2/3).
- `bench.sh` — reproducible measurements.
- Backend engine (temporary, being outgrown): `llama-server -m <Q2_K> -ngl 99 -fa 1 -np 4`.
