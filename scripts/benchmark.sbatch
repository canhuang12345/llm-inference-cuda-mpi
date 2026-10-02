#!/bin/bash
# ============================================================================
# yalm CUDA inference benchmark — SLURM submission script
#
# Invokes src/main.cpp's CLI to exercise the optimized inference pipeline:
#   completion mode — measures decode throughput + per-token latency
#   passkey mode    — exercises long-context KV cache scaling
#   perplexity mode — pure forward-pass workload, no sampling
#
# Submit:   sbatch scripts/benchmark.sbatch
# Status:   squeue -u $USER
# Cancel:   scancel <jobid>
# Output:   benchmark-<jobid>.out (in the directory you submitted from)
#
# Adjust the #SBATCH lines below for your specific cluster.
# Common Stanford clusters:
#   - Sherlock GPU partition: --partition=gpu, may need module load cuda/12.2.0
#   - Local non-SLURM run: bash scripts/benchmark.sbatch
# ============================================================================

#SBATCH --job-name=yalm-bench
#SBATCH --output=benchmark-%j.out
#SBATCH --error=benchmark-%j.out
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# GPU request — Turing-class card (T4, RTX 20xx).
# Most-portable form. If your cluster rejects it, swap to one of the alternates.
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:1
# #SBATCH --gres=gpu:t4:1

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup. Adjust module load lines for your cluster.
# ---------------------------------------------------------------------------
echo "=== Environment ==="
hostname
date
nvidia-smi || { echo "nvidia-smi failed — no GPU visible?"; exit 1; }
echo

# Common module setups (uncomment the one that matches your cluster):
# module load cuda/12.2.0           # Sherlock-style
# module load cuda                  # generic
# module load nvhpc                 # NVIDIA HPC SDK
# On our cluster the system g++ 8.5.0 doesn't support -std=c++20. gnu12/12.3.0
# provides GCC 12.3 which does. Load it if available; harmless if not.
module load gnu12/12.3.0 2>/dev/null || true
module list 2>&1 || true

echo
echo "Compiler versions:"
which nvcc && nvcc --version | tail -2
which g++  && g++ --version | head -1
echo

# ---------------------------------------------------------------------------
# Build for Turing (sm_75). Override with CUARCH=sm_80 etc. if needed.
# ---------------------------------------------------------------------------
# Resolve project root. Under SLURM the batch script is copied to a spool
# directory, so ${BASH_SOURCE[0]} points there — use SLURM_SUBMIT_DIR (the
# directory you ran `sbatch` from) when available. Fall back to the script's
# own location for non-SLURM runs.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
fi
cd "$PROJECT_DIR"
echo "Project root: $PROJECT_DIR"

CUARCH_OVERRIDE="${CUARCH:-sm_75}"
# The nvhpc module sets CXX=nvc++, but the Makefile uses GCC-only
# flags like -ffast-math / -mavx2. Force the host compiler to g++; nvcc is
# still used for .cu files via the Makefile's NVCC variable.
export CXX=g++
export CC=gcc
echo "=== Building (CUARCH=$CUARCH_OVERRIDE, CXX=$CXX) ==="
make clean
time make CUARCH="$CUARCH_OVERRIDE" -j 8
time make CUARCH="$CUARCH_OVERRIDE" test -j 8
echo

# ---------------------------------------------------------------------------
# Correctness: run the unit tests before any benchmark.
# Catches numerical regressions in rmsnorm / matmul / flash attention.
# ---------------------------------------------------------------------------
echo "=== Correctness tests (./build/test) ==="
./build/test
echo

# ---------------------------------------------------------------------------
# Resolve model path. Override with MODEL=path/to/file.yalm
# ---------------------------------------------------------------------------
MODEL="${MODEL:-llama-3.2-3b-instruct-fp16.yalm}"
if [ ! -f "$MODEL" ]; then
  echo "FATAL: model file not found: $MODEL"
  echo "Set MODEL=... or place llama-3.2-3b-instruct-fp16.yalm in $(pwd)"
  exit 1
fi
echo "Using model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo

# ---------------------------------------------------------------------------
# Memory bandwidth baseline (CPU only — used as a reference number).
# Skipped on our cluster: -b wants 32 GB and -b2 wants 128 GB, which
# exceeds either the SLURM allocation or the node's total RAM. GPU benchmarks
# below are the actual focus. Re-enable on a fat-memory host if needed.
# ---------------------------------------------------------------------------
# echo "=== Memory bandwidth: CPU AVX2/SIMD (./build/test -b) ==="
# ./build/test -b
# echo
#
# echo "=== Memory bandwidth: CPU std::thread (./build/test -b2) ==="
# ./build/test -b2
# echo

# ---------------------------------------------------------------------------
# Per-kernel microbenchmarks (single-launch timing, Mistral-7B-sized inputs).
# Useful as raw kernel cost numbers; pair with `ncu` for deeper analysis.
# ---------------------------------------------------------------------------
echo "=== Per-kernel microbenchmarks (./build/test -bk ...) ==="
for k in matmul matmul-wide mha ffn; do
  echo "--- kernel: $k ---"
  time ./build/test -bk "$k"
done
echo

# ---------------------------------------------------------------------------
# Inference benchmarks via src/main.cpp's CLI, with the in-program profiler
# enabled (-p). Three scenarios that stress different parts of the engine:
#   1. Short prompt, short decode  — base kernel + launch overhead
#   2. Long  prompt, short decode  — prefill throughput
#   3. Short prompt, long  decode  — decode scaling with kv_len
#
# `-t 0.0` selects argmax (deterministic) for reproducible outputs.
# ---------------------------------------------------------------------------
PROMPT_SHORT="The capital of France is"
PROMPT_LONG="Write a detailed technical explanation of how Large Language Models work. Include sections on tokenization, attention mechanisms, and inference. Discuss the role of KV cache in autoregressive decoding. Explain why memory bandwidth is the dominant bottleneck on consumer GPUs. Conclude with a comparison of FlashAttention and standard attention. Discuss the difference between paged and contiguous KV cache, and how dynamic batching enables higher GPU utilization in production serving systems. Note the specific challenges of running these systems on commodity hardware like the T4."

echo "=== Scenario 1: completion, short prompt + 64-token decode ==="
./build/main "$MODEL" -i "$PROMPT_SHORT" -n 64 -t 0.0 -p
echo

echo "=== Scenario 2: completion, long prompt + 32-token decode ==="
./build/main "$MODEL" -i "$PROMPT_LONG" -n 32 -t 0.0 -p
echo

echo "=== Scenario 3: completion, short prompt + 256-token decode ==="
./build/main "$MODEL" -i "$PROMPT_SHORT" -n 256 -t 0.0 -p
echo

# ---------------------------------------------------------------------------
# Passkey retrieval — long-context test. Each junk line adds ~30 tokens; 100
# junk lines puts kv_len in the ~3k-token range, near the 4096 ctx cap.
# Bump to -n 130 if you want to fully saturate the cache.
# ---------------------------------------------------------------------------
echo "=== Scenario 4: passkey retrieval (long context) ==="
./build/main "$MODEL" -m passkey -n 100 -p
echo

# ---------------------------------------------------------------------------
# Perplexity — one OUTPUT_LOGITS forward per prompt token, no sampling.
# Pure forward-pass workload; the perplexity number itself should be stable
# across optimization phases (correctness regression signal).
# ---------------------------------------------------------------------------
echo "=== Scenario 5: perplexity ==="
./build/main "$MODEL" -m perplexity -i "$PROMPT_LONG" -p
echo

echo "=== Done ==="
date
