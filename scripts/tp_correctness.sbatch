#!/bin/bash
# ============================================================================
# Tensor-parallel correctness check.
# Runs the same prompt with the same seed at tp_size = 1, 2, 4 and compares
# perplexity. Expectations:
#   - tp=1 should be bit-exact identical to the original single-GPU run
#     (the dist path is bypassed when tp_size==1; this is the
#     unchanged-single-GPU baseline).
#   - tp=2/4 perplexity should agree with tp=1 to ~1e-3 (cross-rank reduction
#     reorder introduces small fp16/fp32 noise).
#   - tp=2/4 greedy completion should match tp=1 for hundreds of tokens.
#
# Submit:   sbatch scripts/tp_correctness.sbatch   (from the repo root)
# Override: MODEL=path/to/model.yalm sbatch scripts/tp_correctness.sbatch
# ============================================================================

#SBATCH --job-name=yalm-tp-check
#SBATCH --output=tp-correctness-%j.out
#SBATCH --error=tp-correctness-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4              # we use up to 4 ranks
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4            # need at least 4 GPUs for tp=4

set -euo pipefail

echo "=== Environment ==="
hostname; date
nvidia-smi || { echo "no GPU visible?"; exit 1; }
echo

module load gnu12/12.3.0 2>/dev/null || true
export CXX=g++ CC=gcc

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  cd "$SLURM_SUBMIT_DIR"
fi
echo "Project: $(pwd)"

echo "=== Build ==="
make clean
time make CUARCH="${CUARCH:-sm_75}" -j 8
echo

MODEL="${MODEL:-llama-3.2-3b-instruct-fp16.yalm}"
if [ ! -f "$MODEL" ]; then
  echo "FATAL: model not found: $MODEL"; exit 1
fi
echo "Model: $MODEL"
echo

# Fixed prompt + seed for deterministic comparison.
PROMPT="The capital of France is"
SEED=42
N=32
TEMP=0.0    # greedy — eliminates RNG path entirely

# --------------------------------------------------------------------------
# Single-GPU baseline (no MPI). Goes through the original CUDA-graph path.
# --------------------------------------------------------------------------
echo "=== Baseline: ./build/main (no mpirun, tp=1, original CUDA graph path) ==="
./build/main "$MODEL" -i "$PROMPT" -n $N -t $TEMP -s $SEED > out.baseline 2>&1 || true
tail -25 out.baseline
echo

# --------------------------------------------------------------------------
# Same prompt under mpirun at various tp sizes. tp=1 still uses the original
# path (tp_size==1 short-circuits the dist path in Model::forward), so
# baseline == tp=1 must be bit-exact. tp=2/4 use the dist path.
# --------------------------------------------------------------------------
for P in 1 2 4; do
  echo "=== mpirun -n $P (tp=$P) ==="
  mpirun -n $P ./build/main "$MODEL" -i "$PROMPT" -n $N -t $TEMP -s $SEED > out.tp$P 2>&1 || true
  tail -25 out.tp$P
  echo
done

# --------------------------------------------------------------------------
# Compare generation outputs (greedy => deterministic given correct math).
# Look at the generated token stream (between the prompt echo and the stats
# block). If everything is right, every file produces the same tokens.
# --------------------------------------------------------------------------
echo "=== Output diffs ==="
echo "--- baseline vs tp=1 (should be identical) ---"
diff out.baseline out.tp1 | head -40 || true
echo
echo "--- baseline vs tp=2 ---"
diff out.baseline out.tp2 | head -40 || true
echo
echo "--- baseline vs tp=4 ---"
diff out.baseline out.tp4 | head -40 || true
echo

# --------------------------------------------------------------------------
# Perplexity check — averages logprobs over a longer sequence, more sensitive
# to numerical drift than a single greedy run.
# --------------------------------------------------------------------------
PERPLEXITY_PROMPT="Large language models predict the next token given a context. The attention mechanism allows the model to weigh different parts of the input differently. Memory bandwidth, not compute, is the dominant bottleneck during decode on most GPUs."

echo "=== Perplexity baseline (no mpirun) ==="
./build/main "$MODEL" -m perplexity -i "$PERPLEXITY_PROMPT" -s $SEED 2>&1 | tee out.ppl.baseline | grep -E "perplexity:" || true
echo

for P in 1 2 4; do
  echo "=== Perplexity mpirun -n $P ==="
  mpirun -n $P ./build/main "$MODEL" -m perplexity -i "$PERPLEXITY_PROMPT" -s $SEED 2>&1 \
    | tee out.ppl.tp$P | grep -E "perplexity:" || true
  echo
done

echo "=== Done ==="
date
