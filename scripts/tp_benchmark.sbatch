#!/bin/bash
# ============================================================================
# yalm tensor-parallel inference benchmark — SLURM submission script
#
# Sweeps tp_size ∈ {1, 2, 4} across the same scenarios as benchmark.sbatch:
#   1. short prompt + 64-token decode    — decode latency, low kv_len
#   2. long  prompt + 32-token decode    — prefill throughput
#   3. short prompt + 256-token decode   — decode scaling with kv_len
#   4. passkey retrieval                 — long-context KV cache
#   5. perplexity                        — pure forward (no sampling)
#
# Reports throughput (tok/s), latency (s/tok), and bandwidth (GB/s) for each
# (scenario, tp_size) combo, then prints a summary table for scaling analysis.
#
# Single-GPU baseline is `mpirun -n 1`, which takes the same code path as
# `./build/main` (tp_size==1 short-circuits the dist path). This is the fair
# comparison vs `mpirun -n 2/4` — same launcher, only tp varies.
#
# Submit:   sbatch scripts/tp_benchmark.sbatch   (from the repo root)
# Override: MODEL=path/to/X.yalm sbatch scripts/tp_benchmark.sbatch
# ============================================================================

#SBATCH --job-name=yalm-tp-bench
#SBATCH --output=tp-benchmark-%j.out
#SBATCH --error=tp-benchmark-%j.out
# gpu-turing partition caps wall-time at 00:30:00. Stay under that.
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G

# 4 GPUs needed for tp=4. Drop to gpu:2 + ntasks=2 + remove "4" from TP_SIZES
# below if your partition can't allocate 4.
#SBATCH --partition=gpu-turing
#SBATCH --gres=gpu:4

set -euo pipefail

echo "=== Environment ==="
hostname
date
nvidia-smi || { echo "no GPU visible?"; exit 1; }
echo

# Cluster setup (mirrors benchmark.sbatch).
module load gnu12/12.3.0 2>/dev/null || true
module list 2>&1 || true
echo

echo "Compiler versions:"
which nvcc && nvcc --version | tail -2
which g++  && g++ --version | head -1
which mpirun && mpirun --version 2>&1 | head -1
echo

# ---------------------------------------------------------------------------
# Build (same flags as benchmark.sbatch).
# ---------------------------------------------------------------------------
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
else
  PROJECT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"
fi
cd "$PROJECT_DIR"
echo "Project root: $PROJECT_DIR"

CUARCH_OVERRIDE="${CUARCH:-sm_75}"
export CXX=g++
export CC=gcc
echo "=== Building (CUARCH=$CUARCH_OVERRIDE, CXX=$CXX) ==="
make clean
time make CUARCH="$CUARCH_OVERRIDE" -j 8
echo

# Skip ./build/test here — partition wall-time is tight (30 min) and the
# kernel tests are already covered by benchmark.sbatch. Re-enable if needed.

# ---------------------------------------------------------------------------
# Model and benchmark parameters.
# ---------------------------------------------------------------------------
MODEL="${MODEL:-llama-3.2-3b-instruct-fp16.yalm}"
if [ ! -f "$MODEL" ]; then
  echo "FATAL: model file not found: $MODEL"; exit 1
fi
echo "Using model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo

# tp_size sweep. Edit this list if your allocation has fewer GPUs or if you
# want intermediate values (Llama-3.2 supports {1,2,4,8} given n_kv_heads=8).
TP_SIZES="${TP_SIZES:-1 2 4}"
SEED=42
TEMP=0.0      # greedy: removes RNG path from the comparison

PROMPT_SHORT="The capital of France is"
PROMPT_LONG="Write a detailed technical explanation of how Large Language Models work. Include sections on tokenization, attention mechanisms, and inference. Discuss the role of KV cache in autoregressive decoding. Explain why memory bandwidth is the dominant bottleneck on consumer GPUs. Conclude with a comparison of FlashAttention and standard attention. Discuss the difference between paged and contiguous KV cache, and how dynamic batching enables higher GPU utilization in production serving systems. Note the specific challenges of running these systems on commodity hardware like the T4."

# Output directory for per-run logs (so we can grep summary stats later).
RESULTS=tp-bench-results-${SLURM_JOB_ID:-$$}
mkdir -p "$RESULTS"
echo "Per-run logs: $RESULTS/"
echo

# ---------------------------------------------------------------------------
# Helper: run one (scenario, tp) and save full output. Continues past
# failures so one bad combo doesn't kill the whole sweep.
# ---------------------------------------------------------------------------
run_one() {
  local label="$1"; shift
  local tp="$1"; shift
  local outfile="$RESULTS/${label}_tp${tp}.out"
  echo "=== [$label] tp=$tp ==="
  echo "command: mpirun -n $tp ./build/main $*"
  set +e
  time mpirun -n "$tp" ./build/main "$@" > "$outfile" 2>&1
  local rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "FAILED rc=$rc (see $outfile)"
    tail -20 "$outfile" || true
  else
    # Echo the perf-relevant lines so they appear in the main log too.
    grep -E "throughput:|latency:|bandwidth:|hydrate:|perplexity:|total:" "$outfile" || true
  fi
  echo
}

# ---------------------------------------------------------------------------
# Scenario sweep. Each (scenario, tp) is one run.
# ---------------------------------------------------------------------------
# Reduced scenario set to fit the 30-min gpu-turing wall-time cap. Each
# (scenario, tp) ran takes ~10–40 s on Llama-3.2-3B + T4; 12 runs ≈ 5–8 min,
# leaving headroom for build + module loads + slow first-allocation startup.
for TP in $TP_SIZES; do
  echo "############################################################"
  echo "# Scenario 1: short prompt + 64-token decode (tp=$TP)"
  echo "#   primary decode-throughput signal"
  echo "############################################################"
  run_one s1_short_decode64 "$TP" \
    "$MODEL" -i "$PROMPT_SHORT" -n 64 -t "$TEMP" -s "$SEED" -p

  echo "############################################################"
  echo "# Scenario 2: long prompt + 32-token decode (tp=$TP)"
  echo "#   prefill (hydrate) throughput contribution"
  echo "############################################################"
  run_one s2_long_decode32 "$TP" \
    "$MODEL" -i "$PROMPT_LONG" -n 32 -t "$TEMP" -s "$SEED" -p

  echo "############################################################"
  echo "# Scenario 3: short prompt + 256-token decode (tp=$TP)"
  echo "#   decode scaling with kv_len"
  echo "############################################################"
  run_one s3_short_decode256 "$TP" \
    "$MODEL" -i "$PROMPT_SHORT" -n 256 -t "$TEMP" -s "$SEED" -p

  echo "############################################################"
  echo "# Scenario 4: perplexity, long prompt (tp=$TP)"
  echo "#   numerical-consistency check across tp + forward-only throughput"
  echo "############################################################"
  run_one s4_perplexity "$TP" \
    "$MODEL" -m perplexity -i "$PROMPT_LONG" -s "$SEED" -p

  # Passkey test omitted — it's a ~3k-token long-context run that
  # dominates the wall-time budget. Add it back with TP_SIZES="1 2" and
  # --time=00:30:00 if you want long-context scaling numbers.
done

# ---------------------------------------------------------------------------
# Summary table. Pulls the "throughput:" line from each run and prints a
# scaling table. Speedup column is computed against tp=1.
# ---------------------------------------------------------------------------
echo
echo "=========================================================="
echo "==              TP scaling summary                      =="
echo "=========================================================="

extract_throughput() {
  # Generation mode prints `throughput: X tok/s`; perplexity prints the same
  # but for forward-only tokens. We just grab the *last* throughput line in
  # the file (skips the encoding-stats line at the top).
  local f="$1"
  grep -E "^\s+throughput:" "$f" 2>/dev/null | tail -1 | sed -E 's/.*throughput:\s*([0-9.]+).*/\1/'
}

extract_bandwidth() {
  local f="$1"
  grep -E "^\s+bandwidth:" "$f" 2>/dev/null | tail -1 | sed -E 's/.*bandwidth:\s*([0-9.]+).*/\1/'
}

extract_perplexity() {
  local f="$1"
  grep -E "perplexity:" "$f" 2>/dev/null | tail -1 | sed -E 's/.*perplexity:\s*([0-9.]+).*/\1/'
}

printf "%-25s %-6s %12s %12s %10s\n" "scenario" "tp" "tok/s" "GB/s" "speedup"
printf "%-25s %-6s %12s %12s %10s\n" "--------" "--" "-----" "----" "-------"

for scenario in s1_short_decode64 s2_long_decode32 s3_short_decode256 s4_perplexity; do
  # Baseline throughput at tp=1 (first value in TP_SIZES list, assumed 1).
  base_file="$RESULTS/${scenario}_tp1.out"
  base_tps=$(extract_throughput "$base_file")
  for TP in $TP_SIZES; do
    f="$RESULTS/${scenario}_tp${TP}.out"
    if [ ! -f "$f" ]; then continue; fi
    tps=$(extract_throughput "$f")
    bw=$(extract_bandwidth "$f")
    if [ -n "$base_tps" ] && [ -n "$tps" ]; then
      speedup=$(awk "BEGIN{ printf \"%.2fx\", $tps / $base_tps }")
    else
      speedup="-"
    fi
    printf "%-25s %-6s %12s %12s %10s\n" "$scenario" "$TP" "${tps:-?}" "${bw:-?}" "$speedup"
  done
  echo
done

echo "==========================================================="
echo "== Perplexity consistency (numerical drift across tp)    =="
echo "== Should agree to ~1e-3 between tp=1 and tp=2/4.        =="
echo "==========================================================="
for TP in $TP_SIZES; do
  f="$RESULTS/s4_perplexity_tp${TP}.out"
  ppl=$(extract_perplexity "$f")
  printf "  tp=%-3s  perplexity = %s\n" "$TP" "${ppl:-?}"
done
echo

echo "=== Done ==="
date
echo "Full per-run logs in: $RESULTS/"
