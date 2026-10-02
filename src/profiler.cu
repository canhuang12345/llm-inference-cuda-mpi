#include "profiler.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>

#define PCUDA_CHECK(x)                                                        \
  do {                                                                        \
    cudaError_t err = (x);                                                    \
    if (err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "[profiler] CUDA error at %s:%d: %s\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(err));              \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

CudaProfiler::~CudaProfiler() {
  reset();
}

void CudaProfiler::reset() {
  for (auto& s : _samples) {
    if (s.begin) cudaEventDestroy(s.begin);
    if (s.end) cudaEventDestroy(s.end);
  }
  _samples.clear();
  _synced = false;
}

void CudaProfiler::begin_forward(cudaStream_t stream) {
  if (!_enabled) return;
  Sample s{};
  PCUDA_CHECK(cudaEventCreate(&s.begin));
  PCUDA_CHECK(cudaEventCreate(&s.end));
  s.is_hydrate = false;
  s.active_bytes = 0;
  s.elapsed_ms = 0.0f;
  PCUDA_CHECK(cudaEventRecord(s.begin, stream));
  _samples.push_back(s);
  _synced = false;
}

void CudaProfiler::end_forward(cudaStream_t stream, bool is_hydrate, std::size_t active_bytes) {
  if (!_enabled) return;
  if (_samples.empty()) return;
  Sample& s = _samples.back();
  s.is_hydrate = is_hydrate;
  s.active_bytes = active_bytes;
  PCUDA_CHECK(cudaEventRecord(s.end, stream));
}

void CudaProfiler::synchronize() {
  if (!_enabled || _samples.empty()) return;
  // Sync only on the last event — events are FIFO on the stream so this
  // implies all earlier events have also completed.
  PCUDA_CHECK(cudaEventSynchronize(_samples.back().end));
  for (auto& s : _samples) {
    PCUDA_CHECK(cudaEventElapsedTime(&s.elapsed_ms, s.begin, s.end));
  }
  _synced = true;
}

namespace {

struct Stats {
  std::size_t n = 0;
  double sum_ms = 0.0;
  double min_ms = std::numeric_limits<double>::infinity();
  double max_ms = 0.0;
  double sum_bytes = 0.0;

  void add(double ms, std::size_t bytes) {
    n += 1;
    sum_ms += ms;
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    sum_bytes += static_cast<double>(bytes);
  }

  double mean_ms() const { return n ? sum_ms / n : 0.0; }
  double tok_per_s() const { return sum_ms > 0.0 ? (n * 1000.0) / sum_ms : 0.0; }
  // Achieved HBM bandwidth assuming each forward reads `active_bytes` of weights+KV
  double gb_per_s() const { return sum_ms > 0.0 ? (sum_bytes / 1e9) / (sum_ms / 1000.0) : 0.0; }
};

void print_stats(const char* label, const Stats& s) {
  if (s.n == 0) {
    std::printf("  %-8s  (none)\n", label);
    return;
  }
  // We deliberately use printf here to keep the profiler dependency-light.
  std::printf(
      "  %-8s  n=%-5zu  mean=%7.3f ms  min=%7.3f ms  max=%7.3f ms  "
      "throughput=%7.2f tok/s  HBM=%6.1f GB/s\n",
      label, s.n, s.mean_ms(), s.min_ms, s.max_ms, s.tok_per_s(), s.gb_per_s());
}

}  // namespace

void CudaProfiler::report() const {
  if (!_enabled || _samples.empty()) return;
  if (!_synced) {
    std::fprintf(stderr,
                 "[profiler] report() called before synchronize() — numbers may be wrong\n");
  }

  Stats hydrate, decode, total;
  for (const auto& s : _samples) {
    (s.is_hydrate ? hydrate : decode).add(s.elapsed_ms, s.active_bytes);
    total.add(s.elapsed_ms, s.active_bytes);
  }

  std::printf("\n=== CUDA forward-pass profile ===\n");
  print_stats("hydrate", hydrate);
  print_stats("decode",  decode);
  print_stats("total",   total);
  std::printf(
      "  (HBM GB/s = active_bytes per forward × forwards / total GPU time;\n"
      "   compare against your card's peak — e.g. RTX 4090 ~1008 GB/s, T4 ~320 GB/s)\n");
}
