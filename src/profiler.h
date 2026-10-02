#pragma once

// Lightweight CUDA-event-based profiler for forward-pass timing.
//
// Why: wall-clock time around model.forward() is misleading because
// HYDRATE_KV_CACHE forwards return asynchronously (no cudaStreamSynchronize).
// CUDA events recorded on the stream give true GPU time even for async launches.
//
// Usage:
//   CudaProfiler prof;
//   prof.set_enabled(true);
//   ...
//   prof.begin_forward(stream);
//   model.forward(...);
//   prof.end_forward(stream, is_hydrate);
//   ...
//   prof.synchronize();
//   prof.report(*model.config);

#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

struct Config;

class CudaProfiler {
 public:
  CudaProfiler() = default;
  ~CudaProfiler();

  CudaProfiler(const CudaProfiler&) = delete;
  CudaProfiler& operator=(const CudaProfiler&) = delete;

  void set_enabled(bool enabled) { _enabled = enabled; }
  bool enabled() const { return _enabled; }

  // Record an event marking the start of a forward pass on `stream`.
  void begin_forward(cudaStream_t stream);
  // Record an event marking the end of a forward pass on `stream`.
  // `is_hydrate` separates prompt-prefill timings from decode-step timings.
  // `active_bytes` is the model's weights+KV traffic per step (used for HBM BW).
  void end_forward(cudaStream_t stream, bool is_hydrate, std::size_t active_bytes);

  // Wait for all recorded events to complete and compute elapsed times.
  // Must be called before report().
  void synchronize();

  // Print summary: per-phase mean/min/max latency, tokens/s, achieved HBM BW.
  void report() const;

  // Drop all recorded events (frees event handles).
  void reset();

 private:
  struct Sample {
    cudaEvent_t begin;
    cudaEvent_t end;
    bool is_hydrate;
    std::size_t active_bytes;
    float elapsed_ms;  // filled in by synchronize()
  };

  bool _enabled = false;
  bool _synced = false;
  std::vector<Sample> _samples;
};
