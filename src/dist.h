#pragma once

// Tensor-parallel distributed runtime: thin wrappers over MPI.
//
// The CUDA path in infer.cu and the model-loading path in model.cpp both call
// into these helpers. Kept process-global rather than threaded through
// constructors because rank/size are immutable for the lifetime of the
// program and pervade the model layout.
//
// CUDA-aware MPI is assumed (OpenMPI built against UCX with CUDA support, or
// equivalent). The OpenMPI bundled with the NVIDIA HPC SDK ships with it.
// MPI_Allreduce is invoked directly on device pointers — no host staging.

#include <mpi.h>

namespace dist {

// Initialize MPI. Safe to call multiple times (subsequent calls are no-ops).
// Must be called early in main() before any other dist:: function or any
// command-line parsing that would benefit from rank-0 gating.
void init(int* argc, char*** argv);

// Tear down MPI. Safe to call before exit even if init() was not called.
void finalize();

// True once init() has run successfully.
bool initialized();

// World rank and size (the TP group is currently the whole world).
int rank();
int size();

// Rank within the local node — used to pick which GPU this process binds to.
// Falls back to 0 when MPI is not initialized.
int local_rank();

// Communicator for collectives. MPI_COMM_WORLD for now; kept as a function so
// we can later split into TP / PP / DP sub-communicators without churning
// callers.
MPI_Comm comm();

// In-place sum-reduce over the TP communicator. Buffer is a device pointer
// (CUDA-aware MPI). count is in number of floats.
void allreduce_sum_f32(float* device_ptr, int count);

}  // namespace dist
