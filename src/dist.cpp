#include "dist.h"

#include <cstdio>
#include <cstdlib>

namespace dist {

namespace {
bool g_initialized = false;
int g_rank = 0;
int g_size = 1;
int g_local_rank = 0;
}  // namespace

void init(int* argc, char*** argv) {
  if (g_initialized) return;

  // FUNNELED: only the main thread will call MPI. Matches our usage —
  // collectives are issued from the same thread that drives the CUDA stream.
  int provided = 0;
  int rc = MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);
  if (rc != MPI_SUCCESS) {
    std::fprintf(stderr, "FATAL: MPI_Init_thread failed (rc=%d)\n", rc);
    std::abort();
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  // Local rank: split COMM_WORLD by shared-memory domain (one group per node)
  // and take this process's rank within that group. Drives GPU binding.
  MPI_Comm shmem;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank,
                      MPI_INFO_NULL, &shmem);
  MPI_Comm_rank(shmem, &g_local_rank);
  MPI_Comm_free(&shmem);

  g_initialized = true;
}

void finalize() {
  if (!g_initialized) return;
  MPI_Finalize();
  g_initialized = false;
}

bool initialized() { return g_initialized; }
int rank()         { return g_rank; }
int size()         { return g_size; }
int local_rank()   { return g_local_rank; }
MPI_Comm comm()    { return MPI_COMM_WORLD; }

void allreduce_sum_f32(float* device_ptr, int count) {
  // CUDA-aware MPI: pass the device pointer directly. Without UCX/CUDA the
  // call will still succeed if the pointer happens to be host-accessible, but
  // for correctness we require CUDA-awareness here.
  int rc = MPI_Allreduce(MPI_IN_PLACE, device_ptr, count, MPI_FLOAT, MPI_SUM,
                         MPI_COMM_WORLD);
  if (rc != MPI_SUCCESS) {
    std::fprintf(stderr, "FATAL: MPI_Allreduce failed (rc=%d)\n", rc);
    std::abort();
  }
}

}  // namespace dist
