#include "model.h"

#include <cuda_fp16.h>
#include "fmt/format.h"

#include <cfloat>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration of the MPI helper. dist.h pulls in <mpi.h> which we
// don't want to expose to nvcc; the symbol is linked from dist.cpp.
namespace dist {
void allreduce_sum_f32(float* device_ptr, int count);
}

#define FULL_MASK 0xffffffff

#define CUDA_CHECK(x)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

#define CUDA_CHECK2(x, msg)                                                                                    \
  do {                                                                                                 \
    cudaError_t err = x;                                                                             \
    if (err != cudaSuccess) {                                                                        \
      fprintf(stderr, "[%s] CUDA error in %s at %s:%d: %s (%s=%d)\n", msg.c_str(), __FUNCTION__, __FILE__, __LINE__, \
              cudaGetErrorString(err), cudaGetErrorName(err), err);                                \
      abort();                                                                                     \
    }                                                                                                \
  } while (0)

static void* cuda_devicecopy(void* host, size_t size) {
  void* device = NULL;
  CUDA_CHECK(cudaMalloc(&device, size));
  CUDA_CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice));
  return device;
}

static void* cuda_hostcopy(void* device, size_t size, std::string debug = "") {
  void* host = NULL;
  CUDA_CHECK2(cudaMallocHost(&host, size), debug);
  CUDA_CHECK2(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost), debug);
  return host;
}

[[maybe_unused]] static void* cuda_devicealloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

[[maybe_unused]] static void* cuda_hostalloc(size_t size) {
  void* ptr = NULL;
  CUDA_CHECK(cudaHostAlloc(&ptr, size, 0));
  return ptr;
}

extern "C" void* upload_cuda(void* host, size_t size) {
  return cuda_devicecopy(host, size);
}

extern "C" void* download_cuda(void* device, size_t size, std::string debug) {
  return cuda_hostcopy(device, size, debug);
}

extern "C" void register_cuda_host(void* host, size_t size) {
  CUDA_CHECK(cudaHostRegister(host, size, cudaHostRegisterDefault));
}

extern "C" void free_cuda(void* device) {
  CUDA_CHECK(cudaFree(device));
}

extern "C" void unregister_cuda_host(void* host) {
  CUDA_CHECK(cudaHostUnregister(host));
}

static int warp_size = 0;
static int max_threads_per_block = 0;

extern "C" void set_cuda_device(int device) {
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device));
  CUDA_CHECK(cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, device));
}

extern "C" void init_cuda_stream(cudaStream_t* stream) {
  CUDA_CHECK(cudaStreamCreate(stream));
}

#if DEBUG_MODEL
#include "fmt/format.h"
static std::map<std::string, DebugTensor> _debug_map;
std::map<std::string, DebugTensor>& debug_map_cuda() {
  return _debug_map;
}
template <typename T>
static std::vector<T> copy_debug_tensor(T* device, size_t numel) {
  T* host = (T*)cuda_hostcopy(device, numel * sizeof(T));
  std::vector<T> fv(host, host + numel);
  return fv;
}
template <typename T>
static void save_debug_tensor(const std::string& name, T* x, size_t size) {
  _debug_map[name] = DebugTensor(copy_debug_tensor<T>(x, size));
}
#endif

__device__ inline float blocktranspose(float v, float def) {
  // Performs block-and-warp transpose operation:
  //   For a block containing K warps where lane 0 contains val_k,
  //   this function returns:
  //   - For warp 0, lane K: val_k
  //   - For all other warps and lanes: def
  int lane = threadIdx.x % warpSize;
  int warp = threadIdx.x / warpSize;

  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float sm[32];
  if (lane == 0) sm[warp] = v;
  __syncthreads();

  return lane < blockDim.x / warpSize ? sm[lane] : def;
}

__device__ 
inline float warp_reduce_sum(float val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);

  return val;
}

__device__ 
inline float warp_all_reduce_max(float val) {
  // Max reduction across a warp.
  // All threads will contain the max of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val = max(val, __shfl_xor_sync(FULL_MASK, val, mask));
  }
  return val;
}

__device__ 
inline float block_all_reduce_max(float val) {
  // Max reduction across a 1-D block implemented as double warp max reduction.
  // All threads will contain the max of all threads in the block.
  
  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_max(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -FLT_MAX;
  }
  val = warp_all_reduce_max(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__device__ 
inline float warp_all_reduce_sum(float val) {
  // Sum reduction across a warp.
  // All threads will contain the sum of all threads in the warp.
  for (int mask = warpSize/2; mask > 0; mask /= 2) {
    val += __shfl_xor_sync(FULL_MASK, val, mask);
  }
  return val;
}

__device__ 
inline float block_all_reduce_sum(float val) {
  // Sum reduction across a 1-D block implemented as double warp sum reduction.
  // All threads will contain the sum of all threads in the block.
  
  // Will hold results of all warps.
  // Capacity 32 since there can be at most 32 warps in a block.
  __shared__ float shared[32];
  const int wid  = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;

  val = warp_all_reduce_sum(val);

  if (blockDim.x < warpSize) return val;
  if (lane == 0) shared[wid] = val;

  __syncthreads();

  if ( wid == 0 ) {
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;
  }
  val = warp_all_reduce_sum(val);
  if (lane == 0) shared[wid] = val;
  
  __syncthreads();
  
  return shared[0];
}

__device__
inline float matmul_row(const float* row, const float* x, int offset, int dim) {
  float sum = 0.0;
  for (int j = offset; j < dim; j += warpSize) {
    float v = row[j] * x[j];
    sum += v;
  }
  return warp_reduce_sum(sum);
}

__device__
inline float matmul_row(const half* row, const float* x, int offset, int dim) {
  // half2-vectorized dot product.
  // Each lane processes pairs of halves at a time via 32-bit half2 loads,
  // halving the number of weight memory instructions vs scalar half loads.
  // Lane `offset` covers element pairs at positions:
  //   2*offset, 2*offset+1, 2*offset+2*warpSize, 2*offset+2*warpSize+1, ...
  // Each iteration issues one 32-bit weight load and one 64-bit x load.
  // PRECOND: weight row is 4-byte aligned (true for cudaMalloc'd FP16 arrays
  // as long as `dim` is even — which is always the case for transformer dims).
  float sum = 0.0f;
  int dim_pair = dim >> 1;
  for (int j = offset; j < dim_pair; j += warpSize) {
    half2 w_pair  = *((const half2*)&row[2 * j]);
    float2 w_f    = __half22float2(w_pair);
    float2 x_pair = *((const float2*)&x[2 * j]);
    sum += w_f.x * x_pair.x + w_f.y * x_pair.y;
  }
  // Scalar tail if dim is odd. Lane 0 handles it.
  if ((dim & 1) && offset == 0) {
    sum += __half2float(row[dim - 1]) * x[dim - 1];
  }
  return warp_reduce_sum(sum);
}

template <typename T>
__global__
void matmul(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Since block is 1-dimensional, thread ID is same as threadIdx.x,
  // and warp partitions thread IDs
  int offset = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, offset, n);
  if (offset == 0) {
    out[i] = rowSum;
  }
}

template <typename T>
__global__
void matmul_wide(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] = rowSum;
  }
}

// In-place add: x[i] += y[i]. Used in the TP path to fold the all-reduced
// row-parallel matmul output (wo or w2) back into the residual stream after
// the cross-rank reduction. Cannot fuse with matmul like
// fused_matmul_add_residuals does because the all-reduce sits in between.
__global__
void add_inplace(float* x, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

template <typename T>
__global__
void fused_matmul_add_residuals(const T* A, const float* x, int n, int d, float* out) {
  // A (d,n) @ x (n,) -> out (d,)
  // PRECOND: Block is 1-D and contains WPB warps.
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (i >= d) return;
  // Warp j computes sum for row at <blockIdx.x*WPB + j>
  // Lane 0 of each warp will hold result
  int k = threadIdx.x % warpSize;
  float rowSum = matmul_row(&A[n * i], x, k, n);
  // Transpose values so lane k in warp 0 contains row at <blockIdx.x*WPB + k>
  // For WPB=32, this allows us to coalesce 32 float32 writes into a single 128-byte store
  rowSum = blocktranspose(rowSum, 1.0);
  if (threadIdx.x < blockDim.x / warpSize) {
    int block_start_i = blockIdx.x * blockDim.x / warpSize;
    out[block_start_i + k] += rowSum;
  }
}

template <typename T>
__global__
void fused_qkv_matmul_clip(
  const T* wq,      // (q_dim, dim)
  const T* wk,      // (kv_dim, dim)
  const T* wv,      // (kv_dim, dim)
  const float* x,   // (dim,)
  int dim,          // input dimension
  int q_dim,        // n_heads * head_dim
  int kv_dim,       // n_kv_heads * head_dim
  float qkv_clip,   // clipping value
  float* q_out,     // (q_dim,)
  float* k_out,     // (kv_dim,)
  float* v_out      // (kv_dim,)
) {
  // Each warp handles one row of either Q, K, or V output
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int total_rows = q_dim + 2 * kv_dim;
  if (warp_id >= total_rows) return;
  
  // Determine which matrix (Q, K, or V) we're computing
  const T* w;
  float* out;
  if (warp_id < q_dim) {
    // Computing Q
    w = wq + warp_id * dim;
    out = q_out + warp_id;
  } else if (warp_id < q_dim + kv_dim) {
    // Computing K
    w = wk + (warp_id - q_dim) * dim;
    out = k_out + (warp_id - q_dim);
  } else {
    // Computing V
    w = wv + (warp_id - q_dim - kv_dim) * dim;
    out = v_out + (warp_id - q_dim - kv_dim);
  }

  // Compute matrix multiplication for this row
  // Since block is 1-dimensional, thread ID is same as threadIdx.x,
  // and warp partitions thread IDs
  int offset = threadIdx.x % warpSize;
  float row_sum = matmul_row(w, x, offset, dim);
  // Write result with clipping
  if (offset == 0) {
    row_sum = row_sum < -qkv_clip ? -qkv_clip : (row_sum > qkv_clip ? qkv_clip : row_sum);
    *out = row_sum;
  }
}

__global__
void attn_dot(
  const half* kb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* q,   // (n_heads, head_dim)
  int head_dim, 
  int kv_len, 
  int max_seq_len, 
  int n_heads, 
  int n_kv_heads,
  float* out        // (n_heads, kv_len)
) {
  int group = blockIdx.y;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  if (t >= kv_len || h >= n_heads) return;
  
  const float* query = q + h * head_dim;
  const half* key = kb + n_kv_heads * head_dim * t + head_dim * group;
  float score = 0.0;
  for (int i = 0; i < head_dim; i++) {
    score += query[i] * __half2float(key[i]);
  }
  out[h * max_seq_len + t] = score / sqrtf((float)head_dim);
}

__global__
void attn_softmax(
  const float* att, 
  int seq_len, 
  int max_seq_len, 
  int n_heads, 
  float* out
) {
  int offset = threadIdx.x;
  int h = blockIdx.x;
  int block_size = blockDim.x;
  if (h >= n_heads) return;
  
  const float* atth = att + max_seq_len * h;
  float* outh = out + max_seq_len * h;
  
  float score_max = -FLT_MAX;
  for (int t = offset; t < seq_len; t += block_size) {
    if (atth[t] > score_max) {
      score_max = atth[t];
    }
  }
  score_max = block_all_reduce_max(score_max);
  float score_sum = 0.0f;
  for (int t = offset; t < seq_len; t += block_size) {
    outh[t] = expf(atth[t] - score_max);
    score_sum += outh[t];
  }
  score_sum = block_all_reduce_sum(score_sum);
  for (int t = offset; t < seq_len; t += block_size) {
    outh[t] /= score_sum;
  }
}

__global__
void att_mix(
  const half* vb,  // (max_seq_len, n_kv_heads, head_dim) 
  const float* att, // (n_heads, kv_len)
  int head_dim, 
  int n_heads, 
  int n_kv_heads,
  int seq_len, 
  int max_seq_len, 
  float* out // (n_heads, head_dim)
) {
  // PRECOND: blocks are 2-D (warp_size, t_stride)
  int h = blockIdx.x;
  int group_size = n_heads / n_kv_heads;
  int g = h / group_size;
  int kv_stride = n_kv_heads * head_dim;
  
  const float* atth = att + max_seq_len * h;
  const half* vh = vb + head_dim * g;
  float* outh = out + head_dim * h;
  
  int warp_id = threadIdx.y;
  int t_stride = blockDim.y;
  
  // Each lane of the warp accumulates across 2 head elements at a time.
  // NOTE: Assumes warpSize is 32
  __shared__ float shared0[32]; // shared0[i] == chunk[2*i]
  __shared__ float shared1[32]; // shared1[i] == chunk[2*i+1]
  
  for (int i = 2*threadIdx.x; i < head_dim; i += 2*warpSize) {
    if (warp_id == 0) {
      shared0[threadIdx.x] = 0;
      shared1[threadIdx.x] = 0;
    }
    __syncthreads();
    float2 sum01 = make_float2(0.0, 0.0);
    constexpr int UNROLL = 16;
    half2 v01_0; float att_0; 
    half2 v01_1; float att_1; 
    half2 v01_2; float att_2; 
    half2 v01_3; float att_3;
    half2 v01_4; float att_4;
    half2 v01_5; float att_5;
    half2 v01_6; float att_6;
    half2 v01_7; float att_7;
    half2 v01_8; float att_8; 
    half2 v01_9; float att_9; 
    half2 v01_10; float att_10; 
    half2 v01_11; float att_11;
    half2 v01_12; float att_12;
    half2 v01_13; float att_13;
    half2 v01_14; float att_14;
    half2 v01_15; float att_15;
    int t = warp_id;
    for (int ctr = 0; ctr < seq_len / t_stride - UNROLL + 1; t += t_stride, ctr++) {
      int ctr_mod = ctr % UNROLL;
      if (ctr_mod == 0) {
        // prefetch every UNROLL iterations
        #define PREFETCH(j) \
          v01_##j = *((half2*)&vh[kv_stride * (t + j*t_stride) + i]); \
          att_##j = atth[t + j*t_stride];
        PREFETCH(0)
        PREFETCH(1)
        PREFETCH(2)
        PREFETCH(3)
        PREFETCH(4)
        PREFETCH(5)
        PREFETCH(6)
        PREFETCH(7)
        PREFETCH(8)
        PREFETCH(9)
        PREFETCH(10)
        PREFETCH(11)
        PREFETCH(12)
        PREFETCH(13)
        PREFETCH(14)
        PREFETCH(15)
        #undef PREFETCH
      }
      // pull one value out of prefetch batch
      float2 v01;
      float att_t;
      switch (ctr_mod) {
        #define CASE(j) \
          case j: v01 = __half22float2(v01_##j); att_t = att_##j; break;
        CASE(0)
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        CASE(9)
        CASE(10)
        CASE(11)
        CASE(12)
        CASE(13)
        CASE(14)
        CASE(15)
        #undef CASE
      }
      // Sadly CUDA does not have float2 SIMD ops
      sum01.x += v01.x * att_t;
      sum01.y += v01.y * att_t;
    }
    for (; t < seq_len; t += t_stride) {
      float2 v01 = __half22float2(*((half2*)&vh[kv_stride * t + i]));
      float att_t = atth[t];
      // Sadly CUDA does not have float2 SIMD ops
      sum01.x += v01.x * att_t;
      sum01.y += v01.y * att_t;
    }
    atomicAdd(&shared0[threadIdx.x], sum01.x);
    atomicAdd(&shared1[threadIdx.x], sum01.y);
    __syncthreads();
    if (warp_id == 0) {
      float even = shared0[threadIdx.x];
      float odd = shared1[threadIdx.x];
      *((float2*)&outh[i]) = make_float2(even, odd);
      shared0[threadIdx.x] = 0;
      shared1[threadIdx.x] = 0;
    }
  }
}

// FlashAttention for autoregressive decode.
//
// Single Q vector per head attends over kv_len cached K/V vectors. The full
// attention pipeline (Q·K -> softmax -> A·V) is fused into one kernel using
// the online softmax trick, eliminating the n_heads x kv_len score matrix
// in HBM.
//
// Launch geometry:
//   grid:  (n_heads,)
//   block: (warpSize,)             - one warp per head
//   smem:  (2 * head_dim + warpSize) * sizeof(float)
//
// Shared memory layout per block:
//   q_sm[head_dim]    - Q vector
//   o_sm[head_dim]    - online output accumulator
//   scores[warpSize]  - per-tile softmax weights (broadcast across lanes)
__global__
void flash_attention_decode(
  const half* kb,           // (max_pages, KV_PAGE_SIZE, n_kv_heads, head_dim)
  const half* vb,           // (max_pages, KV_PAGE_SIZE, n_kv_heads, head_dim)
  const float* q,           // (n_heads, head_dim)
  int head_dim,
  int kv_len,
  int max_seq_len,          // unused; kept for kernel-signature symmetry
  int n_heads,
  int n_kv_heads,
  float* out,               // (n_heads, head_dim)
  const int* page_table     // (max_pages,) logical -> physical page id
) {
  int h = blockIdx.x;
  if (h >= n_heads) return;

  int group_size = n_heads / n_kv_heads;
  int kv_head = h / group_size;
  int kv_stride = n_kv_heads * head_dim;

  int lane = threadIdx.x;

  extern __shared__ float smem[];
  float* q_sm   = smem;
  float* o_sm   = q_sm + head_dim;
  float* scores = o_sm + head_dim;

  // Load Q into shared memory and zero O.
  for (int i = lane; i < head_dim; i += warpSize) {
    q_sm[i] = q[h * head_dim + i];
    o_sm[i] = 0.0f;
  }
  __syncwarp();

  // Online softmax running state.
  float m_max = -FLT_MAX;
  float l_sum = 0.0f;

  const float inv_sqrt_d = rsqrtf((float)head_dim);

  // Process kv_len tokens in tiles of warpSize.
  for (int tile_start = 0; tile_start < kv_len; tile_start += warpSize) {
    int t = tile_start + lane;
    int tile_size = min((int)warpSize, kv_len - tile_start);

    // Each lane computes one Q·K[t] dot product.
    // Uses half2-vectorized K reads + float2 Q reads — half the memory
    // instructions vs scalar (head_dim is always even for transformer models).
    // K is fetched via the page table: logical t -> physical slot.
    float score = -FLT_MAX;
    if (lane < tile_size) {
      int t_page = t / KV_PAGE_SIZE;
      int t_slot = t - t_page * KV_PAGE_SIZE;
      int phys_t = page_table[t_page] * KV_PAGE_SIZE + t_slot;
      const half* k_row = kb + phys_t * kv_stride + kv_head * head_dim;
      float s = 0.0f;
      int head_dim_pair = head_dim >> 1;
      for (int i = 0; i < head_dim_pair; i++) {
        half2 k_pair  = *((const half2*)&k_row[2 * i]);
        float2 k_f    = __half22float2(k_pair);
        float2 q_pair = *((const float2*)&q_sm[2 * i]);
        s += k_f.x * q_pair.x + k_f.y * q_pair.y;
      }
      score = s * inv_sqrt_d;
    }

    // Online softmax: update running max, rescale running sum and O, accumulate.
    float tile_max  = warp_all_reduce_max(score);
    float new_max   = max(m_max, tile_max);
    float exp_score = (lane < tile_size) ? __expf(score - new_max) : 0.0f;
    float tile_sum  = warp_all_reduce_sum(exp_score);
    float correction = __expf(m_max - new_max);
    l_sum = l_sum * correction + tile_sum;
    m_max = new_max;

    // Stash per-lane exp_score so all lanes can read them when mixing V.
    scores[lane] = exp_score;
    __syncwarp();

    // Update O: O[i] = O[i] * correction + Σ_t scores[t] * V[t][i].
    // Pair-stride: lane t owns head_dim pairs (2t, 2t+1, 2t+2*warpSize, ...).
    // Each iteration does one half2 V load and one float2 O accumulator load.
    // V is fetched via the page table per t.
    int head_dim_pair = head_dim >> 1;
    for (int i_pair = lane; i_pair < head_dim_pair; i_pair += warpSize) {
      int i = 2 * i_pair;
      float2 acc;
      acc.x = o_sm[i]     * correction;
      acc.y = o_sm[i + 1] * correction;
      for (int t_idx = 0; t_idx < tile_size; t_idx++) {
        int t_pos  = tile_start + t_idx;
        int t_page = t_pos / KV_PAGE_SIZE;
        int t_slot = t_pos - t_page * KV_PAGE_SIZE;
        int phys_t = page_table[t_page] * KV_PAGE_SIZE + t_slot;
        const half* v_row = vb + phys_t * kv_stride + kv_head * head_dim;
        half2 v_pair = *((const half2*)&v_row[i]);
        float2 v_f   = __half22float2(v_pair);
        acc.x += scores[t_idx] * v_f.x;
        acc.y += scores[t_idx] * v_f.y;
      }
      *((float2*)&o_sm[i]) = acc;
    }
    __syncwarp();
  }

  // Final normalization: out = O / l_sum
  float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
  for (int i = lane; i < head_dim; i += warpSize) {
    out[h * head_dim + i] = o_sm[i] * inv_l;
  }
}

__global__
void rmsnorm(const float* x, const float* weight, int size, float eps, float* out) {
  // PRECOND: only one 1-D block is launched.
  //
  // When `size` is a multiple of 4 and pointers are 16B-aligned (always true for
  // tensors allocated via cudaMalloc), we use 128-bit float4 loads/stores to halve
  // the number of memory instructions in both passes.
  int offset = threadIdx.x;
  int block_dim = blockDim.x;
  float rms = 0.0f;

  bool vec4_ok =
      (size % 4 == 0) &&
      (reinterpret_cast<uintptr_t>(x) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(weight) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(out) % 16 == 0);

  if (vec4_ok) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int size4 = size / 4;
    for (int i = offset; i < size4; i += block_dim) {
      float4 v = x4[i];
      rms += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    rms = block_all_reduce_sum(rms);
    float scale = rsqrtf(rms / size + eps);

    const float4* w4 = reinterpret_cast<const float4*>(weight);
    float4* out4 = reinterpret_cast<float4*>(out);
    for (int i = offset; i < size4; i += block_dim) {
      float4 v = x4[i];
      float4 w = w4[i];
      float4 r;
      r.x = v.x * scale * w.x;
      r.y = v.y * scale * w.y;
      r.z = v.z * scale * w.z;
      r.w = v.w * scale * w.w;
      out4[i] = r;
    }
  } else {
    for (int i = offset; i < size; i += block_dim) {
      rms += x[i] * x[i];
    }
    rms = block_all_reduce_sum(rms);
    float scale = rsqrtf(rms / size + eps);
    for (int i = offset; i < size; i += block_dim) {
      out[i] = x[i] * scale * weight[i];
    }
  }
}

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, float* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    // __sincosf computes sin+cos in one SFU instruction — about 2x faster
    // than separate cosf/sinf calls and still fast-math-accurate enough for RoPE.
    float fcr, fci;
    __sincosf(val, &fci, &fcr);

    float2 v01 = *((float2*)&x[pair_idx]);
    float2 result = make_float2(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((float2*)&out[pair_idx]) = result;
  }
}

__device__
inline void rope(
  const float* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr, fci;
    __sincosf(val, &fci, &fcr);

    float2 v01 = *((float2*)&x[pair_idx]);
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
  }
}

__device__
inline void rope(
  const half* x, int pair_idx, int head_dim, int pos, float theta, int rotary_dim, half* out
) {
  int j_head = pair_idx % head_dim;
  if (j_head < head_dim - 1) {  // Ensure we have a pair of elements
    float freq = j_head >= rotary_dim ? 0.f : 1.0f / powf(theta, (float)j_head / (float)rotary_dim);
    float val = pos * freq;
    float fcr, fci;
    __sincosf(val, &fci, &fcr);

    float2 v01 = __half22float2(*((half2*)&x[pair_idx]));
    half2 result = __floats2half2_rn(
      v01.x * fcr - v01.y * fci,
      v01.x * fci + v01.y * fcr
    );
    *((half2*)&out[pair_idx]) = result;
  }
}

template <ActivationType A> __device__ inline float act(float x);
template<> __device__ inline float act<ActivationType::SILU>(float x) {
  return x / (1.0f + expf(-x));
}
template<> __device__ inline float act<ActivationType::GELU>(float x) {
  float x3 = x * x * x;
  return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x3)));
}

template <typename T, ActivationType A>
__global__
void fused_ffn_w1_w3_glu_act(
  const T* w1,        // (hidden_dim, dim)
  const T* w3,        // (hidden_dim, dim)
  const float* x,     // (dim,)
  int dim,           
  int hidden_dim,
  float* out         // (hidden_dim,)
) {
  // Each warp computes one row of both w1(x) and w3(x), then applies GLU
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  if (warp_id >= hidden_dim) return;
  
  int offset = threadIdx.x % warpSize;
  
  // Compute w1(x) and w3(x) for this row
  float sum1 = matmul_row(&w1[dim * warp_id], x, offset, dim);
  float sum3 = matmul_row(&w3[dim * warp_id], x, offset, dim);
  
  // Apply activation and multiply
  if (offset == 0) {
    out[warp_id] = act<A>(sum1) * sum3;
  }
}

__global__
void copy_embedding_float(
  const float* token_embedding_table, int dim, int token, float* out
) {
  // PRECOND: grid and blocks are 1-D
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= dim) return;
  
  const float* v = token_embedding_table + dim * token;
  out[i] = v[i];
}

__global__
void copy_embedding_half(
  const half* token_embedding_table, int dim, int token, float* out
) {
  // PRECOND: grid and blocks are 1-D
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= dim) return;
  
  const half* v = token_embedding_table + dim * token;
  out[i] = __half2float(v[i]);
}

__global__
void fused_rope_and_cache_update(
  const float* q,         // (n_heads * head_dim,)
  const float* k,         // (n_kv_heads * head_dim,)
  const float* v,         // (n_kv_heads * head_dim,)
  int head_dim,
  int n_heads,
  int n_kv_heads,
  int pos,                // current position
  int kv_pos,             // logical position in KV cache (ring-buffer adjusted)
  float theta,            // RoPE theta parameter
  int rotary_dim,         // how many dimensions to rotate
  float* q_out,           // (n_heads * head_dim,)
  half* kb,               // (max_pages, KV_PAGE_SIZE, n_kv_heads, head_dim)
  half* vb,               // (max_pages, KV_PAGE_SIZE, n_kv_heads, head_dim)
  const int* page_table   // (max_pages,) logical -> physical page id
) {
  // Resolve logical kv_pos through the page table to a physical storage slot.
  // Done once per kernel launch — broadcast through all threads via the kernel
  // launch parameter. Each thread does an L1-hot indirect load on page_table.
  int kv_page  = kv_pos / KV_PAGE_SIZE;
  int kv_slot  = kv_pos - kv_page * KV_PAGE_SIZE;
  int phys_kv  = page_table[kv_page] * KV_PAGE_SIZE + kv_slot;

  // Each thread handles two consecutive elements (for RoPE complex rotation)
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;

  // Handle Q matrix RoPE (Q is not cached, no page table needed)
  if (pair_idx < n_heads * head_dim) {
    rope(
      q, pair_idx, head_dim, pos,
      theta, rotary_dim, q_out
    );
  }

  // Handle K matrix RoPE and cache update — write to physical slot
  if (pair_idx < n_kv_heads * head_dim) {
    half* k_out = &kb[phys_kv * (n_kv_heads * head_dim)];
    rope(
      k, pair_idx, head_dim, pos,
      theta, rotary_dim, k_out
    );
  }

  // Handle V cache update (no RoPE needed)
  if (pair_idx < n_kv_heads * head_dim) {
    int cache_idx = phys_kv * (n_kv_heads * head_dim) + pair_idx;
    if (pair_idx < n_kv_heads * head_dim - 1) {
      vb[cache_idx]     = __float2half(v[pair_idx]);
      vb[cache_idx + 1] = __float2half(v[pair_idx + 1]);
    }
  }
}

__global__
void rotate_sink_tokens(
  half* kb, 
  int kv_sink, 				// number of attention sinks
  int kv_dim, 				// size of each entry (all concatenated heads) in KV cache
  int head_dim,
  float theta, 				// RoPE theta parameter
  int rotary_dim			// how many dimensions to rotate
) {
  if (kv_sink == 0) return;
  // Each thread handles two consecutive elements (for RoPE complex rotation)
  // across all attention sinks
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int pair_idx = tid * 2;
  
  if (pair_idx < kv_dim) {
    for (int r = 0; r < kv_sink; r++) {
      half* k = kb + r * kv_dim;
      rope(k, pair_idx, head_dim, 1, theta, rotary_dim, k);
    }
  }
}

template <typename T>
void Block::_block_cuda(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
#define STATIC_KERNEL(x) if (!s.graph().is_created) x;
  const Config& c = *_config;
  
  // attention pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb()
      )));
      break;
    }
  }
  
  int q_dim = c.n_heads * c.head_dim;
  int kv_dim = c.n_kv_heads * c.head_dim;

  {
    // qkv matmuls for this position
    // some models require clipping qkv values
    int total_rows = q_dim + 2 * kv_dim;  // Total rows across Q, K, V
    STATIC_KERNEL((fused_qkv_matmul_clip<<<total_rows, warp_size, 0, s.stream()>>>(
      wq<T>(),
      wk<T>(),
      wv<T>(),
      s.xb(),
      c.dim,
      q_dim,
      kv_dim,
      c.qkv_clip,
      s.q(),
      s.k(),
      s.v()
    )));
  }
  
  // Update Q, K with RoPE relative positional encoding: 
  // complex-valued rotate q and k in each head
  // Also copy K, V to KV cache
  half* kb = (half*)key_cache();
  half* vb = (half*)value_cache();
  {
    // Calculate number of thread blocks needed
    // We need enough threads to handle the largest of:
    // - n_heads * head_dim (for Q)
    // - n_kv_heads * head_dim (for K and V)
    int max_dim = max(c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim);
    int threads_needed = (max_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;

    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>(num_blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void*>(fused_rope_and_cache_update);
    float* q = s.q();
    float* k = s.k();
    float* v = s.v();
    int* pt = _page_table;
    void* kernelParams[] = {
      &q,
      &k,
      &v,
      (void*)&c.head_dim,
      (void*)&c.n_heads,
      (void*)&c.n_kv_heads,
      &pos,
      &kv_pos,
      (void*)&c.rope_theta,
      (void*)&c.rotary_dim,
      &q,           // Q can be updated in-place
      &kb,
      &vb,
      &pt
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:fused_rope_and_cache_update", _layer_i), params, s.stream());
  }
  {
    // Sink tokens remain untouched while the rest of the KV cache is incrementally 
    // replaced in ring order, but sink i must always be positioned (max_seq_len - i)
    // away from current timestep. Hence, each forward pass, rotate any sink tokens 
    // forward by 1. See https://arxiv.org/abs/2309.17453 for more.
    int threads_needed = (kv_dim + 1) / 2;  // Each thread handles 2 elements
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>(num_blocks), 1, 1};
    params.sharedMemBytes = 0;
    params.func = reinterpret_cast<void*>(rotate_sink_tokens);
    void* kernelParams[] = {
      &kb,
      &kv_sink,
      &kv_dim,
      (void*)&c.head_dim,
      (void*)&c.rope_theta,
      (void*)&c.rotary_dim
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:rotate_sink_tokens", _layer_i), params, s.stream());
  }
  
  // FlashAttention: fuses QK / softmax / AV into one kernel. Replaces the
  // attn_dot + attn_softmax + att_mix trio. Score matrix never hits HBM.
  //
  // Layout:
  //   grid (n_heads,)  block (warpSize,)
  //   smem = (2*head_dim + warpSize) floats
  {
    int shared_bytes = (2 * c.head_dim + warp_size) * (int)sizeof(float);

    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(warp_size), 1, 1};
    params.gridDim = {static_cast<unsigned int>(c.n_heads), 1, 1};
    params.sharedMemBytes = static_cast<unsigned int>(shared_bytes);
    params.func = reinterpret_cast<void*>(flash_attention_decode);
    float* q = s.q();
    float* xb2 = s.xb2();
    int* pt = _page_table;
    void* kernelParams[] = {
      &kb,
      &vb,
      &q,
      (void*)&c.head_dim,
      &kv_len,
      (void*)&c.max_seq_len,
      (void*)&c.n_heads,
      (void*)&c.n_kv_heads,
      &xb2,
      &pt
    };
    params.kernelParams = kernelParams;
    params.extra = nullptr;
    s.graph().add_or_update_kernel_node(fmt::format("{}:flash_attention", _layer_i), params, s.stream());
  }

  // final matmul projection and residual back:
  // x <- wo(...) + x
  STATIC_KERNEL((fused_matmul_add_residuals<<<c.dim/32, warp_size*32, 0, s.stream()>>>(
    wo<T>(), s.xb2(), q_dim, c.dim, s.x()
  )));
  
  // ffn pre-norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb()
      )));
      break;
    }
  }

  if (c.n_experts > 0) {
    assert(false && "Mixture of experts not yet supported for CUDA");
  }
  
  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  switch (c.act) {
    case ActivationType::GELU: {
      STATIC_KERNEL((fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        c.hidden_dim, warp_size, 0, s.stream()
      >>>(
        w1<T>(), w3<T>(), s.xb(), c.dim, c.hidden_dim, s.hb()
      )));
      break;
    }
    case ActivationType::SILU: {
      STATIC_KERNEL((fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        c.hidden_dim, warp_size, 0, s.stream()
      >>>(
        w1<T>(), w3<T>(), s.xb(), c.dim, c.hidden_dim, s.hb()
      )));
      break;
    }
  }
  
  // add residual back: x <- w2(...) + x
  STATIC_KERNEL((fused_matmul_add_residuals<<<c.dim/32, warp_size*32, 0, s.stream()>>>(
    w2<T>(), s.hb(), c.hidden_dim, c.dim, s.x()
  )));
#undef STATIC_KERNEL
}

// Tensor-parallel block. Same forward sequence as _block_cuda but:
//   - All matmul launch dims use the rank-local Q/KV/hidden sizes.
//   - Sink rotate and FlashAttention pass local head counts.
//   - wo and w2 (row-parallel matmuls) split into matmul-into-tp_xpart,
//     MPI_Allreduce, then add-residual. The fused_matmul_add_residuals path
//     can't be used because the all-reduce sits between the matmul and the
//     residual add.
//   - No CUDA graph capture. MPI calls aren't graph-capturable, and the
//     per-step kernel launch overhead is small compared to the inter-rank
//     reduction at TP > 1.
template <typename T>
void Block::_block_cuda_dist(
  InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len
) const {
  const Config& c = *_config;
  cudaStream_t stream = s.stream();
  const int q_dim_l      = c.q_dim_local;
  const int kv_dim_l     = c.kv_dim_local;
  const int hidden_l     = c.hidden_dim_local;
  const int n_heads_l    = c.n_heads_local;
  const int n_kv_heads_l = c.n_kv_heads_local;

  // attention pre-norm — input x is identical on every rank (replicated
  // residual stream), so RMSNorm runs locally with no comm.
  rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
    s.x(), rms_att_weight(), c.dim, c.norm_eps, s.xb()
  );

  // qkv matmuls on the local head slice.
  {
    int total_rows = q_dim_l + 2 * kv_dim_l;
    fused_qkv_matmul_clip<<<total_rows, warp_size, 0, stream>>>(
      wq<T>(), wk<T>(), wv<T>(), s.xb(),
      c.dim, q_dim_l, kv_dim_l, c.qkv_clip,
      s.q(), s.k(), s.v()
    );
  }

  // RoPE + write into the local KV cache slice.
  half* kb = (half*)key_cache();
  half* vb = (half*)value_cache();
  {
    int max_dim = max(q_dim_l, kv_dim_l);
    int threads_needed = (max_dim + 1) / 2;
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    fused_rope_and_cache_update<<<num_blocks, max_threads_per_block, 0, stream>>>(
      s.q(), s.k(), s.v(),
      c.head_dim, n_heads_l, n_kv_heads_l,
      pos, kv_pos, c.rope_theta, c.rotary_dim,
      s.q(), kb, vb, _page_table
    );
  }
  {
    int threads_needed = (kv_dim_l + 1) / 2;
    int num_blocks = (threads_needed + max_threads_per_block - 1) / max_threads_per_block;
    rotate_sink_tokens<<<num_blocks, max_threads_per_block, 0, stream>>>(
      kb, kv_sink, kv_dim_l, c.head_dim, c.rope_theta, c.rotary_dim
    );
  }

  // FlashAttention over the local heads only — every Q head on this rank
  // owns its K/V slice (the KV-head split lines up with the Q-head split).
  {
    int shared_bytes = (2 * c.head_dim + warp_size) * (int)sizeof(float);
    flash_attention_decode<<<n_heads_l, warp_size, shared_bytes, stream>>>(
      kb, vb, s.q(),
      c.head_dim, kv_len, c.max_seq_len, n_heads_l, n_kv_heads_l,
      s.xb2(), _page_table
    );
  }

  // wo row-parallel: matmul -> all-reduce -> add residual.
  // Local input dim is q_dim_l; output dim is full c.dim.
  matmul_wide<<<c.dim/32, warp_size*32, 0, stream>>>(
    wo<T>(), s.xb2(), q_dim_l, c.dim, s.tp_xpart()
  );
  // MPI_Allreduce needs the kernel to be done. CUDA-aware MPI does not
  // synchronize with the CUDA stream; we sync explicitly here.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dist::allreduce_sum_f32(s.tp_xpart(), c.dim);
  add_inplace<<<(c.dim + 255)/256, 256, 0, stream>>>(s.x(), s.tp_xpart(), c.dim);

  // ffn pre-norm — same situation as attn pre-norm; x is identical post-reduce.
  rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
    s.x(), rms_ffn_weight(), c.dim, c.norm_eps, s.xb()
  );

  if (c.n_experts > 0) {
    assert(false && "Mixture of experts not yet supported in TP path");
  }

  // GLU FFN on the local hidden slice.
  switch (c.act) {
    case ActivationType::GELU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        hidden_l, warp_size, 0, stream
      >>>(w1<T>(), w3<T>(), s.xb(), c.dim, hidden_l, s.hb());
      break;
    }
    case ActivationType::SILU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        hidden_l, warp_size, 0, stream
      >>>(w1<T>(), w3<T>(), s.xb(), c.dim, hidden_l, s.hb());
      break;
    }
  }

  // w2 row-parallel: same matmul -> all-reduce -> add residual pattern.
  matmul_wide<<<c.dim/32, warp_size*32, 0, stream>>>(
    w2<T>(), s.hb(), hidden_l, c.dim, s.tp_xpart()
  );
  CUDA_CHECK(cudaStreamSynchronize(stream));
  dist::allreduce_sum_f32(s.tp_xpart(), c.dim);
  add_inplace<<<(c.dim + 255)/256, 256, 0, stream>>>(s.x(), s.tp_xpart(), c.dim);
}

void mha_cuda(
  float* xout,  // (n_heads, head_dim)
  float* att,   // (n_heads, max_seq_len)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  int warp_size = 32;
  int max_threads_per_block = 1024;
  // all cuda uploads leak forever...
  register_cuda_host(xout, n_heads * head_dim * sizeof(float));
  register_cuda_host(att, n_heads * max_seq_len * sizeof(float));
  kb = static_cast<f16_t*>(upload_cuda(kb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  vb = static_cast<f16_t*>(upload_cuda(vb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  q = static_cast<float*>(upload_cuda(q, n_heads * head_dim * sizeof(float)));
  // multihead attention: dot products and softmax
  {
    dim3 tpb;
    tpb.x = warp_size;
    tpb.y = n_heads / n_kv_heads;
    dim3 blocks;
    blocks.x = (kv_len + tpb.x - 1) / tpb.x;
    blocks.y = (n_heads + tpb.y - 1) / tpb.y;
    attn_dot<<<blocks, tpb, 0, cudaStreamLegacy>>>(
      (half*)kb, q, head_dim, kv_len, max_seq_len, n_heads, n_kv_heads, att
    );
    attn_softmax<<<n_heads, warp_size, 0, cudaStreamLegacy>>>(
      att, kv_len, max_seq_len, n_heads, att
    );
  }
  // multihead attention: mix values with attention scores
  {
    dim3 tpb;
    tpb.x = warp_size;
    tpb.y = min(kv_len, max_threads_per_block / warp_size);
    dim3 blocks;
    blocks.x = n_heads;
    att_mix<<<blocks, tpb, 0, cudaStreamLegacy>>>(
      (half*)vb, att,
      head_dim, n_heads, n_kv_heads, 
      kv_len, max_seq_len, xout
    );
  }
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
  unregister_cuda_host(att);
}

// Test-only wrapper that runs the fused FlashAttention path. Mirrors mha_cuda
// but routes through flash_attention_decode so test.cpp can validate numerics
// against mha_cpu independently of the rest of the inference pipeline.
void mha_cuda_flash(
  float* xout,  // (n_heads, head_dim)
  f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim) — flat contiguous from test
  f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
  float* q,     // (n_heads, head_dim)
  int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
) {
  int warp_size = 32;
  register_cuda_host(xout, n_heads * head_dim * sizeof(float));
  kb = static_cast<f16_t*>(upload_cuda(kb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  vb = static_cast<f16_t*>(upload_cuda(vb, max_seq_len * n_kv_heads * head_dim * sizeof(f16_t)));
  q  = static_cast<float*>(upload_cuda(q,  n_heads * head_dim * sizeof(float)));

  // Test passes a flat (max_seq_len, ...) layout, so synthesize an identity
  // page table here to make the paged-kernel see the same byte addresses as
  // before. max_seq_len must be a multiple of KV_PAGE_SIZE for this to be
  // exact — guaranteed for the test sizes (max_seq_len=4 isn't, but with
  // identity page_table[0]=0 the only page used is page 0, so any
  // KV_PAGE_SIZE >= max_seq_len works the same).
  int n_pages = (max_seq_len + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
  std::vector<int> identity(n_pages);
  for (int i = 0; i < n_pages; ++i) identity[i] = i;
  int* page_table = static_cast<int*>(upload_cuda(identity.data(), n_pages * sizeof(int)));

  int shared_bytes = (2 * head_dim + warp_size) * (int)sizeof(float);
  flash_attention_decode<<<n_heads, warp_size, shared_bytes, cudaStreamLegacy>>>(
    (half*)kb, (half*)vb, q,
    head_dim, kv_len, max_seq_len, n_heads, n_kv_heads,
    xout, page_table
  );
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  unregister_cuda_host(xout);
}

template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d) {
  int warp_size = 32;
  // A (d,n) @ x (n,) -> out (d,)

  // all cuda uploads leak forever...
  register_cuda_host(xout, d * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, n * sizeof(float)));
  w = static_cast<T*>(upload_cuda(w, n * d * sizeof(T)));
  matmul<<<d, warp_size, 0, cudaStreamLegacy>>>(w, x, n, d, xout);
  CUDA_CHECK(cudaDeviceSynchronize()); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void matmul_cuda<float>(float*, float*, float*, int, int);
template void matmul_cuda<half>(float*, float*, half*, int, int);
template<> void matmul_cuda<f16_t>(float* xout, float* x, f16_t* w, int n, int d) {
  matmul_cuda<half>(xout, x, (half*)w, n, d);
}

template <typename T>
void ffn_cuda(
  float* xout, float* x, 
  T* w1, T* w2, T* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  int warp_size = 32;
  // all cuda uploads leak forever...
  register_cuda_host(xout, dim * sizeof(float));
  x = static_cast<float*>(upload_cuda(x, dim * sizeof(float)));
  w1 = static_cast<T*>(upload_cuda(w1, hidden_dim * dim * sizeof(T)));
  w2 = static_cast<T*>(upload_cuda(w2, dim * hidden_dim * sizeof(T)));
  w3 = static_cast<T*>(upload_cuda(w3, hidden_dim * dim * sizeof(T)));
  float* hb = new float[hidden_dim];
  float* hb2 = new float[hidden_dim];
  hb = static_cast<float*>(upload_cuda(hb, hidden_dim * sizeof(float)));
  hb2 = static_cast<float*>(upload_cuda(hb2, hidden_dim * sizeof(float)));
  // hb, hb2 leak forever on cpu too...

  // mix self.w2(F.silu(self.w1(x)) * self.w3(x))
  // Note this is a feedforward with a GLU, not a simple MLP.
  switch (act) {
    case ActivationType::GELU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::GELU><<<
        hidden_dim, warp_size, 0, cudaStreamLegacy
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
    case ActivationType::SILU: {
      fused_ffn_w1_w3_glu_act<T, ActivationType::SILU><<<
        hidden_dim, warp_size, 0, cudaStreamLegacy
      >>>(
        w1, w3, x, dim, hidden_dim, hb
      );
      break;
    }
  }
  
  matmul<<<dim, warp_size, 0, cudaStreamLegacy>>>(w2, hb, hidden_dim, dim, xout);
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamLegacy)); // After this, xout contains output
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
  unregister_cuda_host(xout);
}

template void ffn_cuda<float>(float*, float*, float*, float*, float*, int, int, ActivationType);
template void ffn_cuda<half>(float*, float*, half*, half*, half*, int, int, ActivationType);
template <> void ffn_cuda<f16_t>(
  float* xout, float* x, 
  f16_t* w1, f16_t* w2, f16_t* w3, 
  int hidden_dim, int dim,
  ActivationType act
) {
  ffn_cuda<half>(
    xout, x, 
    (half*)w1, (half*)w2, (half*)w3, 
    hidden_dim, dim, act
  );
}

template void Block::_block_cuda<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cuda<half>(InferenceState&, int, int, int, int) const;
template<> void Block::_block_cuda<f16_t>(InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len) const {
  _block_cuda<half>(s, pos, kv_sink, kv_pos, kv_len);
}

template void Block::_block_cuda_dist<float>(InferenceState&, int, int, int, int) const;
template void Block::_block_cuda_dist<half>(InferenceState&, int, int, int, int) const;
template<> void Block::_block_cuda_dist<f16_t>(InferenceState& s, int pos, int kv_sink, int kv_pos, int kv_len) const {
  _block_cuda_dist<half>(s, pos, kv_sink, kv_pos, kv_len);
}

// Tensor-parallel forward pass. Same operation sequence as
// _forward_cuda_build_graph but kernels are dispatched directly (no CUDA
// graph) and the per-layer Block calls go through Block::block, which
// dispatches to _block_cuda_dist when tp_size > 1.
//
// wcls is currently replicated, so every rank computes the full logits
// vector — sampling is deterministic across ranks given the same seed, and
// no token broadcast is needed. (See plan doc for the column-parallel wcls
// alternative; deferred.)
void Model::_forward_cuda_dist(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;
  cudaStream_t stream = s.stream();
  s.set_mode(mode);

  // copy embedding
  int block_dim = max_threads_per_block;
  int grid_dim = (c.dim + block_dim - 1) / block_dim;
  switch (c.weight_dtype) {
    case DType::F32: {
      copy_embedding_float<<<grid_dim, block_dim, 0, stream>>>(
        static_cast<float*>(token_embedding_table), c.dim, token, s.x()
      );
      break;
    }
    case DType::F16: {
      copy_embedding_half<<<grid_dim, block_dim, 0, stream>>>(
        static_cast<half*>(token_embedding_table), c.dim, token, s.x()
      );
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }

  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos  = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len  = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

  // Forward all layers. Block::block() routes to _block_cuda_dist because
  // _config->tp_size > 1.
  for (auto& b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  // final layer norm
  rmsnorm<<<1, max_threads_per_block, 0, stream>>>(
    s.x(), rms_final_weight, c.dim, c.norm_eps, s.x()
  );

  // classifier into logits — wcls replicated, full vocab on every rank.
  switch (c.weight_dtype) {
    case DType::F32: {
      matmul_wide<<<c.vocab_size/32, warp_size*32, 0, stream>>>(
        static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      );
      break;
    }
    case DType::F16: {
      matmul_wide<<<c.vocab_size/32, warp_size*32, 0, stream>>>(
        static_cast<half*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      );
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

void Model::_forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode) {
  const Config& c = *config;
  s.set_mode(mode);
  CudaGraph& g = s.graph();

  // Dispatch all the kernels that comprise the work being done on the GPU 
  // for the forward pass of the model. These calls will be recorded in the 
  // InferenceState stream to form a CUDA graph, which we can save and call again,
  // which is more efficient to execute on the device than the equivalent series
  // of kernel dispatches.
  g.wrap([&]() {
    _forward_cuda_build_graph(s, token, pos, mode);
  }, s.stream());

  g.launch(s.stream());
  
  if (mode == InferenceMode::OUTPUT_LOGITS) {
    CUDA_CHECK(cudaStreamSynchronize(s.stream())); // After this, s.logits contains logits of output token
  }
  CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
}

void Model::_forward_cuda_build_graph(InferenceState& s, int token, int pos, InferenceMode mode) {
#define STATIC_KERNEL(x) if (!s.graph().is_created) x;
  const Config& c = *config;

  {
    cudaKernelNodeParams params;
    params.blockDim = {static_cast<unsigned int>(max_threads_per_block), 1, 1};
    params.gridDim = {static_cast<unsigned int>((c.dim + max_threads_per_block - 1)/max_threads_per_block), 1, 1};
    params.sharedMemBytes = 0;
    params.extra = nullptr;
    switch (c.weight_dtype) {
      case DType::F32: {
        params.func = reinterpret_cast<void*>(copy_embedding_float);
        break;
      }
      case DType::F16: {
        params.func = reinterpret_cast<void*>(copy_embedding_half);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for CUDA");
      }
    }
    float* x = s.x();
    void* kernelParams[] = {
      &token_embedding_table,
      (void*)&c.dim,
      &token,
      &x
    };
    params.kernelParams = kernelParams;

    s.graph().add_or_update_kernel_node("copy_embedding", params, s.stream());
  }
  
  // When decoding past the context length, keep the first few tokens in the KV cache
  // untouched as "attention sinks" while replacing the rest in ring order.
  // See StreamingLLM (https://arxiv.org/pdf/2309.17453) for more.
  int kv_sink = pos >= c.max_seq_len ? KV_SINKS : 0;
  int kv_pos = kv_sink + (pos - kv_sink) % (c.max_seq_len - kv_sink);
  int kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;
  
  // forward all layers in order
  for (auto b : blocks) {
    b->block(s, pos, kv_sink, kv_pos, kv_len);
  }

  if (mode == InferenceMode::HYDRATE_KV_CACHE) {
    // only hydrate the KV cache and don't compute output logits
    CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
    return;
  }
  
  // final layer norm
  switch (c.norm_type) {
    case LayerNormType::RMSNorm: {
      STATIC_KERNEL((rmsnorm<<<1, max_threads_per_block, 0, s.stream()>>>(
        s.x(), rms_final_weight, c.dim, c.norm_eps, s.x()
      )));
      break;
    }
  }
  
  // classifier into logits
  switch (c.weight_dtype) {
    case DType::F32: {
      STATIC_KERNEL((matmul_wide<<<c.vocab_size/32, warp_size*32, 0, s.stream()>>>(
        static_cast<float*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      )));
      break;
    }
    case DType::F16: {
      STATIC_KERNEL((matmul_wide<<<c.vocab_size/32, warp_size*32, 0, s.stream()>>>(
        static_cast<half*>(wcls), s.x(), c.dim, c.vocab_size, s.logits()
      )));
      break;
    }
    default: {
      assert(false && "unsupported weight dtype for CUDA");
    }
  }
#undef STATIC_KERNEL
}

void CudaGraph::wrap(std::function<void()> func, cudaStream_t s) {
  if (!is_created) {
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    func();
    CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));
    is_created = true;
  } else {
    func();
  }
}

void CudaGraph::launch(cudaStream_t s) {
  CUDA_CHECK(cudaGraphLaunch(instance, s));
}

void CudaGraph::add_or_update_kernel_node(std::string key, cudaKernelNodeParams params, cudaStream_t stream) {
  if (!is_created) {
    // Get the currently capturing graph (since `graph` starts out null when recording)
    cudaStreamCaptureStatus capture_status;
    cudaGraph_t current_graph;
    const cudaGraphNode_t *deps;
    size_t dep_count;
    CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream, &capture_status, nullptr, &current_graph, &deps, &dep_count));

    // Now add a new node
    cudaGraphNode_t new_node;
    CUDA_CHECK(cudaGraphAddKernelNode(&new_node, current_graph, deps, dep_count, &params));
    nodes[key] = new_node;
    // Update the stream dependency
    CUDA_CHECK(cudaStreamUpdateCaptureDependencies(stream, &new_node, 1, 1));
  } else {
    auto it = nodes.find(key);
    if (it != nodes.end()) {
      CUDA_CHECK(cudaGraphExecKernelNodeSetParams(instance, it->second, &params));
    } else {
      assert(false && "adding new graph nodes after capture currently not supported");
    }
  }
}