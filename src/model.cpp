#include "model.h"

#include "json.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cfloat>
#include <cstdlib>
#include <cstring>
#include "fmt/format.h"
#include <iostream>
#include <limits.h>
#include <string>

#include "immintrin.h"

using json = nlohmann::json;

void Config::from_yalm(YALMData& yalm, int context) {
  dim = std::stoi(yalm.metadata.at("dim").get<std::string>());
  hidden_dim = std::stoi(yalm.metadata.at("hidden_dim").get<std::string>());
  head_dim = std::stoi(yalm.metadata.at("head_dim").get<std::string>());
  n_layers = std::stoi(yalm.metadata.at("n_layers").get<std::string>());
  n_heads = std::stoi(yalm.metadata.at("n_heads").get<std::string>());
  n_kv_heads = std::stoi(yalm.metadata.at("n_kv_heads").get<std::string>());
  vocab_size = std::stoi(yalm.metadata.at("vocab_size").get<std::string>());
  // mixture of experts
  n_experts = yalm.metadata.contains("n_experts") ? std::stoi(yalm.metadata.at("n_experts").get<std::string>()) : 0;
  n_experts_active = yalm.metadata.contains("n_experts_active") ? std::stoi(yalm.metadata.at("n_experts_active").get<std::string>()) : 0;

  // for now limit seq_len to 4096 to avoid KV cache OOM for models like Mistral since window size isn't correctly specified
  max_seq_len = std::min(std::stoi(yalm.metadata.at("max_seq_len").get<std::string>()), 4096);
  if (context) {
    max_seq_len = context;
  }

  rope_theta = std::stof(yalm.metadata.at("rope_theta").get<std::string>());
  rotary_dim = std::stoi(yalm.metadata.at("rotary_dim").get<std::string>());

  norm_eps = std::stof(yalm.metadata.value("norm_eps", "1e-5"));

  std::string act_str = yalm.metadata.value("act_type", "gelu");
  if (act_str == "gelu") {
    act = ActivationType::GELU;
  } else if (act_str == "silu") {
    act = ActivationType::SILU;
  } else {
    std::cerr << "unsupported act_type, defaulting to gelu" << std::endl;
    act = ActivationType::GELU;
  }

  std::string norm_type_str = yalm.metadata.value("norm_type", "rmsnorm");
  if (norm_type_str == "rmsnorm") {
    norm_type = LayerNormType::RMSNorm;
  } else {
    std::cerr << "unsupported norm_type, defaulting to rmsnorm" << std::endl;
    norm_type = LayerNormType::RMSNorm;
  }

  qkv_clip = yalm.metadata.contains("qkv_clip") ? std::stof(yalm.metadata.at("qkv_clip").get<std::string>()) : FLT_MAX;

  std::string dtype = yalm.metadata.at("dtype").get<std::string>();
  // TODO: support fp8
  if (dtype == "fp32") {
    weight_dtype = DType::F32;
  } else if (dtype == "fp16") {
    weight_dtype = DType::F16;
  } else {
    std::cerr << "FATAL: unsupported dtype: " << dtype << std::endl;
    assert(false);
  }

  // Default to no TP partition (single-GPU). main() overrides this via
  // set_tp() once it knows the world size.
  set_tp(0, 1);
}

void Config::set_tp(int rank, int size) {
  assert(size >= 1);
  if (n_heads % size != 0 || n_kv_heads % size != 0 || hidden_dim % size != 0) {
    std::cerr << "FATAL: tp_size=" << size
              << " must divide n_heads=" << n_heads
              << ", n_kv_heads=" << n_kv_heads
              << ", hidden_dim=" << hidden_dim << std::endl;
    assert(false);
  }
  tp_rank = rank;
  tp_size = size;
  n_heads_local    = n_heads    / size;
  n_kv_heads_local = n_kv_heads / size;
  q_dim_local      = n_heads_local    * head_dim;
  kv_dim_local     = n_kv_heads_local * head_dim;
  hidden_dim_local = hidden_dim / size;
}

size_t Config::active_bytes(size_t pos) const {
  size_t weight_size = dtype_size(weight_dtype);

  size_t bytes_per_block = 0;
  bytes_per_block += 2 * dim * sizeof(float); // rms_att_weight, rms_ffn_weight
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wq
  bytes_per_block += 2 * n_kv_heads * head_dim * dim * weight_size; // wk, wv
  bytes_per_block += n_heads * head_dim * dim * weight_size; // wo
  if (n_experts > 0) {
    bytes_per_block += n_experts * dim * weight_size; // moegate
    bytes_per_block += n_experts_active * 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  } else {
    bytes_per_block += 3 * dim * hidden_dim * weight_size; // w1, w2, w3
  }
  size_t kv_len = std::min(static_cast<size_t>(max_seq_len), pos + 1);
  size_t kv_entry_size = sizeof(f16_t);
  bytes_per_block += 2 * kv_len * n_kv_heads * head_dim * kv_entry_size; // key_cache, value_cache

  size_t bytes = 0;
  bytes += dim * weight_size; // 1 row of token_embedding_table
  bytes += n_layers * bytes_per_block; // blocks
  bytes += dim * sizeof(float); // rms_final_weight
  bytes += vocab_size * dim * sizeof(float); // wcls

  return bytes;
}

void* check_tensor(const Tensor* tensor, DType weight_dtype, std::array<int, 4> shape) {
  if (tensor == nullptr) {
    std::cerr << "FATAL: missing tensor" << std::endl;
    assert(false);
    return nullptr;
  }
  if (tensor->dtype != weight_dtype || tensor->shape != shape) {
    std::cerr << "FATAL: tensor mismatch for " << tensor->name << std::endl;
    std::cerr 
      << fmt::format("expected: dtype={}, shape=[{},{},{},{}]", dtype_to_string(weight_dtype), shape[0], shape[1], shape[2], shape[3]) 
      << std::endl;
    std::cerr 
      << fmt::format("got: dtype={}, shape=[{},{},{},{}]", dtype_to_string(tensor->dtype), tensor->shape[0], tensor->shape[1], tensor->shape[2], tensor->shape[3]) 
      << std::endl;
    assert(false);
  }
  return tensor->data;
};

const Tensor* get_tensor(const YALMData& yalm, const std::string& key) {
  auto it = yalm.tensors.find(key);
  if (it == yalm.tensors.end()) {
    std::cerr << "FATAL: missing tensor: " << key << std::endl;
    assert(false);
    return nullptr;
  }
  const Tensor& tensor = it->second;
  return &tensor;
};

Block::Block(
  int layer_i,
  const std::shared_ptr<Config> config,
  const Tensor* rms_att_weight,
  const Tensor* rms_ffn_weight,
  const Tensor* wq,
  const Tensor* wk,
  const Tensor* wv,
  const Tensor* wo,
  const Tensor* w1,
  const Tensor* w2,
  const Tensor* w3,
  const Tensor* moegate
) {
  _layer_i = layer_i;
  _config = config;
  switch (config->weight_dtype) {
    case DType::F32:
    case DType::F16: {
      break;
    }
    default: {
      std::cerr << "FATAL: unsupported weight dtype " << dtype_to_string(config->weight_dtype) << std::endl;
      assert(false);
      break;
    }
  }

  _rms_att_weight = static_cast<float*>(check_tensor(
    rms_att_weight, DType::F32, {config->dim, 0, 0, 0}
  ));
  _rms_ffn_weight = static_cast<float*>(check_tensor(
    rms_ffn_weight, DType::F32, {config->dim, 0, 0, 0}
  ));

  _wq = check_tensor(
    wq, config->weight_dtype, {config->n_heads * config->head_dim, config->dim, 0, 0}
  );
  _wk = check_tensor(
    wk, config->weight_dtype, {config->n_kv_heads * config->head_dim, config->dim, 0, 0}
  );
  _wv = check_tensor(
    wv, config->weight_dtype, {config->n_kv_heads * config->head_dim, config->dim, 0, 0}
  );
  _wo = check_tensor(
    wo, config->weight_dtype, {config->dim, config->n_heads * config->head_dim, 0, 0}
  );

  if (config->n_experts > 0) {
    _moegate = check_tensor(
      moegate, config->weight_dtype, {config->n_experts, config->dim, 0, 0}
    );
    _w1 = check_tensor(
      w1, config->weight_dtype, {config->n_experts, config->hidden_dim, config->dim, 0}
    );
    _w2 = check_tensor(
      w2, config->weight_dtype, {config->n_experts, config->dim, config->hidden_dim, 0}
    );
    _w3 = check_tensor(
      w3, config->weight_dtype, {config->n_experts, config->hidden_dim, config->dim, 0}
    );
  } else {
    _w1 = check_tensor(
      w1, config->weight_dtype, {config->hidden_dim, config->dim, 0, 0}
    );
    _w2 = check_tensor(
      w2, config->weight_dtype, {config->dim, config->hidden_dim, 0, 0}
    );
    _w3 = check_tensor(
      w3, config->weight_dtype, {config->hidden_dim, config->dim, 0, 0}
    );
  }

  // KV cache: round up max_seq_len to a whole number of pages.
  // Physical storage is (max_pages, KV_PAGE_SIZE, n_kv_heads_local, head_dim).
  // n_kv_heads_local == n_kv_heads when tp_size==1; otherwise this rank owns
  // only its KV-head slice (see Config::set_tp).
  int max_pages = (config->max_seq_len + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
  size_t cache_slots = static_cast<size_t>(max_pages) * KV_PAGE_SIZE;
  int n_kvh_l = config->n_kv_heads_l();
  _key_cache = new f16_t[cache_slots * n_kvh_l * config->head_dim]();
  _value_cache = new f16_t[cache_slots * n_kvh_l * config->head_dim]();
  // _page_table is allocated on device in Block::cuda(); CPU inference path
  // doesn't need it because _block_cpu indexes the cache directly.
}

int Block::max_pages() const {
  return (_config->max_seq_len + KV_PAGE_SIZE - 1) / KV_PAGE_SIZE;
}

Block::~Block() {
  if (_device == Device::CPU) {
    delete[] _key_cache;
    delete[] _value_cache;
  } else {
    free_cuda(_key_cache);
    free_cuda(_value_cache);
    if (_page_table) free_cuda(_page_table);
  }
}

// Pack a row-major column slice (n_rows x n_cols_full -> n_rows x n_cols_local
// covering cols [col_offset, col_offset+n_cols_local)) into a contiguous host
// buffer. Used to extract row-parallel weight slices for wo and w2, where
// the rank-local input columns are non-contiguous in the source layout.
static void* pack_col_slice(
  const void* src,
  int n_rows,
  int n_cols_full,
  int col_offset,
  int n_cols_local,
  size_t elem_size
) {
  size_t row_bytes_local = static_cast<size_t>(n_cols_local) * elem_size;
  size_t total_bytes = static_cast<size_t>(n_rows) * row_bytes_local;
  void* dst = std::malloc(total_bytes);
  assert(dst);
  const char* s = static_cast<const char*>(src);
  char* d = static_cast<char*>(dst);
  size_t src_row_bytes = static_cast<size_t>(n_cols_full) * elem_size;
  size_t src_col_offset_bytes = static_cast<size_t>(col_offset) * elem_size;
  for (int i = 0; i < n_rows; ++i) {
    std::memcpy(d + i * row_bytes_local,
                s + i * src_row_bytes + src_col_offset_bytes,
                row_bytes_local);
  }
  return dst;
}

void Block::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  const size_t weight_size = dtype_size(_config->weight_dtype);
  const int dim       = _config->dim;
  const int hidden    = _config->hidden_dim;
  const int q_dim     = _config->n_heads    * _config->head_dim;
  const int kv_dim    = _config->n_kv_heads * _config->head_dim;
  const int P         = _config->tp_size;
  const int r         = _config->tp_rank;
  const int q_dim_l   = _config->q_dim_l();
  const int kv_dim_l  = _config->kv_dim_l();
  const int hidden_l  = _config->hidden_dim_l();

  // norms — replicated on every rank.
  _rms_att_weight = static_cast<float*>(upload_cuda(_rms_att_weight, dim * sizeof(float)));
  _rms_ffn_weight = static_cast<float*>(upload_cuda(_rms_ffn_weight, dim * sizeof(float)));

  if (P == 1) {
    // Single-GPU: full weights.
    _wq = upload_cuda(_wq, q_dim  * dim * weight_size);
    _wk = upload_cuda(_wk, kv_dim * dim * weight_size);
    _wv = upload_cuda(_wv, kv_dim * dim * weight_size);
    _wo = upload_cuda(_wo, dim    * q_dim  * weight_size);
    _w1 = upload_cuda(_w1, hidden * dim    * weight_size);
    _w2 = upload_cuda(_w2, dim    * hidden * weight_size);
    _w3 = upload_cuda(_w3, hidden * dim    * weight_size);
  } else {
    // Tensor parallel: column-parallel matmuls split along the output-row
    // axis (wq/wk/wv/w1/w3 — rows of the row-major host buffer are
    // contiguous, so the rank-local slice is a single pointer offset). Row-
    // parallel matmuls (wo/w2) split along the input-column axis — non-
    // contiguous in row-major source, so we pack into a temporary buffer
    // before upload. The temp is freed after upload.

    // wq: rows [r*q_dim_l, (r+1)*q_dim_l) of a (q_dim, dim) matrix.
    _wq = upload_cuda(
      static_cast<char*>(_wq) + (size_t)r * q_dim_l * dim * weight_size,
      (size_t)q_dim_l * dim * weight_size
    );
    // wk, wv: rows [r*kv_dim_l, (r+1)*kv_dim_l) of (kv_dim, dim).
    _wk = upload_cuda(
      static_cast<char*>(_wk) + (size_t)r * kv_dim_l * dim * weight_size,
      (size_t)kv_dim_l * dim * weight_size
    );
    _wv = upload_cuda(
      static_cast<char*>(_wv) + (size_t)r * kv_dim_l * dim * weight_size,
      (size_t)kv_dim_l * dim * weight_size
    );
    // w1, w3: rows [r*hidden_l, (r+1)*hidden_l) of (hidden, dim).
    _w1 = upload_cuda(
      static_cast<char*>(_w1) + (size_t)r * hidden_l * dim * weight_size,
      (size_t)hidden_l * dim * weight_size
    );
    _w3 = upload_cuda(
      static_cast<char*>(_w3) + (size_t)r * hidden_l * dim * weight_size,
      (size_t)hidden_l * dim * weight_size
    );
    // wo: cols [r*q_dim_l, (r+1)*q_dim_l) of (dim, q_dim). Pack first.
    {
      void* packed = pack_col_slice(
        _wo, dim, q_dim, r * q_dim_l, q_dim_l, weight_size
      );
      _wo = upload_cuda(packed, (size_t)dim * q_dim_l * weight_size);
      std::free(packed);
    }
    // w2: cols [r*hidden_l, (r+1)*hidden_l) of (dim, hidden). Pack first.
    {
      void* packed = pack_col_slice(
        _w2, dim, hidden, r * hidden_l, hidden_l, weight_size
      );
      _w2 = upload_cuda(packed, (size_t)dim * hidden_l * weight_size);
      std::free(packed);
    }
  }

  // kv cache — local n_kv_heads slice. Size matches the host alloc made in
  // the Block constructor.
  int n_pages = max_pages();
  size_t cache_slots = static_cast<size_t>(n_pages) * KV_PAGE_SIZE;
  size_t cache_bytes = cache_slots * _config->n_kv_heads_l() * _config->head_dim * sizeof(f16_t);
  _key_cache   = static_cast<f16_t*>(upload_cuda(_key_cache,   cache_bytes));
  _value_cache = static_cast<f16_t*>(upload_cuda(_value_cache, cache_bytes));

  // Page table: identity mapping (logical_page == physical_page).
  // This makes paged-KV byte-equivalent to the old contiguous layout for now.
  // Phase 6 (dynamic batching) will replace this with a real allocator.
  std::vector<int> identity(n_pages);
  for (int i = 0; i < n_pages; ++i) identity[i] = i;
  _page_table = static_cast<int*>(upload_cuda(identity.data(), n_pages * sizeof(int)));
}

void Block::block(
  InferenceState& s,  // inference state
  int pos,            // index of the current token in the sequence
  int kv_sink,        // number of sink tokens currently in the KV cache
  int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
  int kv_len          // number of tokens in the kv cache that we will attend over
) const {
  if (_device == Device::CUDA) {
    const bool use_dist = _config->tp_size > 1;
    switch (_config->weight_dtype) {
      case DType::F32: {
        if (use_dist) _block_cuda_dist<float>(s, pos, kv_sink, kv_pos, kv_len);
        else          _block_cuda<float>     (s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
        if (use_dist) _block_cuda_dist<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
        else          _block_cuda<f16_t>     (s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cuda");
      }
    }
  } else {
    switch (_config->weight_dtype) {
      case DType::F32: {
        _block_cpu<float>(s, pos, kv_sink, kv_pos, kv_len);
        break;
      }
      case DType::F16: {
#if defined(__AVX2__) && defined(__F16C__)
        _block_cpu<f16_t>(s, pos, kv_sink, kv_pos, kv_len);
#else
        assert(false && "float16 not supported on this platform");
#endif
        break;
      }
      default: {
        assert(false && "unsupported weight dtype for cpu");
      }
    }
  }

}

InferenceState::InferenceState(const std::shared_ptr<Config> config):
  _config(config) {
  assert(config);
  // x and xb live in the (replicated) residual stream and stay at full dim.
  // Everything else is rank-local: when tp_size==1 the *_l() accessors return
  // the full dim, so single-GPU behavior is unchanged.
  const int dim       = config->dim;
  const int q_dim_l   = config->q_dim_l();
  const int kv_dim_l  = config->kv_dim_l();
  const int hidden_l  = config->hidden_dim_l();
  const int n_heads_l = config->n_heads_l();

  _x   = new float[dim]();
  _xb  = new float[dim]();
  _xb2 = new float[q_dim_l]();
  _hb  = new float[hidden_l]();
  _hb2 = new float[hidden_l]();
  _q   = new float[q_dim_l]();
  _k   = new float[kv_dim_l]();
  _v   = new float[kv_dim_l]();
  _att = new float[n_heads_l * config->max_seq_len]();
  _logits = new float[config->vocab_size]();
  _moe_weights = new float[config->n_experts]();
  _active_experts = new int[config->n_experts_active]();
  _active_experts_weights = new float[config->n_experts_active]();
  // _tp_xpart allocated on-device in cuda() when tp_size > 1.
}

InferenceState::~InferenceState() {
  if (_device == Device::CPU) {
    delete[] _x;
    delete[] _xb;
    delete[] _xb2;
    delete[] _hb;
    delete[] _hb2;
    delete[] _q;
    delete[] _k;
    delete[] _v;
    delete[] _att;
    delete[] _logits;
    delete[] _moe_weights;
    delete[] _active_experts;
    delete[] _active_experts_weights;
  } else {
    free_cuda(_x);
    free_cuda(_xb);
    free_cuda(_xb2);
    free_cuda(_hb);
    free_cuda(_hb2);
    free_cuda(_q);
    free_cuda(_k);
    free_cuda(_v);
    free_cuda(_att);
    unregister_cuda_host(_logits);
    delete[] _logits;
    free_cuda(_moe_weights);
    free_cuda(_active_experts);
    free_cuda(_active_experts_weights);
    if (_tp_xpart) free_cuda(_tp_xpart);
  }
}

void InferenceState::cuda() {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  init_cuda_stream(&_stream);
  const int dim       = _config->dim;
  const int q_dim_l   = _config->q_dim_l();
  const int kv_dim_l  = _config->kv_dim_l();
  const int hidden_l  = _config->hidden_dim_l();
  const int n_heads_l = _config->n_heads_l();

  _x   = static_cast<float*>(upload_cuda(_x,   dim       * sizeof(float)));
  _xb  = static_cast<float*>(upload_cuda(_xb,  dim       * sizeof(float)));
  _xb2 = static_cast<float*>(upload_cuda(_xb2, q_dim_l   * sizeof(float)));
  _hb  = static_cast<float*>(upload_cuda(_hb,  hidden_l  * sizeof(float)));
  _hb2 = static_cast<float*>(upload_cuda(_hb2, hidden_l  * sizeof(float)));
  _q   = static_cast<float*>(upload_cuda(_q,   q_dim_l   * sizeof(float)));
  _k   = static_cast<float*>(upload_cuda(_k,   kv_dim_l  * sizeof(float)));
  _v   = static_cast<float*>(upload_cuda(_v,   kv_dim_l  * sizeof(float)));
  _att = static_cast<float*>(upload_cuda(_att, n_heads_l * _config->max_seq_len * sizeof(float)));
  register_cuda_host(_logits, _config->vocab_size * sizeof(float));
  _moe_weights = static_cast<float*>(upload_cuda(_moe_weights, _config->n_experts * sizeof(float)));
  _active_experts = static_cast<int*>(upload_cuda(_active_experts, _config->n_experts_active * sizeof(int)));
  _active_experts_weights = static_cast<float*>(upload_cuda(_active_experts_weights, _config->n_experts_active * sizeof(float)));

  // TP partial-reduction buffer: allocated only when actually doing TP.
  if (_config->tp_size > 1) {
    std::vector<float> zeros(dim, 0.0f);
    _tp_xpart = static_cast<float*>(upload_cuda(zeros.data(), dim * sizeof(float)));
  }
}

Model::Model(YALMData& yalm, int context, int tp_rank, int tp_size) {
  config = std::make_shared<Config>();
  config->from_yalm(yalm, context);
  // Apply TP partition before creating Blocks so KV cache is allocated at
  // the per-rank size from the start. from_yalm() already called set_tp(0,1);
  // this overrides it when the caller knows the real rank/size.
  config->set_tp(tp_rank, tp_size);
  if (tp_rank == 0) {
    std::cout << "loading model with dtype: " << dtype_to_string(config->weight_dtype);
    if (tp_size > 1) {
      std::cout << " (TP=" << tp_size << ")";
    }
    std::cout << std::endl;
  }

  token_embedding_table = check_tensor(
    get_tensor(yalm, "model.embed.weight"), 
    config->weight_dtype,
    {config->vocab_size, config->dim, 0, 0}
  );

  for (int i = 0; i < config->n_layers; ++i) {
    blocks.emplace_back(std::make_shared<Block>(
      i,
      config,
      get_tensor(yalm, fmt::format("model.layers.{}.attn.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.norm.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wq.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wk.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wv.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.attn.wo.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w1.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w2.weight", i)),
      get_tensor(yalm, fmt::format("model.layers.{}.mlp.w3.weight", i)),
      config->n_experts > 0 ? get_tensor(yalm, fmt::format("model.layers.{}.moegate.weight", i)) : nullptr
    ));
  }

  rms_final_weight = static_cast<float*>(check_tensor(
    get_tensor(yalm, "model.norm.weight"), 
    DType::F32, 
    {config->dim, 0, 0, 0}
  ));
  bool tie_word_embeddings = yalm.tensors.count("model.output.weight") == 0;
  if (tie_word_embeddings) {
    wcls = token_embedding_table;
  } else {
    wcls = check_tensor(
      get_tensor(yalm, "model.output.weight"), 
      config->weight_dtype, 
      {config->vocab_size, config->dim, 0, 0}
    );
  }
}

void Model::cuda(int device_id) {
  if (_device != Device::CPU) {
    return;
  }
  _device = Device::CUDA;
  // Bind this process to `device_id`. main() passes dist::local_rank() so each
  // MPI rank owns one GPU on its node; single-process callers (tests) keep
  // the default 0.
  set_cuda_device(device_id);
  size_t weight_size = dtype_size(config->weight_dtype);
  // Embedding and wcls are kept replicated on every rank for now — see plan.
  token_embedding_table = upload_cuda(token_embedding_table, config->vocab_size * config->dim * weight_size);
  for (auto& block : blocks) {
    block->cuda();
  }
  rms_final_weight = static_cast<float*>(upload_cuda(rms_final_weight, config->dim * sizeof(float)));
  wcls = upload_cuda(wcls, config->vocab_size * config->dim * weight_size);
}

void Model::forward(InferenceState& s, int token, int pos, InferenceMode mode) {
  if (s.device() != _device) {
    std::cerr << "FATAL: inference state device mismatch" << std::endl;
    assert(false);
    return;
  }
  if (_device == Device::CUDA) {
    if (config->tp_size > 1) {
      _forward_cuda_dist(s, token, pos, mode);
    } else {
      _forward_cuda(s, token, pos, mode);
    }
  } else {
    _forward_cpu(s, token, pos, mode);
  }
}

#if DEBUG_MODEL
DebugTensor::DebugTensor(const std::vector<float>& data) {
  data_f32 = data;
  data_type = DataType::F32;
}
DebugTensor::DebugTensor(const std::vector<f16_t>& data) {
  data_f16 = data;
  data_type = DataType::F16;
}

float DebugTensor::max_err(const DebugTensor& other) const {
  if (data_type != other.data_type) {
    return -1;
  }
  if (data_type == DataType::F32) {
    float max_err = 0;
    for (size_t i = 0; i < data_f32.size(); i++) {
      max_err = std::max(max_err, std::abs(data_f32[i] - other.data_f32[i]));
    }
    return max_err;
  } else {
#if defined(__F16C__)
    float max_err = 0;
    for (size_t i = 0; i < data_f16.size(); i++) {
      max_err = std::max(max_err, std::abs(_cvtsh_ss(data_f16[i]) - _cvtsh_ss(other.data_f16[i])));
    }
    return max_err;
#else
  assert(false && "float16 not supported on this platform");
#endif
  }
}
#endif