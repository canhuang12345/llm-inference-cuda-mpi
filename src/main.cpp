#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdio.h>

#include "fmt/format.h"

#include "codec.h"
#include "dist.h"
#include "model.h"
#include "profiler.h"
#include "sampler.h"
#include "time.h"
#include "tokenizer.h"

// Convenience predicate for gating console output. Every rank reads the model
// file and runs the forward, but only rank 0 produces user-visible output;
// otherwise the same lines would be duplicated tp_size times.
static inline bool is_rank0() { return dist::rank() == 0; }

void error_usage() {
  // Only rank 0 prints the usage to avoid N copies when invoked via mpirun
  // with bad args. We still exit on every rank so MPI doesn't hang.
  if (dist::initialized() && dist::rank() != 0) {
    exit(1);
  }
  fprintf(stderr, "Usage:   main <checkpoint> [options]\n");
  fprintf(stderr, "Example: main model.yalm -i \"Q: What is the meaning of life?\"\n");
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  -h Display this help message\n");
  fprintf(stderr, "  -d [cpu,cuda] which device to use (default - cuda)\n");
  fprintf(stderr, "  -m [completion,passkey,perplexity] which mode to run in (default - completion)\n");
  fprintf(stderr, "  -T <int> sliding window context length (0 - max)\n");
  fprintf(stderr, "  -p enable CUDA-event-based forward-pass profiler\n");
  fprintf(stderr, "  -s <int> sampler seed (default - timestamp). Use a fixed seed for cross-rank correctness checks.\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Perplexity mode options:\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -t <float> temperature (default - 1.0)\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "Completion mode options:\n");
  fprintf(stderr, "  -n <int>    number of steps to run for in completion mode, default 256. 0 = max_seq_len, -1 = infinite\n");
  fprintf(stderr, "  Choose one:\n");
  fprintf(stderr, "    -i <string> input prompt\n");
  fprintf(stderr, "    -f <filepath> input file with prompt\n");
  fprintf(stderr, "Passkey mode options:\n");
  fprintf(stderr, "  -n <int>    number of junk lines to insert (default - 250)\n");
  fprintf(stderr, "  -l <int>    passkey position (-1 - random)\n");
  exit(1);
}

#if DEBUG_MODEL
void debug_tensors(Config& c) {
  assert(debug_map_cpu().size() == debug_map_cuda().size());
  for (auto& [name, cpu] : debug_map_cpu()) {
    DebugTensor& cuda = debug_map_cuda().at(name);
    float maxerr = cpu.max_err(cuda);
    std::cout << fmt::format("{} maxerr: {}", name, maxerr) << std::endl;
  }
  std::cout << std::endl;
}
#endif

void run_completion(
  const std::string& checkpoint_path,
  const std::string& device,
  const std::string& prompt,
  const int context,
  int num_steps,
  float temperature,
  bool profile,
  uint64_t seed
) {
  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data, context, dist::rank(), dist::size());
  InferenceState state(model.config);
  Sampler sampler(model.config, seed);
  Tokenizer tokenizer(model_data);

  if (is_rank0()) {
    std::cout << "Model active bytes with full context window: " << model.config->active_bytes(model.config->max_seq_len) << std::endl;
  }

  if (num_steps == 0) {
    // `-n 0` means use the full context length
    num_steps = model.config->max_seq_len;
  }
  if (device == "cuda") {
    if (is_rank0()) std::cout << "Using CUDA" << std::endl;
    model.cuda(dist::local_rank());
    state.cuda();
  }

  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and
  // kernels are compiled + instantiated.
  model.forward(state, 0, 0);

  CudaProfiler prof;
  prof.set_enabled(profile && device == "cuda");

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    if (is_rank0()) {
      std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
      uint64_t encoding_ms = encode_end_ms - encode_start_ms;
      std::cout << fmt::format(
        "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
        encoding.size(),
        encoding.size() / (encoding_ms / 1000.0),
        (encoding_ms / 1000.0) / encoding.size(),
        encoding_ms / 1000.0
      ) << std::endl;
    }
  }

  uint64_t start_ms = get_timestamp_ms();
  size_t read_bytes = 0;
  // Hydrate KV cache by forwarding model on all prompt tokens and discarding output.
  // This also generates output logits for the last token.
  for (size_t pos = 0; pos < encoding.size(); pos++) {
    int token_id = encoding[pos];
    InferenceMode inferMode = pos + 1 == encoding.size() ?
      InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
    size_t step_bytes = model.config->active_bytes(pos);
    prof.begin_forward(state.stream());
    model.forward(state, token_id, pos, inferMode);
    prof.end_forward(state.stream(), /*is_hydrate=*/true, step_bytes);
    read_bytes += step_bytes;
  }
  uint64_t end_hydrate_ms = get_timestamp_ms();
  // For N steps:
  // - Sample + decode output logits
  // - Forward the model
  // TP note: every rank computes the same logits (replicated wcls) and runs
  // an identical RNG sequence given the same seed, so each rank picks the
  // same token without any explicit broadcast.
  for (int i = 0; i < num_steps || num_steps == -1; i++) {
    int token_id = sampler.sample(state, temperature);
    if (is_rank0()) {
      std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
      std::cout << token_str << std::flush;
    }
    encoding.push_back(token_id);
    if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
      break;
    }
    size_t step_bytes = model.config->active_bytes(encoding.size() - 1);
    prof.begin_forward(state.stream());
    model.forward(state, token_id, encoding.size() - 1);
    prof.end_forward(state.stream(), /*is_hydrate=*/false, step_bytes);
    read_bytes += step_bytes;
  }
  if (is_rank0()) {
    std::cout << "\n" << std::endl;
    uint64_t end_ms = get_timestamp_ms();
    double elapsed_s = (end_ms - start_ms) / 1000.0;
    std::cout << fmt::format(
      "Generation stats:\n"
      "  {} tokens\n"
      "  throughput: {:.5}tok/s\n"
      "  latency: {:.5}s/tok\n"
      "  hydrate: {:.5}s\n"
      "  bandwidth: {:.5}GB/s\n"
      "  total: {:.5}s\n",
      encoding.size(),
      encoding.size() / elapsed_s,
      elapsed_s / encoding.size(),
      (end_hydrate_ms - start_ms) / 1000.0,
      ((double)read_bytes / 1e9) / elapsed_s,
      elapsed_s
    ) << std::endl;
  }

  prof.synchronize();
  if (is_rank0()) prof.report();
}

void run_perplexity(
  const std::string& checkpoint_path,
  const std::string& device,
  const std::string& prompt,
  const int context,
  bool profile,
  uint64_t seed
) {
  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data, context, dist::rank(), dist::size());
  InferenceState state(model.config);
  Sampler sampler(model.config, seed);
  Tokenizer tokenizer(model_data);

  if (is_rank0()) {
    std::cout << "Model active bytes with full context window: " << model.config->active_bytes(model.config->max_seq_len) << std::endl;
  }

  if (device == "cuda") {
    if (is_rank0()) std::cout << "Using CUDA" << std::endl;
    model.cuda(dist::local_rank());
    state.cuda();
  }

  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and
  // kernels are compiled + instantiated.
  model.forward(state, 0, 0);

  CudaProfiler prof;
  prof.set_enabled(profile && device == "cuda");

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    if (is_rank0()) {
      std::cout << tokenizer.encoding_to_debug_string(encoding) << std::endl;
      uint64_t encoding_ms = encode_end_ms - encode_start_ms;
      std::cout << fmt::format(
        "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
        encoding.size(),
        encoding.size() / (encoding_ms / 1000.0),
        (encoding_ms / 1000.0) / encoding.size(),
        encoding_ms / 1000.0
      ) << std::endl;
    }
  }

  double sum_logprob = 0.0;
  double ss_logprob = 0.0;
  // Generates output logits for all tokens in the prompt and sum log probs to
  // compute perplexity.
  uint64_t start_ms = get_timestamp_ms();
  size_t read_bytes = 0;
  size_t N = encoding.size() - 1;
  for (size_t pos = 0; pos + 1 < encoding.size(); pos++) {
    if (is_rank0()) {
      std::cout << "\r Computing perplexity..." << pos + 1 << "/" << N << std::flush;
    }

    int token_id = encoding[pos];
    size_t step_bytes = model.config->active_bytes(pos);
    prof.begin_forward(state.stream());
    model.forward(state, token_id, pos);
    prof.end_forward(state.stream(), /*is_hydrate=*/false, step_bytes);
    read_bytes += step_bytes;

    double logprob = std::log(sampler.sample_prob(encoding[pos + 1], state));
    sum_logprob += logprob;
    ss_logprob += logprob * logprob;
  }
  if (is_rank0()) {
    std::cout << std::endl;
    uint64_t end_ms = get_timestamp_ms();
    double elapsed_s = (end_ms - start_ms)/1000.0;
    double perplexity = std::exp(-sum_logprob / N);
    double perplexity_error = perplexity * std::sqrt(
      (ss_logprob - sum_logprob * sum_logprob / N) / N / N
    );
    std::cout << fmt::format(
      "Stats:\n"
      "  {} tokens\n"
      "  perplexity: {:.5} ± {:.5}\n"
      "  throughput: {:.5}tok/s\n"
      "  latency: {:.5}s/tok\n"
      "  bandwidth: {:.5}GB/s\n"
      "  total: {:.5}s\n",
      N,
      perplexity,
      perplexity_error,
      N / elapsed_s,
      elapsed_s / N,
      ((double)read_bytes / 1e9) / elapsed_s,
      elapsed_s
    ) << std::endl;
  }

  prof.synchronize();
  if (is_rank0()) prof.report();
}

void run_passkey(
  const std::string& checkpoint_path,
  const std::string& device,
  const int context,
  const int n_junk,
  const int passkey_pos,
  bool profile,
  uint64_t seed
) {
  YALMData model_data;
  model_data.from_file(checkpoint_path);
  Model model(model_data, context, dist::rank(), dist::size());
  InferenceState state(model.config);
  Sampler sampler(model.config, seed);
  Tokenizer tokenizer(model_data);

  if (is_rank0()) {
    std::cout << "Model active bytes with full context window: " << model.config->active_bytes(model.config->max_seq_len) << std::endl;
  }

  if (device == "cuda") {
    if (is_rank0()) std::cout << "Using CUDA" << std::endl;
    model.cuda(dist::local_rank());
    state.cuda();
  }

  // Do one inference as warmup.
  // On CPU, this ensures all tensors are loaded into memory via mmap.
  // On GPU, this ensures all tensors are loaded into device memory and
  // kernels are compiled + instantiated.
  model.forward(state, 0, 0);

  CudaProfiler prof;
  prof.set_enabled(profile && device == "cuda");

  const std::string PROMPT_PREFIX =
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize them. I will quiz you about the important information there.";
  const std::string PROMPT_SUFFIX = " What is the pass key? The pass key is";

  // std::rand() draws here are deterministic per seed; every rank seeded the
  // same way, so passkey/position agree across ranks without a broadcast.
  const int passkey = std::rand() % 50000 + 1;
  const int pos = passkey_pos == -1 ? std::rand() % n_junk : passkey_pos;

  std::string prompt = PROMPT_PREFIX;
  for (int i = 0; i < n_junk; i++) {
    if (i % n_junk == pos) {
      prompt += " The pass key is " + std::to_string(passkey) + ". Remember it. " + std::to_string(passkey) + " is the pass key.";
    }
    prompt += " The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.";
  }
  prompt += PROMPT_SUFFIX;

  std::vector<int> encoding;
  {
    uint64_t encode_start_ms = get_timestamp_ms();
    encoding = tokenizer.encode(prompt, true);
    uint64_t encode_end_ms = get_timestamp_ms();

    if (is_rank0()) {
      uint64_t encoding_ms = encode_end_ms - encode_start_ms;
      std::cout << fmt::format(
        "Encoding stats: ({} tokens, throughput: {:.5}tok/s, latency: {:.5}s/tok, total: {:.5}s)\n",
        encoding.size(),
        encoding.size() / (encoding_ms / 1000.0),
        (encoding_ms / 1000.0) / encoding.size(),
        encoding_ms / 1000.0
      ) << std::endl;
    }
  }

  // Allow max 16 steps to generate passkey
  const size_t MAX_GENERATION_STEPS = 16;

  if (is_rank0()) {
    std::cout << fmt::format(
      "Passkey test:\n"
      "  prompt: {} tokens\n"
      "  passkey: {}\n"
      "  passkey token index: ~{}\n",
      encoding.size(),
      passkey,
      (int)(((float)pos) / n_junk * encoding.size())
    ) << std::endl;
  }

  size_t N = encoding.size();
  for (size_t pos = 0; pos < N; pos++) {
    if (is_rank0()) {
      std::cout << "\r Running passkey test..." << pos + 1 << "/" << N << std::flush;
    }
    int token_id = encoding[pos];
    InferenceMode inferMode = pos + 1 == N ?
      InferenceMode::OUTPUT_LOGITS : InferenceMode::HYDRATE_KV_CACHE;
    size_t step_bytes = model.config->active_bytes(pos);
    prof.begin_forward(state.stream());
    model.forward(state, token_id, pos, inferMode);
    prof.end_forward(state.stream(), /*is_hydrate=*/true, step_bytes);
  }
  if (is_rank0()) {
    std::cout << std::endl;
    std::cout << PROMPT_SUFFIX << std::flush;
  }
  for (size_t pos = N; pos < N + MAX_GENERATION_STEPS; pos++) {
    int token_id = sampler.sample_argmax(state);
    if (is_rank0()) {
      std::string token_str = tokenizer.decode_one(encoding.back(), token_id);
      std::cout << token_str << std::flush;
    }
    encoding.push_back(token_id);
    if (token_id == tokenizer.eos_id || token_id == tokenizer.eot_id) {
      break;
    }
    size_t step_bytes = model.config->active_bytes(pos);
    prof.begin_forward(state.stream());
    model.forward(state, token_id, pos);
    prof.end_forward(state.stream(), /*is_hydrate=*/false, step_bytes);
  }
  if (is_rank0()) std::cout << std::endl;

  prof.synchronize();
  if (is_rank0()) prof.report();
}

int main(int argc, char* argv[]) {
  dist::init(&argc, &argv);

  std::string checkpoint_path = "";    // e.g. out/model.bin
  // Options
  std::string device = "cuda";         // cpu or cuda
  std::string mode = "completion";     // completion, passkey, or perplexity
  std::string prompt = "";             // prompt string
  std::string prompt_path = "";        // prompt file path
  int context = 0;
  // Completion mode options
  int num_steps = 256;                 // number of steps to run for
  float temperature = 1.0;             // temperature
  // Passkey mode options
  int n_junk = 250;                   // number of junk lines to insert
  int passkey_pos = -1;                 // passkey position (-1 - random)
  // Profiling
  bool profile = false;                 // enable CUDA-event profiler
  // Sampler seed. <0 means "pick fresh on rank 0 and broadcast" — preserves
  // the original behavior. Pass -s N for reproducible runs (essential for
  // cross-rank correctness comparisons).
  int64_t seed_arg = -1;

  if (argc >= 2) {
    checkpoint_path = argv[1];
  } else {
    error_usage();
  }
  for (int i = 2; i < argc;) {
    // basic shape validation
    if (argv[i][0] != '-') {
      error_usage();
    } // must start with dash
    if (strlen(argv[i]) != 2) {
      error_usage();
    } // must be -x (one dash, one letter)

    // boolean flags (no argument)
    if (argv[i][1] == 'p') {
      profile = true;
      i += 1;
      continue;
    }

    // flags below take exactly one argument
    if (i + 1 >= argc) {
      error_usage();
    }

    // read in the args
    if (argv[i][1] == 'h') {
      error_usage();
    } else if (argv[i][1] == 'm') {
      if (i + 1 >= argc) {
        error_usage();
      }
      mode = argv[i + 1];
      if (std::string("completion").starts_with(mode)) {
        mode = "completion";
      } else if (std::string("passkey").starts_with(mode)) {
        mode = "passkey";
      } else if (std::string("perplexity").starts_with(mode)) {
        mode = "perplexity";
      } else {
        error_usage();
      }
      i += 2;
    } else if (argv[i][1] == 'd') {
      if (i + 1 >= argc) {
        error_usage();
      }
      device = argv[i + 1];
      if (std::string("cpu").starts_with(device)) {
        device = "cpu";
      } else if (std::string("cuda").starts_with(device)) {
        device = "cuda";
      } else {
        error_usage();
      }
      i += 2;
    } else if (argv[i][1] == 'i') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt = argv[i + 1];
      i += 2;
    } else if (argv[i][1] == 't') {
      if (i + 1 >= argc) {
        error_usage();
      }
      temperature = std::stof(argv[i + 1]);
      i += 2;
    } else if (argv[i][1] == 'f') {
      if (i + 1 >= argc) {
        error_usage();
      }
      prompt_path = argv[i + 1];
      i += 2;
    } else if (argv[i][1] == 'T') {
      if (i + 1 >= argc) {
        error_usage();
      }
      context = std::stoi(argv[i + 1]);
      i += 2;
    } else if (argv[i][1] == 'l') {
      if (i + 1 >= argc) {
        error_usage();
      }
      passkey_pos = std::stoi(argv[i + 1]);
      i += 2;
    } else if (argv[i][1] == 'n') {
      if (i + 1 >= argc) {
        error_usage();
      }
      num_steps = std::stoi(argv[i + 1]);
      n_junk = num_steps;
      i += 2;
    } else if (argv[i][1] == 's') {
      if (i + 1 >= argc) {
        error_usage();
      }
      seed_arg = std::stoll(argv[i + 1]);
      i += 2;
    } else {
      error_usage();
    }
  }
  int has_prompt = prompt.size() > 0 ? 1 : 0;
  int has_prompt_path = prompt_path.size() > 0 ? 1 : 0;
  if (mode == "completion" || mode == "perplexity") {
    if ((has_prompt + has_prompt_path) != 1) {
      error_usage();
    } else if (has_prompt_path) {
      std::ifstream file(prompt_path);
      if (!file.is_open()) {
        std::cerr << "Error: could not open file " << prompt_path << std::endl;
        return 1;
      }

      std::stringstream buffer;
      buffer << file.rdbuf();
      prompt = buffer.str();
    }
  } else {
    if (passkey_pos != -1 && (passkey_pos >= n_junk || passkey_pos < 0)) {
      std::cerr << "Error: passkey position must be between 0 and " << n_junk - 1 << std::endl;
      return 1;
    }
  }

  // Resolve sampler seed. Rank 0 picks (timestamp if -s wasn't given) and
  // broadcasts. Every rank ends up with the same seed so RNG sequences match,
  // which means each rank samples the same token without an explicit token
  // broadcast.
  uint64_t seed = 0;
  if (is_rank0()) {
    seed = seed_arg >= 0 ? (uint64_t)seed_arg : get_timestamp_ms();
  }
  MPI_Bcast(&seed, 1, MPI_UINT64_T, 0, dist::comm());

  if (mode == "completion") {
    run_completion(checkpoint_path, device, prompt, context, num_steps, temperature, profile, seed);
  } else if (mode == "passkey") {
    run_passkey(checkpoint_path, device, context, n_junk, passkey_pos, profile, seed);
  } else if (mode == "perplexity") {
    run_perplexity(checkpoint_path, device, prompt, context, profile, seed);
  }

  dist::finalize();
  return 0;
}
