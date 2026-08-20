#define OVLLM_RUNTIME_ONLY
#include "transformer_block.cpp"

#include <cctype>
#include <cstdio>
#include <random>
#include <unordered_map>

#pragma pack(push, 1)
struct ModelHeader {
    char magic[8];
    uint32_t version;
    uint32_t dimension;
    uint32_t hidden_dimension;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t vocabulary;
    uint32_t max_sequence;
    uint32_t bos;
    uint32_t eos;
    uint32_t unknown;
    float rope_theta;
    float rms_epsilon;
};
#pragma pack(pop)
static_assert(sizeof(ModelHeader) == 60, "Unexpected model header layout");

struct GpuMatrix {
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t packed_stride = 0;
    Buffer scales;
    Buffer weights;
};

struct GpuLayer {
    Buffer attention_norm;
    Buffer feed_forward_norm;
    GpuMatrix qkv;
    GpuMatrix output;
    GpuMatrix gate_up;
    GpuMatrix down;
    Buffer attention_state;
};

struct GpuModel {
    ModelHeader header{};
    GpuMatrix token_embedding;
    Buffer final_norm;
    std::vector<GpuLayer> layers;
    uint64_t buffer_bytes = 0;
};

template <typename T>
static std::vector<T> read_values(std::ifstream& input, size_t count, const char* description) {
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
    if (!input) throw std::runtime_error(std::string("Truncated model while reading ") + description);
    return values;
}

static GpuMatrix load_matrix(std::ifstream& input, const Runtime& runtime,
                             uint32_t rows, uint32_t columns, uint64_t& buffer_bytes) {
    GpuMatrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.packed_stride = (columns + 3) / 4;
    const std::vector<float> scales = read_values<float>(input, rows, "matrix scales");
    const std::vector<uint32_t> weights = read_values<uint32_t>(
        input, static_cast<size_t>(rows) * matrix.packed_stride, "packed matrix weights");
    matrix.scales = upload_vector(runtime, scales);
    matrix.weights = upload_vector(runtime, weights);
    buffer_bytes += matrix.scales.size + matrix.weights.size;
    return matrix;
}

static Buffer load_float_buffer(std::ifstream& input, const Runtime& runtime, size_t count,
                                uint64_t& buffer_bytes, const char* description) {
    const std::vector<float> values = read_values<float>(input, count, description);
    Buffer buffer = upload_vector(runtime, values);
    buffer_bytes += buffer.size;
    return buffer;
}

static GpuModel load_model(const std::string& path, const Runtime& runtime) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Could not open converted model: " + path);
    GpuModel model;
    input.read(reinterpret_cast<char*>(&model.header), sizeof(model.header));
    if (!input || std::memcmp(model.header.magic, "OVLLM3\0\0", 8) != 0 ||
        model.header.version != 1) {
        throw std::runtime_error("Unsupported converted model format");
    }
    if (model.header.heads != model.header.kv_heads) {
        throw std::runtime_error("This milestone build expects equal query and KV head counts");
    }
    if (model.header.max_sequence > 256) {
        throw std::runtime_error("This milestone build supports at most 256 cached positions");
    }

    const uint32_t dimension = model.header.dimension;
    const uint32_t hidden = model.header.hidden_dimension;
    model.token_embedding = load_matrix(input, runtime, model.header.vocabulary, dimension,
                                        model.buffer_bytes);
    model.final_norm = load_float_buffer(input, runtime, dimension, model.buffer_bytes, "final norm");
    model.layers.resize(model.header.layers);
    for (uint32_t layer_index = 0; layer_index < model.header.layers; ++layer_index) {
        GpuLayer& layer = model.layers[layer_index];
        layer.attention_norm = load_float_buffer(input, runtime, dimension, model.buffer_bytes,
                                                 "attention norm");
        layer.feed_forward_norm = load_float_buffer(input, runtime, dimension, model.buffer_bytes,
                                                    "feed-forward norm");
        layer.qkv = load_matrix(input, runtime, 3 * dimension, dimension, model.buffer_bytes);
        layer.output = load_matrix(input, runtime, dimension, dimension, model.buffer_bytes);
        layer.gate_up = load_matrix(input, runtime, 2 * hidden, dimension, model.buffer_bytes);
        layer.down = load_matrix(input, runtime, dimension, hidden, model.buffer_bytes);
        const VkDeviceSize state_size = static_cast<VkDeviceSize>(
            dimension + 2ull * model.header.max_sequence * dimension) * sizeof(float);
        layer.attention_state = create_buffer(runtime, state_size);
        model.buffer_bytes += state_size;
    }
    char extra = 0;
    if (input.read(&extra, 1)) throw std::runtime_error("Converted model has unexpected trailing data");
    return model;
}

struct TokenPiece {
    std::string text;
    float score = 0.0f;
    uint32_t flags = 0;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open tokenizer: " + path);
        char magic[8]{};
        uint32_t version = 0;
        uint32_t vocabulary = 0;
        input.read(magic, 8);
        input.read(reinterpret_cast<char*>(&version), 4);
        input.read(reinterpret_cast<char*>(&vocabulary), 4);
        input.read(reinterpret_cast<char*>(&bos_), 4);
        input.read(reinterpret_cast<char*>(&eos_), 4);
        input.read(reinterpret_cast<char*>(&unknown_), 4);
        if (!input || std::memcmp(magic, "OVTOK3\0\0", 8) != 0 || version != 1) {
            throw std::runtime_error("Unsupported tokenizer format");
        }
        pieces_.resize(vocabulary);
        for (uint32_t token = 0; token < vocabulary; ++token) {
            uint32_t length = 0;
            input.read(reinterpret_cast<char*>(&pieces_[token].score), 4);
            input.read(reinterpret_cast<char*>(&pieces_[token].flags), 4);
            input.read(reinterpret_cast<char*>(&length), 4);
            pieces_[token].text.resize(length);
            input.read(pieces_[token].text.data(), length);
            if (!input) throw std::runtime_error("Truncated tokenizer file");
            lookup_[pieces_[token].text] = token;
        }
    }

    std::vector<uint32_t> encode(const std::string& text, bool add_bos,
                                 bool prepend_space_marker = true) const {
        static const std::string space_marker = "\xe2\x96\x81";
        std::string normalized = prepend_space_marker ? space_marker : std::string();
        bool previous_space = false;
        for (unsigned char byte : text) {
            if (std::isspace(byte)) {
                if (!previous_space) normalized += space_marker;
                previous_space = true;
            } else {
                normalized.push_back(static_cast<char>(byte));
                previous_space = false;
            }
        }

        std::vector<uint32_t> tokens;
        std::vector<std::string> symbols;
        if (add_bos) tokens.push_back(bos_);
        size_t offset = 0;
        while (offset < normalized.size()) {
            const unsigned char lead = static_cast<unsigned char>(normalized[offset]);
            size_t length = 1;
            if ((lead & 0xe0u) == 0xc0u) length = 2;
            else if ((lead & 0xf0u) == 0xe0u) length = 3;
            else if ((lead & 0xf8u) == 0xf0u) length = 4;
            length = std::min(length, normalized.size() - offset);
            const std::string symbol = normalized.substr(offset, length);
            const auto found = lookup_.find(symbol);
            if (found != lookup_.end()) {
                tokens.push_back(found->second);
                symbols.push_back(symbol);
            } else {
                for (size_t byte_index = 0; byte_index < length; ++byte_index) {
                    char byte_piece[7];
                    std::snprintf(byte_piece, sizeof(byte_piece), "<0x%02X>",
                                  static_cast<unsigned char>(symbol[byte_index]));
                    const auto byte_found = lookup_.find(byte_piece);
                    tokens.push_back(byte_found == lookup_.end() ? unknown_ : byte_found->second);
                    symbols.push_back(byte_piece);
                }
            }
            offset += length;
        }

        const size_t prefix = add_bos ? 1 : 0;
        while (symbols.size() >= 2) {
            float best_score = -std::numeric_limits<float>::infinity();
            size_t best_index = symbols.size();
            uint32_t best_token = unknown_;
            std::string best_symbol;
            for (size_t index = 0; index + 1 < symbols.size(); ++index) {
                const std::string combined = symbols[index] + symbols[index + 1];
                const auto found = lookup_.find(combined);
                if (found != lookup_.end() && pieces_[found->second].score > best_score) {
                    best_score = pieces_[found->second].score;
                    best_index = index;
                    best_token = found->second;
                    best_symbol = combined;
                }
            }
            if (best_index == symbols.size()) break;
            symbols[best_index] = std::move(best_symbol);
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
            tokens[prefix + best_index] = best_token;
            tokens.erase(tokens.begin() + static_cast<std::ptrdiff_t>(prefix + best_index + 1));
        }
        return tokens;
    }

    std::string decode(const std::vector<uint32_t>& tokens) const {
        static const std::string space_marker = "\xe2\x96\x81";
        std::string result;
        for (uint32_t token : tokens) {
            if (token >= pieces_.size()) continue;
            const TokenPiece& piece = pieces_[token];
            if (piece.flags & 2u) continue;
            if (piece.flags & 4u) {
                unsigned int value = 0;
                if (std::sscanf(piece.text.c_str(), "<0x%02X>", &value) == 1) {
                    result.push_back(static_cast<char>(value));
                }
                continue;
            }
            if (piece.flags & 1u) {
                result += "\xef\xbf\xbd";
                continue;
            }
            std::string text = piece.text;
            size_t position = 0;
            while ((position = text.find(space_marker, position)) != std::string::npos) {
                text.replace(position, space_marker.size(), " ");
                ++position;
            }
            result += text;
        }
        if (!result.empty() && result.front() == ' ') result.erase(result.begin());
        return result;
    }

    uint32_t eos() const { return eos_; }
    size_t vocabulary_size() const { return pieces_.size(); }
    uint32_t token_id(const std::string& piece) const {
        const auto found = lookup_.find(piece);
        if (found == lookup_.end()) throw std::runtime_error("Tokenizer piece is unavailable: " + piece);
        return found->second;
    }

private:
    std::vector<TokenPiece> pieces_;
    std::unordered_map<std::string, uint32_t> lookup_;
    uint32_t bos_ = 1;
    uint32_t eos_ = 2;
    uint32_t unknown_ = 0;
};

struct LayerSets {
    VkDescriptorSet norm1;
    VkDescriptorSet qkv;
    VkDescriptorSet rope;
    VkDescriptorSet attention;
    VkDescriptorSet output;
    VkDescriptorSet residual1;
    VkDescriptorSet norm2;
    VkDescriptorSet gate_up;
    VkDescriptorSet swiglu;
    VkDescriptorSet down;
    VkDescriptorSet residual2;
};

struct EmbeddingPush { uint32_t token, dimension, packed_stride, unused; };
struct CachePush { uint32_t position, model_dimension, head_dimension, kv_dimension; };

static uint32_t sample_token(const std::vector<float>& logits, std::mt19937& random) {
    constexpr float temperature = 0.8f;
    constexpr float top_p = 0.9f;
    constexpr size_t top_k = 40;
    std::vector<uint32_t> indices(logits.size());
    for (uint32_t index = 0; index < indices.size(); ++index) indices[index] = index;
    std::partial_sort(indices.begin(), indices.begin() + std::min(top_k, indices.size()), indices.end(),
        [&](uint32_t left, uint32_t right) { return logits[left] > logits[right]; });
    const size_t candidate_count = std::min(top_k, indices.size());
    const float maximum = logits[indices[0]] / temperature;
    std::vector<float> probabilities(candidate_count);
    float total = 0.0f;
    for (size_t index = 0; index < candidate_count; ++index) {
        probabilities[index] = std::exp(logits[indices[index]] / temperature - maximum);
        total += probabilities[index];
    }
    float cumulative = 0.0f;
    size_t retained = candidate_count;
    for (size_t index = 0; index < candidate_count; ++index) {
        cumulative += probabilities[index] / total;
        if (cumulative >= top_p) {
            retained = index + 1;
            break;
        }
    }
    float retained_total = 0.0f;
    for (size_t index = 0; index < retained; ++index) retained_total += probabilities[index];
    std::uniform_real_distribution<float> distribution(0.0f, retained_total);
    float target = distribution(random);
    for (size_t index = 0; index < retained; ++index) {
        target -= probabilities[index];
        if (target <= 0.0f) return indices[index];
    }
    return indices[retained - 1];
}

static uint32_t greedy_token(const std::vector<float>& logits) {
    return static_cast<uint32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static void dump_logits(const std::string& path, const std::vector<float>& logits) {
    if (path.empty()) return;
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Could not create logits dump: " + path);
    output.write(reinterpret_cast<const char*>(logits.data()),
                 static_cast<std::streamsize>(logits.size() * sizeof(float)));
    if (!output) throw std::runtime_error("Could not write logits dump: " + path);
}

static void destroy_matrix(const Runtime& runtime, GpuMatrix& matrix) {
    destroy_buffer(runtime, matrix.weights);
    destroy_buffer(runtime, matrix.scales);
}

#ifndef OVLLM_STREAMING_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\tinystories-llama-15m\\runtime";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "Once upon a time, there was a little girl named Lily.";
        const uint32_t requested_tokens = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 48;
        const bool greedy = argc > 5 && std::string(argv[5]) == "greedy";
        const std::string logits_dump_path = argc > 6 ? argv[6] : "";

        Runtime runtime = create_runtime();
        GpuModel model = load_model(model_directory + "\\model.ovm", runtime);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        if (tokenizer.vocabulary_size() != model.header.vocabulary) {
            throw std::runtime_error("Model/tokenizer vocabulary mismatch");
        }
        const uint32_t dimension = model.header.dimension;
        const uint32_t hidden = model.header.hidden_dimension;
        const uint32_t head_dimension = dimension / model.header.heads;

        std::vector<float> rope_cos(static_cast<size_t>(model.header.max_sequence) * head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t position = 0; position < model.header.max_sequence; ++position) {
            for (uint32_t frequency = 0; frequency < head_dimension / 2; ++frequency) {
                const float inverse_frequency = std::pow(model.header.rope_theta,
                    -2.0f * static_cast<float>(frequency) / static_cast<float>(head_dimension));
                const float angle = static_cast<float>(position) * inverse_frequency;
                rope_cos[static_cast<size_t>(position) * (head_dimension / 2) + frequency] = std::cos(angle);
                rope_sin[static_cast<size_t>(position) * (head_dimension / 2) + frequency] = std::sin(angle);
            }
        }
        Buffer rope_cos_buffer = upload_vector(runtime, rope_cos);
        Buffer rope_sin_buffer = upload_vector(runtime, rope_sin);
        Buffer dummy = create_buffer(runtime, sizeof(float));
        Buffer hidden_a = create_buffer(runtime, dimension * sizeof(float));
        Buffer hidden_b = create_buffer(runtime, dimension * sizeof(float));
        Buffer norm = create_buffer(runtime, dimension * sizeof(float));
        Buffer qkv = create_buffer(runtime, 3 * dimension * sizeof(float));
        Buffer context = create_buffer(runtime, dimension * sizeof(float));
        Buffer projection = create_buffer(runtime, dimension * sizeof(float));
        Buffer gate_up = create_buffer(runtime, 2 * hidden * sizeof(float));
        Buffer feed_forward = create_buffer(runtime, hidden * sizeof(float));
        Buffer down = create_buffer(runtime, dimension * sizeof(float));
        Buffer logits = create_buffer(runtime, model.header.vocabulary * sizeof(float));
        std::array<Buffer*, 12> auxiliary_buffers = {
            &rope_cos_buffer, &rope_sin_buffer, &dummy, &hidden_a, &hidden_b, &norm,
            &qkv, &context, &projection, &gate_up, &feed_forward, &down};
        for (const Buffer* buffer : auxiliary_buffers) model.buffer_bytes += buffer->size;
        model.buffer_bytes += logits.size;

        const uint32_t descriptor_count = 3 + 11 * model.header.layers;
        ComputeResources resources = create_compute_resources(runtime, descriptor_count);
        const auto shader = [&](const char* name) {
            return shader_directory + "\\" + name + ".comp.spv";
        };
        const VkPipeline embedding_pipeline = create_pipeline(runtime, resources, shader("embedding"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline qgemm_pipeline = create_pipeline(runtime, resources, shader("qgemm"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope_cache"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention_cache"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));

        const VkDescriptorSet embedding_set = create_descriptor_set(runtime, resources,
            {&model.token_embedding.weights, &model.token_embedding.scales, &hidden_a, &dummy});
        std::vector<LayerSets> layer_sets(model.header.layers);
        for (uint32_t index = 0; index < model.header.layers; ++index) {
            GpuLayer& layer = model.layers[index];
            LayerSets& sets = layer_sets[index];
            sets.norm1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &layer.attention_norm, &norm, &dummy});
            sets.qkv = create_descriptor_set(runtime, resources,
                {&norm, &layer.qkv.weights, &layer.qkv.scales, &qkv});
            sets.rope = create_descriptor_set(runtime, resources,
                {&qkv, &rope_cos_buffer, &rope_sin_buffer, &layer.attention_state});
            sets.attention = create_descriptor_set(runtime, resources,
                {&layer.attention_state, &context, &dummy, &dummy});
            sets.output = create_descriptor_set(runtime, resources,
                {&context, &layer.output.weights, &layer.output.scales, &projection});
            sets.residual1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &projection, &hidden_b, &dummy});
            sets.norm2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &layer.feed_forward_norm, &norm, &dummy});
            sets.gate_up = create_descriptor_set(runtime, resources,
                {&norm, &layer.gate_up.weights, &layer.gate_up.scales, &gate_up});
            sets.swiglu = create_descriptor_set(runtime, resources,
                {&gate_up, &feed_forward, &dummy, &dummy});
            sets.down = create_descriptor_set(runtime, resources,
                {&feed_forward, &layer.down.weights, &layer.down.scales, &down});
            sets.residual2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &down, &hidden_a, &dummy});
        }
        const VkDescriptorSet final_norm_set = create_descriptor_set(runtime, resources,
            {&hidden_a, &model.final_norm, &norm, &dummy});
        const VkDescriptorSet logits_set = create_descriptor_set(runtime, resources,
            {&norm, &model.token_embedding.weights, &model.token_embedding.scales, &logits});

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = runtime.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &command_pool_info, nullptr, &command_pool));

        std::vector<float> host_logits(model.header.vocabulary);
        auto run_token = [&](uint32_t token, uint32_t position) {
            VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocation.commandPool = command_pool;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command_buffer));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkfn::BeginCommandBuffer(command_buffer, &begin));
            VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &upload, 0, nullptr, 0, nullptr);

            const EmbeddingPush embedding_push{token, dimension,
                                                model.token_embedding.packed_stride, 0};
            dispatch(command_buffer, resources, embedding_pipeline, embedding_set, &embedding_push,
                     (dimension + 63) / 64, 1);
            compute_barrier(command_buffer);
            const RmsPush rms_push{1, dimension, model.header.rms_epsilon, 0};
            const CachePush cache_push{position, dimension, head_dimension, dimension};
            const AddPush add_push{dimension, 0, 0, 0};
            const SwiGluPush swiglu_push{1, hidden, 0, 0};
            for (uint32_t index = 0; index < model.header.layers; ++index) {
                const GpuLayer& layer = model.layers[index];
                const LayerSets& sets = layer_sets[index];
                dispatch(command_buffer, resources, rmsnorm_pipeline, sets.norm1, &rms_push, 1, 1);
                compute_barrier(command_buffer);
                const LinearPush qkv_push{1, 3 * dimension, dimension, layer.qkv.packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets.qkv, &qkv_push,
                         (3 * dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, rope_pipeline, sets.rope, &cache_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, attention_pipeline, sets.attention, &cache_push,
                         model.header.heads, 1);
                compute_barrier(command_buffer);
                const LinearPush output_push{1, dimension, dimension, layer.output.packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets.output, &output_push,
                         (dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, add_pipeline, sets.residual1, &add_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, rmsnorm_pipeline, sets.norm2, &rms_push, 1, 1);
                compute_barrier(command_buffer);
                const LinearPush gate_up_push{1, 2 * hidden, dimension, layer.gate_up.packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets.gate_up, &gate_up_push,
                         (2 * hidden + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, swiglu_pipeline, sets.swiglu, &swiglu_push,
                         (hidden + 63) / 64, 1);
                compute_barrier(command_buffer);
                const LinearPush down_push{1, dimension, hidden, layer.down.packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets.down, &down_push,
                         (dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, add_pipeline, sets.residual2, &add_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
            }
            dispatch(command_buffer, resources, rmsnorm_pipeline, final_norm_set, &rms_push, 1, 1);
            compute_barrier(command_buffer);
            const LinearPush logits_push{1, model.header.vocabulary, dimension,
                                         model.token_embedding.packed_stride};
            dispatch(command_buffer, resources, qgemm_pipeline, logits_set, &logits_push,
                     (model.header.vocabulary + 15) / 16, 1);
            VkMemoryBarrier download{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            download.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            download.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT, 0,
                                     1, &download, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command_buffer));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
            VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
            invalidate_buffer(runtime, logits);
            std::memcpy(host_logits.data(), logits.mapped, static_cast<size_t>(logits.size));
        };

        std::vector<uint32_t> tokens = tokenizer.encode(prompt, true);
        if (tokens.empty()) throw std::runtime_error("Prompt encoded to no tokens");
        if (tokens.size() + requested_tokens > model.header.max_sequence) {
            throw std::runtime_error("Prompt plus requested generation exceeds the model context");
        }
        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Vendor/device ID: 0x" << std::hex << runtime.properties.vendorID << ":0x"
                  << runtime.properties.deviceID << std::dec << "\n"
                  << "Model: sdobson/tinystories-llama-15m (15.2M pretrained parameters)\n"
                  << "Architecture: " << model.header.layers << " layers, d=" << dimension
                  << ", heads=" << model.header.heads << ", ffn=" << hidden << "\n"
                  << "Quantization: symmetric INT8 per output row; FP32 scales/norms/activations\n"
                  << "Vulkan buffer payload: " << std::fixed << std::setprecision(2)
                  << (static_cast<double>(model.buffer_bytes) / (1024.0 * 1024.0)) << " MiB\n"
                  << "Prompt tokens: " << tokens.size() << "\n";

        for (uint32_t position = 0; position < tokens.size(); ++position) {
            run_token(tokens[position], position);
        }
        dump_logits(logits_dump_path, host_logits);
        const size_t prompt_token_count = tokens.size();
        std::mt19937 random(0x6700u);
        uint32_t generated = 0;
        while (generated < requested_tokens && tokens.size() < model.header.max_sequence) {
            const uint32_t next = greedy ? greedy_token(host_logits) : sample_token(host_logits, random);
            if (next == tokenizer.eos()) break;
            tokens.push_back(next);
            ++generated;
            if (generated < requested_tokens) run_token(next, static_cast<uint32_t>(tokens.size() - 1));
        }
        const std::string generated_text = tokenizer.decode(tokens);
        std::cout << "Generated tokens: " << generated << "\n"
                  << "Generated token IDs:";
        for (size_t index = prompt_token_count; index < tokens.size(); ++index) {
            std::cout << " " << tokens[index];
        }
        std::cout << "\nPeak Vulkan buffer allocations: " << std::fixed << std::setprecision(3)
                  << (peak_vulkan_buffer_bytes / (1024.0 * 1024.0)) << " MiB\n"
                  << "--- generated text ---\n" << generated_text << "\n--- end ---\n"
                  << "RESULT: PASS - pretrained LLM generated text through our Vulkan runtime\n";

        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines) vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        for (VkShaderModule module : resources.shader_modules) vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        destroy_buffer(runtime, logits);
        for (Buffer* buffer : auxiliary_buffers) destroy_buffer(runtime, *buffer);
        for (GpuLayer& layer : model.layers) {
            destroy_buffer(runtime, layer.attention_state);
            destroy_matrix(runtime, layer.down);
            destroy_matrix(runtime, layer.gate_up);
            destroy_matrix(runtime, layer.output);
            destroy_matrix(runtime, layer.qkv);
            destroy_buffer(runtime, layer.feed_forward_norm);
            destroy_buffer(runtime, layer.attention_norm);
        }
        destroy_buffer(runtime, model.final_norm);
        destroy_matrix(runtime, model.token_embedding);
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
#endif
