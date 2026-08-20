#define OVLLM_M5_RUNTIME_ONLY
#include "streaming_llm.cpp"

#include <chrono>

constexpr uint32_t M6_CACHE_SEQUENCE = 256;
constexpr uint32_t M6_LOGIT_TILE_ROWS = 1024;
constexpr uint64_t M6_WEIGHT_BUDGET = 768ull * 1024;

#pragma pack(push, 1)
struct MoEExtension {
    uint32_t experts;
    uint32_t top_k;
    float rope_factor;
};
#pragma pack(pop)
static_assert(sizeof(MoEExtension) == 12, "Unexpected MoE extension layout");

struct ExpertIndex {
    MatrixIndex gate_up;
    MatrixIndex down;
};

struct MoELayerIndex {
    uint64_t attention_norm_offset = 0;
    uint64_t feed_forward_norm_offset = 0;
    MatrixIndex qkv;
    MatrixIndex output;
    uint64_t router_offset = 0;
    std::vector<ExpertIndex> experts;
};

struct MoEModelIndex {
    ModelHeader header{};
    MoEExtension extension{};
    MatrixIndex token_embedding;
    MatrixIndex lm_head;
    uint64_t final_norm_offset = 0;
    std::vector<MoELayerIndex> layers;
    uint64_t matrix_bytes = 0;
    uint64_t norm_bytes = 0;
    uint64_t router_bytes = 0;
    uint64_t file_bytes = 0;
    uint64_t total_parameters = 0;
    uint64_t expert_parameters = 0;
};

static MatrixIndex index_moe_matrix(uint64_t& cursor, uint32_t rows, uint32_t columns,
                                    MoEModelIndex& index, bool expert) {
    MatrixIndex matrix = index_matrix(cursor, rows, columns, index.matrix_bytes);
    const uint64_t parameters = static_cast<uint64_t>(rows) * columns;
    index.total_parameters += parameters;
    if (expert) index.expert_parameters += parameters;
    return matrix;
}

static MoEModelIndex index_moe_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open converted MoE model: " + path);
    MoEModelIndex index;
    index.file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    input.read(reinterpret_cast<char*>(&index.extension), sizeof(index.extension));
    if (!input || std::memcmp(index.header.magic, "OVLLM3\0\0", 8) != 0 ||
        index.header.version != 3 || index.extension.experts < 2 ||
        index.extension.experts > 16 || index.extension.top_k != 2) {
        throw std::runtime_error("Unsupported converted MoE model format");
    }
    if (index.header.max_sequence < M6_CACHE_SEQUENCE) {
        throw std::runtime_error("MoE model context is smaller than the runtime KV cache");
    }
    uint64_t cursor = sizeof(ModelHeader) + sizeof(MoEExtension);
    const uint32_t dimension = index.header.dimension;
    const uint32_t hidden = index.header.hidden_dimension;
    index.token_embedding = index_moe_matrix(cursor, index.header.vocabulary, dimension,
                                             index, false);
    index.lm_head = index_moe_matrix(cursor, index.header.vocabulary, dimension, index, false);
    index.final_norm_offset = cursor;
    cursor += static_cast<uint64_t>(dimension) * sizeof(float);
    index.norm_bytes += static_cast<uint64_t>(dimension) * sizeof(float);
    index.total_parameters += dimension;
    index.layers.resize(index.header.layers);
    const uint32_t head_dimension = dimension / index.header.heads;
    const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
    for (MoELayerIndex& layer : index.layers) {
        layer.attention_norm_offset = cursor;
        cursor += static_cast<uint64_t>(dimension) * sizeof(float);
        layer.feed_forward_norm_offset = cursor;
        cursor += static_cast<uint64_t>(dimension) * sizeof(float);
        index.norm_bytes += static_cast<uint64_t>(2) * dimension * sizeof(float);
        index.total_parameters += static_cast<uint64_t>(2) * dimension;
        layer.qkv = index_moe_matrix(cursor, dimension + 2 * kv_dimension, dimension,
                                     index, false);
        layer.output = index_moe_matrix(cursor, dimension, dimension, index, false);
        layer.router_offset = cursor;
        const uint64_t router_values = static_cast<uint64_t>(index.extension.experts) * dimension;
        cursor += router_values * sizeof(float);
        index.router_bytes += router_values * sizeof(float);
        index.total_parameters += router_values;
        layer.experts.resize(index.extension.experts);
        for (ExpertIndex& expert : layer.experts) {
            expert.gate_up = index_moe_matrix(cursor, 2 * hidden, dimension, index, true);
            expert.down = index_moe_matrix(cursor, dimension, hidden, index, true);
        }
    }
    if (cursor != index.file_bytes) {
        throw std::runtime_error("Converted MoE model size does not match its indexed layout");
    }
    return index;
}

struct ResidentMoELayer {
    Buffer attention_norm;
    Buffer feed_forward_norm;
    Buffer router;
    Buffer attention_state;
};

struct ResidentMoEModel {
    Buffer final_norm;
    std::vector<ResidentMoELayer> layers;
};

static ResidentMoEModel load_moe_resident_state(const std::string& path, const Runtime& runtime,
                                                const MoEModelIndex& index) {
    std::ifstream input(path, std::ios::binary);
    ResidentMoEModel model;
    const uint32_t dimension = index.header.dimension;
    const uint32_t head_dimension = dimension / index.header.heads;
    const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
    model.final_norm = load_norm_at(input, runtime, index.final_norm_offset, dimension);
    model.layers.resize(index.header.layers);
    for (uint32_t number = 0; number < index.header.layers; ++number) {
        const MoELayerIndex& source = index.layers[number];
        ResidentMoELayer& layer = model.layers[number];
        layer.attention_norm = load_norm_at(input, runtime, source.attention_norm_offset, dimension);
        layer.feed_forward_norm = load_norm_at(input, runtime, source.feed_forward_norm_offset,
                                                dimension);
        std::vector<float> router(static_cast<size_t>(index.extension.experts) * dimension);
        read_file_at(input, source.router_offset, router.data(), router.size() * sizeof(float),
                     "MoE router weights");
        layer.router = upload_vector(runtime, router);
        const VkDeviceSize state_size = static_cast<VkDeviceSize>(
            dimension + 2ull * M6_CACHE_SEQUENCE * kv_dimension) * sizeof(float);
        layer.attention_state = create_buffer(runtime, state_size);
    }
    return model;
}

struct MoELayerSets {
    VkDescriptorSet norm1 = VK_NULL_HANDLE;
    VkDescriptorSet qkv = VK_NULL_HANDLE;
    VkDescriptorSet rope = VK_NULL_HANDLE;
    VkDescriptorSet attention = VK_NULL_HANDLE;
    VkDescriptorSet output = VK_NULL_HANDLE;
    VkDescriptorSet residual1 = VK_NULL_HANDLE;
    VkDescriptorSet norm2 = VK_NULL_HANDLE;
    VkDescriptorSet router = VK_NULL_HANDLE;
    VkDescriptorSet gate_up = VK_NULL_HANDLE;
    VkDescriptorSet swiglu = VK_NULL_HANDLE;
    VkDescriptorSet down = VK_NULL_HANDLE;
    VkDescriptorSet weighted = VK_NULL_HANDLE;
    VkDescriptorSet residual2 = VK_NULL_HANDLE;
};

struct RouterPush { uint32_t dimension, experts, top_k, unused; };
struct WeightedPush { uint32_t count; float weight; uint32_t clear, unused; };

#ifndef OVLLM_M7_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\tinymoe-100m-2x8-chat\\runtime";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "In a small village, there lived a friendly dragon named";
        const uint32_t requested_tokens = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 24;
        const bool stochastic = argc > 5 && std::string(argv[5]) == "sample";
        const bool chat_template = argc > 6 && std::string(argv[6]) == "chat";
        const std::string model_path = model_directory + "\\model.ovm";

        Runtime runtime = create_runtime();
        const MoEModelIndex index = index_moe_model(model_path);
        ResidentMoEModel model = load_moe_resident_state(model_path, runtime, index);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        if (tokenizer.vocabulary_size() != index.header.vocabulary) {
            throw std::runtime_error("MoE model/tokenizer vocabulary mismatch");
        }
        const uint32_t dimension = index.header.dimension;
        const uint32_t hidden = index.header.hidden_dimension;
        const uint32_t head_dimension = dimension / index.header.heads;
        const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
        const uint32_t qkv_dimension = dimension + 2 * kv_dimension;

        uint32_t maximum_scale_rows = M6_LOGIT_TILE_ROWS;
        uint32_t maximum_packed_values = M6_LOGIT_TILE_ROWS * index.lm_head.packed_stride;
        const auto include_matrix = [&](const MatrixIndex& matrix) {
            maximum_scale_rows = std::max(maximum_scale_rows, matrix.rows);
            maximum_packed_values = std::max(maximum_packed_values,
                                              matrix.rows * matrix.packed_stride);
        };
        for (const MoELayerIndex& layer : index.layers) {
            include_matrix(layer.qkv);
            include_matrix(layer.output);
            for (const ExpertIndex& expert : layer.experts) {
                include_matrix(expert.gate_up);
                include_matrix(expert.down);
            }
        }
        WeightStreamer weights(model_path, runtime, maximum_scale_rows, maximum_packed_values);
        const uint64_t learned_weight_residency = weights.capacity_bytes() + index.norm_bytes +
                                                  index.router_bytes;
        if (learned_weight_residency > M6_WEIGHT_BUDGET) {
            throw std::runtime_error("MoE learned weights exceed the 768 KiB residency budget");
        }

        std::vector<float> rope_cos(static_cast<size_t>(M6_CACHE_SEQUENCE) * head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t position = 0; position < M6_CACHE_SEQUENCE; ++position) {
            for (uint32_t frequency = 0; frequency < head_dimension / 2; ++frequency) {
                const float inverse_frequency = std::pow(index.header.rope_theta,
                    -2.0f * static_cast<float>(frequency) / static_cast<float>(head_dimension));
                const float angle = (static_cast<float>(position) / index.extension.rope_factor) *
                                    inverse_frequency;
                const size_t offset = static_cast<size_t>(position) * (head_dimension / 2) + frequency;
                rope_cos[offset] = std::cos(angle);
                rope_sin[offset] = std::sin(angle);
            }
        }
        Buffer rope_cos_buffer = upload_vector(runtime, rope_cos);
        Buffer rope_sin_buffer = upload_vector(runtime, rope_sin);
        Buffer dummy = create_buffer(runtime, sizeof(float));
        Buffer hidden_a = create_buffer(runtime, dimension * sizeof(float));
        Buffer hidden_b = create_buffer(runtime, dimension * sizeof(float));
        Buffer norm = create_buffer(runtime, dimension * sizeof(float));
        Buffer qkv = create_buffer(runtime, qkv_dimension * sizeof(float));
        Buffer context = create_buffer(runtime, dimension * sizeof(float));
        Buffer projection = create_buffer(runtime, dimension * sizeof(float));
        Buffer gate_up = create_buffer(runtime, 2 * hidden * sizeof(float));
        Buffer feed_forward = create_buffer(runtime, hidden * sizeof(float));
        Buffer expert_output = create_buffer(runtime, dimension * sizeof(float));
        Buffer moe_sum = create_buffer(runtime, dimension * sizeof(float));
        Buffer routing = create_buffer(runtime, 4 * sizeof(float));
        Buffer logit_output = create_buffer(runtime, M6_LOGIT_TILE_ROWS * sizeof(float));
        std::array<Buffer*, 13> auxiliary_buffers = {
            &rope_cos_buffer, &rope_sin_buffer, &dummy, &hidden_a, &hidden_b, &norm, &qkv,
            &context, &projection, &gate_up, &feed_forward, &expert_output, &moe_sum};

        ComputeResources resources = create_compute_resources(runtime, 4 + 13 * index.header.layers);
        const auto shader = [&](const char* name) { return shader_directory + "\\" + name + ".comp.spv"; };
        const VkPipeline embedding_pipeline = create_pipeline(runtime, resources, shader("embedding"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline qgemm_pipeline = create_pipeline(runtime, resources, shader("qgemm"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope_cache"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention_cache"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));
        const VkPipeline router_pipeline = create_pipeline(runtime, resources, shader("moe_router"));
        const VkPipeline weighted_pipeline = create_pipeline(runtime, resources,
                                                              shader("weighted_accumulate"));

        const VkDescriptorSet embedding_set = create_descriptor_set(runtime, resources,
            {&weights.weights(), &weights.scales(), &hidden_a, &dummy});
        const VkDescriptorSet logits_set = create_descriptor_set(runtime, resources,
            {&norm, &weights.weights(), &weights.scales(), &logit_output});
        std::vector<MoELayerSets> sets(index.header.layers);
        for (uint32_t number = 0; number < index.header.layers; ++number) {
            ResidentMoELayer& layer = model.layers[number];
            MoELayerSets& set = sets[number];
            set.norm1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &layer.attention_norm, &norm, &dummy});
            set.qkv = create_descriptor_set(runtime, resources,
                {&norm, &weights.weights(), &weights.scales(), &qkv});
            set.rope = create_descriptor_set(runtime, resources,
                {&qkv, &rope_cos_buffer, &rope_sin_buffer, &layer.attention_state});
            set.attention = create_descriptor_set(runtime, resources,
                {&layer.attention_state, &context, &dummy, &dummy});
            set.output = create_descriptor_set(runtime, resources,
                {&context, &weights.weights(), &weights.scales(), &projection});
            set.residual1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &projection, &hidden_b, &dummy});
            set.norm2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &layer.feed_forward_norm, &norm, &dummy});
            set.router = create_descriptor_set(runtime, resources,
                {&norm, &layer.router, &routing, &dummy});
            set.gate_up = create_descriptor_set(runtime, resources,
                {&norm, &weights.weights(), &weights.scales(), &gate_up});
            set.swiglu = create_descriptor_set(runtime, resources,
                {&gate_up, &feed_forward, &dummy, &dummy});
            set.down = create_descriptor_set(runtime, resources,
                {&feed_forward, &weights.weights(), &weights.scales(), &expert_output});
            set.weighted = create_descriptor_set(runtime, resources,
                {&expert_output, &moe_sum, &dummy, &dummy});
            set.residual2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &moe_sum, &hidden_a, &dummy});
        }
        const VkDescriptorSet final_norm_set = create_descriptor_set(runtime, resources,
            {&hidden_a, &model.final_norm, &norm, &dummy});

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = runtime.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &command_pool));
        const auto execute = [&](const auto& record) {
            VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocation.commandPool = command_pool;
            allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocation.commandBufferCount = 1;
            VkCommandBuffer command = VK_NULL_HANDLE;
            VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkfn::BeginCommandBuffer(command, &begin));
            VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                     1, &upload, 0, nullptr, 0, nullptr);
            record(command);
            VkMemoryBarrier completion{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            completion.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            completion.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT, 0,
                                     1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
            VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        };

        std::vector<float> host_logits(index.header.vocabulary);
        std::vector<std::vector<uint64_t>> expert_histogram(
            index.header.layers, std::vector<uint64_t>(index.extension.experts));
        uint64_t selected_expert_executions = 0;
        uint64_t expert_matrix_loads = 0;
        const RmsPush rms_push{1, dimension, index.header.rms_epsilon, 0};
        const AddPush add_push{dimension, 0, 0, 0};
        const SwiGluPush swiglu_push{1, hidden, 0, 0};
        const RouterPush router_push{dimension, index.extension.experts,
                                     index.extension.top_k, 0};

        const auto run_token = [&](uint32_t token, uint32_t position) {
            weights.load(index.token_embedding, token, 1);
            execute([&](VkCommandBuffer command) {
                const EmbeddingPush push{0, dimension, index.token_embedding.packed_stride, 0};
                dispatch(command, resources, embedding_pipeline, embedding_set, &push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command);
                dispatch(command, resources, rmsnorm_pipeline, sets[0].norm1, &rms_push, 1, 1);
            });

            for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
                const MoELayerIndex& layer = index.layers[layer_number];
                const MoELayerSets& set = sets[layer_number];
                weights.load(layer.qkv);
                execute([&](VkCommandBuffer command) {
                    const LinearPush linear{1, qkv_dimension, dimension, layer.qkv.packed_stride};
                    const CachePush cache{position, dimension, head_dimension, kv_dimension};
                    dispatch(command, resources, qgemm_pipeline, set.qkv, &linear,
                             (qkv_dimension + 15) / 16, 1);
                    compute_barrier(command);
                    dispatch(command, resources, rope_pipeline, set.rope, &cache,
                             (dimension + 63) / 64, 1);
                    compute_barrier(command);
                    dispatch(command, resources, attention_pipeline, set.attention, &cache,
                             index.header.heads, 1);
                });

                weights.load(layer.output);
                execute([&](VkCommandBuffer command) {
                    const LinearPush linear{1, dimension, dimension, layer.output.packed_stride};
                    dispatch(command, resources, qgemm_pipeline, set.output, &linear,
                             (dimension + 15) / 16, 1);
                    compute_barrier(command);
                    dispatch(command, resources, add_pipeline, set.residual1, &add_push,
                             (dimension + 63) / 64, 1);
                    compute_barrier(command);
                    dispatch(command, resources, rmsnorm_pipeline, set.norm2, &rms_push, 1, 1);
                    compute_barrier(command);
                    dispatch(command, resources, router_pipeline, set.router, &router_push, 1, 1);
                });
                invalidate_buffer(runtime, routing);
                const float* route = static_cast<const float*>(routing.mapped);
                std::array<uint32_t, 2> selected{
                    static_cast<uint32_t>(std::lround(route[0])),
                    static_cast<uint32_t>(std::lround(route[1]))};
                std::array<float, 2> routing_weights{route[2], route[3]};
                if (selected[0] >= index.extension.experts ||
                    selected[1] >= index.extension.experts || selected[0] == selected[1] ||
                    !std::isfinite(routing_weights[0]) || !std::isfinite(routing_weights[1])) {
                    throw std::runtime_error("Vulkan MoE router produced an invalid top-2 result");
                }

                for (uint32_t rank = 0; rank < index.extension.top_k; ++rank) {
                    const uint32_t expert_number = selected[rank];
                    const ExpertIndex& expert = layer.experts[expert_number];
                    ++expert_histogram[layer_number][expert_number];
                    ++selected_expert_executions;
                    weights.load(expert.gate_up);
                    ++expert_matrix_loads;
                    execute([&](VkCommandBuffer command) {
                        const LinearPush linear{1, 2 * hidden, dimension,
                                                expert.gate_up.packed_stride};
                        dispatch(command, resources, qgemm_pipeline, set.gate_up, &linear,
                                 (2 * hidden + 15) / 16, 1);
                        compute_barrier(command);
                        dispatch(command, resources, swiglu_pipeline, set.swiglu, &swiglu_push,
                                 (hidden + 63) / 64, 1);
                    });
                    weights.load(expert.down);
                    ++expert_matrix_loads;
                    execute([&](VkCommandBuffer command) {
                        const LinearPush linear{1, dimension, hidden, expert.down.packed_stride};
                        const WeightedPush weighted_push{dimension, routing_weights[rank],
                                                         rank == 0 ? 1u : 0u, 0};
                        dispatch(command, resources, qgemm_pipeline, set.down, &linear,
                                 (dimension + 15) / 16, 1);
                        compute_barrier(command);
                        dispatch(command, resources, weighted_pipeline, set.weighted,
                                 &weighted_push, (dimension + 63) / 64, 1);
                        if (rank + 1 == index.extension.top_k) {
                            compute_barrier(command);
                            dispatch(command, resources, add_pipeline, set.residual2, &add_push,
                                     (dimension + 63) / 64, 1);
                            compute_barrier(command);
                            if (layer_number + 1 < index.header.layers) {
                                dispatch(command, resources, rmsnorm_pipeline,
                                         sets[layer_number + 1].norm1, &rms_push, 1, 1);
                            } else {
                                dispatch(command, resources, rmsnorm_pipeline, final_norm_set,
                                         &rms_push, 1, 1);
                            }
                        }
                    });
                }
            }

            for (uint32_t first = 0; first < index.header.vocabulary;
                 first += M6_LOGIT_TILE_ROWS) {
                const uint32_t rows = std::min(M6_LOGIT_TILE_ROWS,
                                               index.header.vocabulary - first);
                weights.load(index.lm_head, first, rows);
                execute([&](VkCommandBuffer command) {
                    const LinearPush linear{1, rows, dimension, index.lm_head.packed_stride};
                    dispatch(command, resources, qgemm_pipeline, logits_set, &linear,
                             (rows + 15) / 16, 1);
                });
                invalidate_buffer(runtime, logit_output);
                std::memcpy(host_logits.data() + first, logit_output.mapped,
                            static_cast<size_t>(rows) * sizeof(float));
            }
        };

        // Exact checkpoint template: BOS, user marker, newline, content, EOS, newline,
        // assistant marker, newline. Content follows a newline, so Metaspace does not
        // prepend its usual word-boundary marker to the first word.
        const uint32_t newline = tokenizer.token_id("<0x0A>");
        const std::vector<uint32_t> content = tokenizer.encode(
            prompt, chat_template ? false : true, chat_template ? false : true);
        std::vector<uint32_t> tokens;
        if (chat_template) {
            tokens = {index.header.bos, index.header.vocabulary - 2, newline};
            tokens.insert(tokens.end(), content.begin(), content.end());
            tokens.push_back(tokenizer.eos());
            tokens.push_back(newline);
            tokens.push_back(index.header.vocabulary - 1);
            tokens.push_back(newline);
        } else {
            tokens = content;
        }
        if (content.empty()) throw std::runtime_error("Prompt tokenization produced no tokens");
        if (tokens.size() + requested_tokens > M6_CACHE_SEQUENCE) {
            throw std::runtime_error("Prompt and generation exceed the 256-token KV cache");
        }
        const size_t prompt_tokens = tokens.size();
        for (uint32_t position = 0; position < tokens.size(); ++position) {
            run_token(tokens[position], position);
        }

        const auto start = std::chrono::steady_clock::now();
        std::mt19937 random(0x6d6f6536u);
        uint32_t generated = 0;
        for (; generated < requested_tokens; ++generated) {
            const uint32_t next = stochastic ? sample_token(host_logits, random)
                                             : greedy_token(host_logits);
            tokens.push_back(next);
            run_token(next, static_cast<uint32_t>(tokens.size() - 1));
            if (next == tokenizer.eos()) {
                ++generated;
                break;
            }
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const double tokens_per_second = generated / seconds;
        const uint64_t shared_parameters = index.total_parameters - index.expert_parameters;
        const uint64_t active_parameters = shared_parameters +
            index.expert_parameters * index.extension.top_k / index.extension.experts;
        const uint64_t expected_executions = static_cast<uint64_t>(tokens.size()) *
            index.header.layers * index.extension.top_k;
        if (selected_expert_executions != expected_executions ||
            expert_matrix_loads != 2 * expected_executions) {
            throw std::runtime_error("Conditional expert execution accounting failed");
        }
        uint32_t distinct_experts = 0;
        for (const auto& layer : expert_histogram) {
            for (uint64_t count : layer) distinct_experts += count != 0 ? 1u : 0u;
        }

        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Model: FlameF0X/TinyMoE-100m-2x8-chat-stage1\n"
                  << "Total parameters: " << index.total_parameters
                  << ", active parameters/token: " << active_parameters << "\n"
                  << "Routing: Vulkan top-" << index.extension.top_k << " of "
                  << index.extension.experts << " experts per layer\n"
                  << "Learned-weight budget: " << (M6_WEIGHT_BUDGET / 1024)
                  << " KiB, peak resident: " << std::fixed << std::setprecision(1)
                  << (learned_weight_residency / 1024.0) << " KiB\n"
                  << "Expert conditional executions: " << selected_expert_executions
                  << ", selected expert matrix loads: " << expert_matrix_loads
                  << ", inactive expert matrix loads: 0, layer/expert pairs observed: "
                  << distinct_experts << "\n"
                  << "Weight-cache replacements: " << weights.replacements()
                  << ", maximum single load: " << (weights.maximum_loaded_bytes() / 1024.0)
                  << " KiB\n"
                  << "Prompt tokens: " << prompt_tokens << ", generated tokens: " << generated
                  << ", generation: " << std::setprecision(3) << tokens_per_second << " tok/s\n"
                  << "Generated token IDs:";
        for (size_t token = prompt_tokens; token < tokens.size(); ++token) {
            std::cout << " " << tokens[token];
        }
        std::cout << "\n--- generated text ---\n" << tokenizer.decode(tokens)
                  << "\n--- end ---\n"
                  << "Peak total Vulkan buffer allocations: "
                  << (peak_vulkan_buffer_bytes / (1024.0 * 1024.0)) << " MiB\n"
                  << "RESULT: PASS - real MoE generated with selected-expert-only Vulkan streaming\n";

        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines) {
            vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        }
        for (VkShaderModule module : resources.shader_modules) {
            vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        }
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        destroy_buffer(runtime, logit_output);
        destroy_buffer(runtime, routing);
        for (Buffer* buffer : auxiliary_buffers) destroy_buffer(runtime, *buffer);
        weights.destroy();
        for (ResidentMoELayer& layer : model.layers) {
            destroy_buffer(runtime, layer.attention_state);
            destroy_buffer(runtime, layer.router);
            destroy_buffer(runtime, layer.feed_forward_norm);
            destroy_buffer(runtime, layer.attention_norm);
        }
        destroy_buffer(runtime, model.final_norm);
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
