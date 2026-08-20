#define OVLLM_STREAMING_RUNTIME_ONLY
#include "llm.cpp"

struct MatrixIndex {
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t packed_stride = 0;
    uint64_t scale_offset = 0;
    uint64_t weight_offset = 0;

    uint64_t byte_size() const {
        return static_cast<uint64_t>(rows) * sizeof(float) +
               static_cast<uint64_t>(rows) * packed_stride * sizeof(uint32_t);
    }
};

struct StreamLayerIndex {
    uint64_t attention_norm_offset = 0;
    uint64_t feed_forward_norm_offset = 0;
    MatrixIndex qkv;
    MatrixIndex output;
    MatrixIndex gate_up;
    MatrixIndex down;
};

struct ModelIndex {
    ModelHeader header{};
    MatrixIndex token_embedding;
    MatrixIndex lm_head;
    uint64_t final_norm_offset = 0;
    std::vector<StreamLayerIndex> layers;
    uint64_t matrix_bytes = 0;
    uint64_t norm_bytes = 0;
    uint64_t file_bytes = 0;
};

static MatrixIndex index_matrix(uint64_t& cursor, uint32_t rows, uint32_t columns,
                                uint64_t& total_matrix_bytes) {
    MatrixIndex matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.packed_stride = (columns + 3) / 4;
    matrix.scale_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * sizeof(float);
    matrix.weight_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * matrix.packed_stride * sizeof(uint32_t);
    total_matrix_bytes += matrix.byte_size();
    return matrix;
}

static ModelIndex index_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open converted model: " + path);
    ModelIndex index;
    index.file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    if (!input || std::memcmp(index.header.magic, "OVLLM3\0\0", 8) != 0 ||
        (index.header.version != 1 && index.header.version != 2)) {
        throw std::runtime_error("Unsupported converted model format");
    }
    uint64_t cursor = sizeof(ModelHeader);
    const uint32_t dimension = index.header.dimension;
    const uint32_t hidden = index.header.hidden_dimension;
    index.token_embedding = index_matrix(cursor, index.header.vocabulary, dimension,
                                         index.matrix_bytes);
    if (index.header.version >= 2) {
        index.lm_head = index_matrix(cursor, index.header.vocabulary, dimension,
                                     index.matrix_bytes);
    } else {
        index.lm_head = index.token_embedding;
    }
    index.final_norm_offset = cursor;
    cursor += static_cast<uint64_t>(dimension) * sizeof(float);
    index.norm_bytes += static_cast<uint64_t>(dimension) * sizeof(float);
    index.layers.resize(index.header.layers);
    for (StreamLayerIndex& layer : index.layers) {
        layer.attention_norm_offset = cursor;
        cursor += static_cast<uint64_t>(dimension) * sizeof(float);
        layer.feed_forward_norm_offset = cursor;
        cursor += static_cast<uint64_t>(dimension) * sizeof(float);
        index.norm_bytes += static_cast<uint64_t>(2) * dimension * sizeof(float);
        const uint32_t head_dimension = dimension / index.header.heads;
        const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
        layer.qkv = index_matrix(cursor, dimension + 2 * kv_dimension, dimension,
                                 index.matrix_bytes);
        layer.output = index_matrix(cursor, dimension, dimension, index.matrix_bytes);
        layer.gate_up = index_matrix(cursor, 2 * hidden, dimension, index.matrix_bytes);
        layer.down = index_matrix(cursor, dimension, hidden, index.matrix_bytes);
    }
    if (cursor != index.file_bytes) {
        throw std::runtime_error("Converted model size does not match its indexed tensor layout");
    }
    return index;
}

static void read_file_at(std::ifstream& input, uint64_t offset, void* destination, size_t bytes,
                         const char* description) {
    input.clear();
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (!input) throw std::runtime_error(std::string("Model read failed for ") + description);
}

static Buffer load_norm_at(std::ifstream& input, const Runtime& runtime, uint64_t offset,
                           uint32_t dimension) {
    std::vector<float> values(dimension);
    read_file_at(input, offset, values.data(), values.size() * sizeof(float), "RMSNorm weights");
    return upload_vector(runtime, values);
}

struct StreamLayer {
    Buffer attention_norm;
    Buffer feed_forward_norm;
    Buffer attention_state;
};

struct ResidentModel {
    ModelIndex index;
    Buffer final_norm;
    std::vector<StreamLayer> layers;
};

static ResidentModel load_resident_state(const std::string& path, const Runtime& runtime,
                                         const ModelIndex& index) {
    std::ifstream input(path, std::ios::binary);
    ResidentModel model;
    model.index = index;
    const uint32_t dimension = index.header.dimension;
    const uint32_t head_dimension = dimension / index.header.heads;
    const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
    model.final_norm = load_norm_at(input, runtime, index.final_norm_offset, dimension);
    model.layers.resize(index.header.layers);
    for (uint32_t layer_index = 0; layer_index < index.header.layers; ++layer_index) {
        StreamLayer& layer = model.layers[layer_index];
        const StreamLayerIndex& layer_index_data = index.layers[layer_index];
        layer.attention_norm = load_norm_at(input, runtime,
            layer_index_data.attention_norm_offset, dimension);
        layer.feed_forward_norm = load_norm_at(input, runtime,
            layer_index_data.feed_forward_norm_offset, dimension);
        const VkDeviceSize state_size = static_cast<VkDeviceSize>(
            dimension + 2ull * std::min<uint32_t>(index.header.max_sequence, 256) * kv_dimension) *
            sizeof(float);
        layer.attention_state = create_buffer(runtime, state_size);
    }
    return model;
}

class WeightStreamer {
public:
    WeightStreamer(const std::string& path, const Runtime& runtime,
                   uint32_t maximum_scale_rows, uint32_t maximum_packed_values)
        : input_(path, std::ios::binary), runtime_(runtime) {
        if (!input_) throw std::runtime_error("Could not open model for weight streaming");
        scales_ = create_buffer(runtime_, static_cast<VkDeviceSize>(maximum_scale_rows) * sizeof(float));
        weights_ = create_buffer(runtime_, static_cast<VkDeviceSize>(maximum_packed_values) * sizeof(uint32_t));
    }

    void load(const MatrixIndex& matrix, uint32_t first_row = 0, uint32_t row_count = 0) {
        if (row_count == 0) row_count = matrix.rows;
        if (first_row + row_count > matrix.rows) throw std::runtime_error("Invalid streamed matrix row range");
        const size_t scale_bytes = static_cast<size_t>(row_count) * sizeof(float);
        const size_t weight_bytes = static_cast<size_t>(row_count) * matrix.packed_stride * sizeof(uint32_t);
        if (scale_bytes > scales_.size || weight_bytes > weights_.size) {
            throw std::runtime_error("A streamed matrix exceeds the imposed Vulkan weight cache");
        }
        read_file_at(input_, matrix.scale_offset + static_cast<uint64_t>(first_row) * sizeof(float),
                     scales_.mapped, scale_bytes, "matrix scales");
        read_file_at(input_, matrix.weight_offset +
                     static_cast<uint64_t>(first_row) * matrix.packed_stride * sizeof(uint32_t),
                     weights_.mapped, weight_bytes, "packed matrix weights");
        flush_buffer(runtime_, scales_);
        flush_buffer(runtime_, weights_);
        ++loads_;
        if (loads_ > 1) ++replacements_;
        bytes_streamed_ += scale_bytes + weight_bytes;
        maximum_loaded_bytes_ = std::max<uint64_t>(maximum_loaded_bytes_, scale_bytes + weight_bytes);
    }

    Buffer& scales() { return scales_; }
    Buffer& weights() { return weights_; }
    uint64_t capacity_bytes() const { return scales_.size + weights_.size; }
    uint64_t loads() const { return loads_; }
    uint64_t replacements() const { return replacements_; }
    uint64_t bytes_streamed() const { return bytes_streamed_; }
    uint64_t maximum_loaded_bytes() const { return maximum_loaded_bytes_; }

    void destroy() {
        destroy_buffer(runtime_, weights_);
        destroy_buffer(runtime_, scales_);
    }

private:
    std::ifstream input_;
    const Runtime& runtime_;
    Buffer scales_;
    Buffer weights_;
    uint64_t loads_ = 0;
    uint64_t replacements_ = 0;
    uint64_t bytes_streamed_ = 0;
    uint64_t maximum_loaded_bytes_ = 0;
};

#ifndef OVLLM_M5_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        constexpr uint64_t WEIGHT_BUDGET = 512ull * 1024;
        constexpr uint32_t LOGIT_TILE_ROWS = 1024;
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\tinystories-llama-15m\\runtime";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "Once upon a time, there was a little girl named Lily.";
        const uint32_t requested_tokens = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 48;
        const bool greedy = argc > 5 && std::string(argv[5]) == "greedy";
        const std::string logits_dump_path = argc > 6 ? argv[6] : "";
        const std::string model_path = model_directory + "\\model.ovm";

        Runtime runtime = create_runtime();
        const ModelIndex index = index_model(model_path);
        ResidentModel model = load_resident_state(model_path, runtime, index);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        const uint32_t dimension = index.header.dimension;
        const uint32_t hidden = index.header.hidden_dimension;
        const uint32_t head_dimension = dimension / index.header.heads;

        uint32_t maximum_scale_rows = LOGIT_TILE_ROWS;
        uint32_t maximum_packed_values = LOGIT_TILE_ROWS * index.token_embedding.packed_stride;
        const auto include_matrix = [&](const MatrixIndex& matrix) {
            maximum_scale_rows = std::max(maximum_scale_rows, matrix.rows);
            maximum_packed_values = std::max(maximum_packed_values,
                                              matrix.rows * matrix.packed_stride);
        };
        for (const StreamLayerIndex& layer : index.layers) {
            include_matrix(layer.qkv);
            include_matrix(layer.output);
            include_matrix(layer.gate_up);
            include_matrix(layer.down);
        }
        WeightStreamer streamer(model_path, runtime, maximum_scale_rows, maximum_packed_values);
        const uint64_t resident_weight_capacity = streamer.capacity_bytes() + index.norm_bytes;
        if (resident_weight_capacity > WEIGHT_BUDGET) {
            throw std::runtime_error("Configured streaming cache exceeds the 512 KiB weight budget");
        }

        std::vector<float> rope_cos(static_cast<size_t>(index.header.max_sequence) * head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t position = 0; position < index.header.max_sequence; ++position) {
            for (uint32_t frequency = 0; frequency < head_dimension / 2; ++frequency) {
                const float inverse_frequency = std::pow(index.header.rope_theta,
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
        Buffer logit_tile = create_buffer(runtime, LOGIT_TILE_ROWS * sizeof(float));
        std::array<Buffer*, 12> auxiliary_buffers = {
            &rope_cos_buffer, &rope_sin_buffer, &dummy, &hidden_a, &hidden_b, &norm,
            &qkv, &context, &projection, &gate_up, &feed_forward, &down};

        const uint32_t descriptor_count = 3 + 11 * index.header.layers;
        ComputeResources resources = create_compute_resources(runtime, descriptor_count);
        const auto shader = [&](const char* name) { return shader_directory + "\\" + name + ".comp.spv"; };
        const VkPipeline embedding_pipeline = create_pipeline(runtime, resources, shader("embedding"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline qgemm_pipeline = create_pipeline(runtime, resources, shader("qgemm"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope_cache"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention_cache"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));

        const VkDescriptorSet embedding_set = create_descriptor_set(runtime, resources,
            {&streamer.weights(), &streamer.scales(), &hidden_a, &dummy});
        std::vector<LayerSets> layer_sets(index.header.layers);
        for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
            StreamLayer& layer = model.layers[layer_number];
            LayerSets& sets = layer_sets[layer_number];
            sets.norm1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &layer.attention_norm, &norm, &dummy});
            sets.qkv = create_descriptor_set(runtime, resources,
                {&norm, &streamer.weights(), &streamer.scales(), &qkv});
            sets.rope = create_descriptor_set(runtime, resources,
                {&qkv, &rope_cos_buffer, &rope_sin_buffer, &layer.attention_state});
            sets.attention = create_descriptor_set(runtime, resources,
                {&layer.attention_state, &context, &dummy, &dummy});
            sets.output = create_descriptor_set(runtime, resources,
                {&context, &streamer.weights(), &streamer.scales(), &projection});
            sets.residual1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &projection, &hidden_b, &dummy});
            sets.norm2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &layer.feed_forward_norm, &norm, &dummy});
            sets.gate_up = create_descriptor_set(runtime, resources,
                {&norm, &streamer.weights(), &streamer.scales(), &gate_up});
            sets.swiglu = create_descriptor_set(runtime, resources,
                {&gate_up, &feed_forward, &dummy, &dummy});
            sets.down = create_descriptor_set(runtime, resources,
                {&feed_forward, &streamer.weights(), &streamer.scales(), &down});
            sets.residual2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &down, &hidden_a, &dummy});
        }
        const VkDescriptorSet final_norm_set = create_descriptor_set(runtime, resources,
            {&hidden_a, &model.final_norm, &norm, &dummy});
        const VkDescriptorSet logits_set = create_descriptor_set(runtime, resources,
            {&norm, &streamer.weights(), &streamer.scales(), &logit_tile});

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = runtime.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &command_pool_info, nullptr, &command_pool));
        auto execute = [&](VkPipeline pipeline, VkDescriptorSet set, const void* push,
                           uint32_t groups_x, uint32_t groups_y) {
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
            dispatch(command_buffer, resources, pipeline, set, push, groups_x, groups_y);
            VkMemoryBarrier completion{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            completion.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            completion.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
                                     1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command_buffer));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command_buffer;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
            VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        };

        std::vector<float> host_logits(index.header.vocabulary);
        auto run_token = [&](uint32_t token, uint32_t position) {
            streamer.load(index.token_embedding, token, 1);
            const EmbeddingPush embedding_push{0, dimension, index.token_embedding.packed_stride, 0};
            execute(embedding_pipeline, embedding_set, &embedding_push, (dimension + 63) / 64, 1);
            const RmsPush rms_push{1, dimension, index.header.rms_epsilon, 0};
            const CachePush cache_push{position, dimension, head_dimension, dimension};
            const AddPush add_push{dimension, 0, 0, 0};
            const SwiGluPush swiglu_push{1, hidden, 0, 0};
            for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
                const StreamLayerIndex& layer_index_data = index.layers[layer_number];
                const LayerSets& sets = layer_sets[layer_number];
                execute(rmsnorm_pipeline, sets.norm1, &rms_push, 1, 1);
                streamer.load(layer_index_data.qkv);
                const LinearPush qkv_push{1, 3 * dimension, dimension,
                                          layer_index_data.qkv.packed_stride};
                execute(qgemm_pipeline, sets.qkv, &qkv_push, (3 * dimension + 15) / 16, 1);
                execute(rope_pipeline, sets.rope, &cache_push, (dimension + 63) / 64, 1);
                execute(attention_pipeline, sets.attention, &cache_push, index.header.heads, 1);
                streamer.load(layer_index_data.output);
                const LinearPush output_push{1, dimension, dimension,
                                             layer_index_data.output.packed_stride};
                execute(qgemm_pipeline, sets.output, &output_push, (dimension + 15) / 16, 1);
                execute(add_pipeline, sets.residual1, &add_push, (dimension + 63) / 64, 1);
                execute(rmsnorm_pipeline, sets.norm2, &rms_push, 1, 1);
                streamer.load(layer_index_data.gate_up);
                const LinearPush gate_up_push{1, 2 * hidden, dimension,
                                              layer_index_data.gate_up.packed_stride};
                execute(qgemm_pipeline, sets.gate_up, &gate_up_push, (2 * hidden + 15) / 16, 1);
                execute(swiglu_pipeline, sets.swiglu, &swiglu_push, (hidden + 63) / 64, 1);
                streamer.load(layer_index_data.down);
                const LinearPush down_push{1, dimension, hidden, layer_index_data.down.packed_stride};
                execute(qgemm_pipeline, sets.down, &down_push, (dimension + 15) / 16, 1);
                execute(add_pipeline, sets.residual2, &add_push, (dimension + 63) / 64, 1);
            }
            execute(rmsnorm_pipeline, final_norm_set, &rms_push, 1, 1);
            for (uint32_t first_row = 0; first_row < index.header.vocabulary;
                 first_row += LOGIT_TILE_ROWS) {
                const uint32_t rows = std::min(LOGIT_TILE_ROWS, index.header.vocabulary - first_row);
                streamer.load(index.token_embedding, first_row, rows);
                const LinearPush logits_push{1, rows, dimension, index.token_embedding.packed_stride};
                execute(qgemm_pipeline, logits_set, &logits_push, (rows + 15) / 16, 1);
                invalidate_buffer(runtime, logit_tile);
                std::memcpy(host_logits.data() + first_row, logit_tile.mapped,
                            static_cast<size_t>(rows) * sizeof(float));
            }
        };

        std::vector<uint32_t> tokens = tokenizer.encode(prompt, true);
        if (tokens.empty() || tokens.size() + requested_tokens > index.header.max_sequence) {
            throw std::runtime_error("Prompt/generation does not fit the model context");
        }
        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Model: sdobson/tinystories-llama-15m\n"
                  << "Imposed learned-weight residency budget: " << (WEIGHT_BUDGET / 1024) << " KiB\n"
                  << "Allocated streamed-weight cache + resident norms: "
                  << (resident_weight_capacity / 1024.0) << " KiB\n"
                  << "Full quantized matrix data: " << std::fixed << std::setprecision(2)
                  << (index.matrix_bytes / (1024.0 * 1024.0)) << " MiB\n"
                  << "Logit matrix tile: " << LOGIT_TILE_ROWS << " / " << index.header.vocabulary
                  << " rows\n";
        for (uint32_t position = 0; position < tokens.size(); ++position) {
            run_token(tokens[position], position);
        }
        dump_logits(logits_dump_path, host_logits);
        const size_t prompt_token_count = tokens.size();
        std::mt19937 random(0x6700u);
        uint32_t generated = 0;
        while (generated < requested_tokens && tokens.size() < index.header.max_sequence) {
            const uint32_t next = greedy ? greedy_token(host_logits) : sample_token(host_logits, random);
            if (next == tokenizer.eos()) break;
            tokens.push_back(next);
            ++generated;
            if (generated < requested_tokens) run_token(next, static_cast<uint32_t>(tokens.size() - 1));
        }
        std::cout << "Weight cache loads: " << streamer.loads() << "\n"
                  << "Weight cache replacements: " << streamer.replacements() << "\n"
                  << "Weight bytes read from model file: " << std::fixed << std::setprecision(2)
                  << (streamer.bytes_streamed() / (1024.0 * 1024.0)) << " MiB\n"
                  << "Largest single resident streamed payload: "
                  << (streamer.maximum_loaded_bytes() / 1024.0) << " KiB\n"
                  << "Generated tokens: " << generated << "\n"
                  << "Generated token IDs:";
        for (size_t token_index = prompt_token_count; token_index < tokens.size(); ++token_index) {
            std::cout << " " << tokens[token_index];
        }
        std::cout << "\nPeak Vulkan buffer allocations: " << std::fixed << std::setprecision(3)
                  << (peak_vulkan_buffer_bytes / (1024.0 * 1024.0)) << " MiB\n"
                  << "--- generated text ---\n" << tokenizer.decode(tokens) << "\n--- end ---\n"
                  << "RESULT: PASS - coherent generation with bounded Vulkan weight streaming\n";

        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines) vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        for (VkShaderModule module : resources.shader_modules) vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        destroy_buffer(runtime, logit_tile);
        for (Buffer* buffer : auxiliary_buffers) destroy_buffer(runtime, *buffer);
        streamer.destroy();
        for (StreamLayer& layer : model.layers) {
            destroy_buffer(runtime, layer.attention_state);
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
