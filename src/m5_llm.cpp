#define OVLLM_M5_RUNTIME_ONLY
#include "streaming_llm.cpp"

#include <chrono>
#include <future>

constexpr uint32_t M5_CACHE_SEQUENCE = 256;
constexpr uint32_t M5_LOGIT_TILE_ROWS = 1024;
constexpr uint64_t M5_WEIGHT_BUDGET = 48ull * 1024 * 1024;

class WeightSlot {
public:
    WeightSlot(const std::string& path, const Runtime& runtime,
               uint32_t maximum_scale_rows, uint32_t maximum_packed_values)
        : input_(path, std::ios::binary), runtime_(runtime) {
        if (!input_) throw std::runtime_error("Could not open model stream for cache slot");
        scales = create_buffer(runtime, static_cast<VkDeviceSize>(maximum_scale_rows) * sizeof(float));
        weights = create_buffer(runtime, static_cast<VkDeviceSize>(maximum_packed_values) * sizeof(uint32_t));
        logit_output = create_buffer(runtime, M5_LOGIT_TILE_ROWS * sizeof(float));
    }

    void load(const MatrixIndex& matrix, uint32_t first_row, uint32_t row_count) {
        const size_t scale_bytes = static_cast<size_t>(row_count) * sizeof(float);
        const size_t weight_bytes = static_cast<size_t>(row_count) * matrix.packed_stride * sizeof(uint32_t);
        if (first_row + row_count > matrix.rows || scale_bytes > scales.size ||
            weight_bytes > weights.size) {
            throw std::runtime_error("Stream request exceeds a weight cache slot");
        }
        read_file_at(input_, matrix.scale_offset + static_cast<uint64_t>(first_row) * sizeof(float),
                     scales.mapped, scale_bytes, "prefetched matrix scales");
        read_file_at(input_, matrix.weight_offset +
                     static_cast<uint64_t>(first_row) * matrix.packed_stride * sizeof(uint32_t),
                     weights.mapped, weight_bytes, "prefetched packed weights");
        flush_buffer(runtime_, scales);
        flush_buffer(runtime_, weights);
        ++loads;
        bytes_read += scale_bytes + weight_bytes;
    }

    uint64_t capacity() const { return scales.size + weights.size; }

    void destroy() {
        destroy_buffer(runtime_, logit_output);
        destroy_buffer(runtime_, weights);
        destroy_buffer(runtime_, scales);
    }

    Buffer scales;
    Buffer weights;
    Buffer logit_output;
    uint64_t loads = 0;
    uint64_t bytes_read = 0;

private:
    std::ifstream input_;
    const Runtime& runtime_;
};

struct M5LayerSets {
    VkDescriptorSet norm1 = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> qkv{};
    VkDescriptorSet rope = VK_NULL_HANDLE;
    VkDescriptorSet attention = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> output{};
    VkDescriptorSet residual1 = VK_NULL_HANDLE;
    VkDescriptorSet norm2 = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> gate_up{};
    VkDescriptorSet swiglu = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> down{};
    VkDescriptorSet residual2 = VK_NULL_HANDLE;
};

enum class StageKind { Embedding, Qkv, Output, GateUp, Down, Logits };

struct Stage {
    StageKind kind;
    const MatrixIndex* matrix;
    uint32_t first_row = 0;
    uint32_t row_count = 0;
    uint32_t layer = 0;
    uint32_t token = 0;
};

struct ModeResult {
    std::vector<uint32_t> tokens;
    double seconds = 0.0;
    double tokens_per_second = 0.0;
};

int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\tinyllama-1.1b-chat\\runtime";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "Once upon a time, there was a little girl named Lily.";
        const uint32_t generation_tokens = argc > 4 ? static_cast<uint32_t>(std::stoul(argv[4])) : 8;
        const std::string model_path = model_directory + "\\model.ovm";

        Runtime runtime = create_runtime();
        const ModelIndex index = index_model(model_path);
        if (index.header.version != 2) throw std::runtime_error("Milestone 5 expects model format v2");
        if (index.header.max_sequence < M5_CACHE_SEQUENCE) {
            throw std::runtime_error("Model context is smaller than the configured KV cache");
        }
        ResidentModel model = load_resident_state(model_path, runtime, index);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        const uint32_t dimension = index.header.dimension;
        const uint32_t hidden = index.header.hidden_dimension;
        const uint32_t head_dimension = dimension / index.header.heads;
        const uint32_t kv_dimension = head_dimension * index.header.kv_heads;
        const uint32_t qkv_dimension = dimension + 2 * kv_dimension;

        uint32_t maximum_scale_rows = M5_LOGIT_TILE_ROWS;
        uint32_t maximum_packed_values = M5_LOGIT_TILE_ROWS * index.lm_head.packed_stride;
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
        WeightSlot cache0(model_path, runtime, maximum_scale_rows, maximum_packed_values);
        WeightSlot cache1(model_path, runtime, maximum_scale_rows, maximum_packed_values);
        std::array<WeightSlot*, 2> caches{&cache0, &cache1};
        const uint64_t learned_weight_capacity = cache0.capacity() + cache1.capacity() + index.norm_bytes;
        if (learned_weight_capacity > M5_WEIGHT_BUDGET) {
            throw std::runtime_error("Double-buffered cache exceeds the 48 MiB weight budget");
        }

        std::vector<float> rope_cos(static_cast<size_t>(M5_CACHE_SEQUENCE) * head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t position = 0; position < M5_CACHE_SEQUENCE; ++position) {
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
        Buffer qkv = create_buffer(runtime, qkv_dimension * sizeof(float));
        Buffer context = create_buffer(runtime, dimension * sizeof(float));
        Buffer projection = create_buffer(runtime, dimension * sizeof(float));
        Buffer gate_up = create_buffer(runtime, 2 * hidden * sizeof(float));
        Buffer feed_forward = create_buffer(runtime, hidden * sizeof(float));
        Buffer down = create_buffer(runtime, dimension * sizeof(float));
        std::array<Buffer*, 11> auxiliary_buffers = {
            &rope_cos_buffer, &rope_sin_buffer, &dummy, &hidden_a, &hidden_b, &norm,
            &qkv, &context, &projection, &gate_up, &feed_forward};

        const uint32_t descriptor_count = 5 + 15 * index.header.layers;
        ComputeResources resources = create_compute_resources(runtime, descriptor_count);
        const auto shader = [&](const char* name) { return shader_directory + "\\" + name + ".comp.spv"; };
        const VkPipeline embedding_pipeline = create_pipeline(runtime, resources, shader("embedding"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline qgemm_pipeline = create_pipeline(runtime, resources, shader("qgemm"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope_cache"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention_cache"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));

        std::array<VkDescriptorSet, 2> embedding_sets{};
        std::array<VkDescriptorSet, 2> logits_sets{};
        for (uint32_t slot = 0; slot < 2; ++slot) {
            embedding_sets[slot] = create_descriptor_set(runtime, resources,
                {&caches[slot]->weights, &caches[slot]->scales, &hidden_a, &dummy});
            logits_sets[slot] = create_descriptor_set(runtime, resources,
                {&norm, &caches[slot]->weights, &caches[slot]->scales,
                 &caches[slot]->logit_output});
        }
        std::vector<M5LayerSets> layer_sets(index.header.layers);
        for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
            StreamLayer& layer = model.layers[layer_number];
            M5LayerSets& sets = layer_sets[layer_number];
            sets.norm1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &layer.attention_norm, &norm, &dummy});
            for (uint32_t slot = 0; slot < 2; ++slot) {
                sets.qkv[slot] = create_descriptor_set(runtime, resources,
                    {&norm, &caches[slot]->weights, &caches[slot]->scales, &qkv});
                sets.output[slot] = create_descriptor_set(runtime, resources,
                    {&context, &caches[slot]->weights, &caches[slot]->scales, &projection});
                sets.gate_up[slot] = create_descriptor_set(runtime, resources,
                    {&norm, &caches[slot]->weights, &caches[slot]->scales, &gate_up});
                sets.down[slot] = create_descriptor_set(runtime, resources,
                    {&feed_forward, &caches[slot]->weights, &caches[slot]->scales, &down});
            }
            sets.rope = create_descriptor_set(runtime, resources,
                {&qkv, &rope_cos_buffer, &rope_sin_buffer, &layer.attention_state});
            sets.attention = create_descriptor_set(runtime, resources,
                {&layer.attention_state, &context, &dummy, &dummy});
            sets.residual1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &projection, &hidden_b, &dummy});
            sets.norm2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &layer.feed_forward_norm, &norm, &dummy});
            sets.swiglu = create_descriptor_set(runtime, resources,
                {&gate_up, &feed_forward, &dummy, &dummy});
            sets.residual2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &down, &hidden_a, &dummy});
        }
        const VkDescriptorSet final_norm_set = create_descriptor_set(runtime, resources,
            {&hidden_a, &model.final_norm, &norm, &dummy});

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = runtime.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &command_pool_info, nullptr, &command_pool));
        std::array<VkFence, 2> fences{};
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (VkFence& fence : fences) {
            VK_CHECK(vkfn::CreateFence(runtime.device, &fence_info, nullptr, &fence));
        }

        auto make_stages = [&](uint32_t token) {
            std::vector<Stage> stages;
            stages.push_back({StageKind::Embedding, &index.token_embedding, token, 1, 0, token});
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                const StreamLayerIndex& layer_index = index.layers[layer];
                stages.push_back({StageKind::Qkv, &layer_index.qkv, 0, layer_index.qkv.rows, layer, 0});
                stages.push_back({StageKind::Output, &layer_index.output, 0, layer_index.output.rows, layer, 0});
                stages.push_back({StageKind::GateUp, &layer_index.gate_up, 0, layer_index.gate_up.rows, layer, 0});
                stages.push_back({StageKind::Down, &layer_index.down, 0, layer_index.down.rows, layer, 0});
            }
            for (uint32_t first = 0; first < index.header.vocabulary; first += M5_LOGIT_TILE_ROWS) {
                const uint32_t rows = std::min(M5_LOGIT_TILE_ROWS, index.header.vocabulary - first);
                stages.push_back({StageKind::Logits, &index.lm_head, first, rows, 0, 0});
            }
            return stages;
        };

        auto record_stage = [&](const Stage& stage, uint32_t slot, uint32_t position) {
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
            const RmsPush rms_push{1, dimension, index.header.rms_epsilon, 0};
            const CachePush cache_push{position, dimension, head_dimension, kv_dimension};
            const AddPush add_push{dimension, 0, 0, 0};
            const SwiGluPush swiglu_push{1, hidden, 0, 0};
            const M5LayerSets* sets = stage.kind == StageKind::Embedding ||
                                      stage.kind == StageKind::Logits ? nullptr :
                                      &layer_sets[stage.layer];
            if (stage.kind == StageKind::Embedding) {
                const EmbeddingPush push{0, dimension, stage.matrix->packed_stride, 0};
                dispatch(command_buffer, resources, embedding_pipeline, embedding_sets[slot], &push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, rmsnorm_pipeline, layer_sets[0].norm1,
                         &rms_push, 1, 1);
            } else if (stage.kind == StageKind::Qkv) {
                const LinearPush push{1, qkv_dimension, dimension, stage.matrix->packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets->qkv[slot], &push,
                         (qkv_dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, rope_pipeline, sets->rope, &cache_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, attention_pipeline, sets->attention, &cache_push,
                         index.header.heads, 1);
            } else if (stage.kind == StageKind::Output) {
                const LinearPush push{1, dimension, dimension, stage.matrix->packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets->output[slot], &push,
                         (dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, add_pipeline, sets->residual1, &add_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, rmsnorm_pipeline, sets->norm2, &rms_push, 1, 1);
            } else if (stage.kind == StageKind::GateUp) {
                const LinearPush push{1, 2 * hidden, dimension, stage.matrix->packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets->gate_up[slot], &push,
                         (2 * hidden + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, swiglu_pipeline, sets->swiglu, &swiglu_push,
                         (hidden + 63) / 64, 1);
            } else if (stage.kind == StageKind::Down) {
                const LinearPush push{1, dimension, hidden, stage.matrix->packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, sets->down[slot], &push,
                         (dimension + 15) / 16, 1);
                compute_barrier(command_buffer);
                dispatch(command_buffer, resources, add_pipeline, sets->residual2, &add_push,
                         (dimension + 63) / 64, 1);
                compute_barrier(command_buffer);
                if (stage.layer + 1 < index.header.layers) {
                    dispatch(command_buffer, resources, rmsnorm_pipeline,
                             layer_sets[stage.layer + 1].norm1, &rms_push, 1, 1);
                } else {
                    dispatch(command_buffer, resources, rmsnorm_pipeline, final_norm_set,
                             &rms_push, 1, 1);
                }
            } else {
                const LinearPush push{1, stage.row_count, dimension, stage.matrix->packed_stride};
                dispatch(command_buffer, resources, qgemm_pipeline, logits_sets[slot], &push,
                         (stage.row_count + 15) / 16, 1);
            }
            VkMemoryBarrier completion{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            completion.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            completion.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                     0, 1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command_buffer));
            return command_buffer;
        };

        std::vector<float> host_logits(index.header.vocabulary);
        auto run_token_sync = [&](uint32_t token, uint32_t position) {
            const std::vector<Stage> stages = make_stages(token);
            for (const Stage& stage : stages) {
                cache0.load(*stage.matrix, stage.first_row, stage.row_count);
                const VkCommandBuffer command = record_stage(stage, 0, position);
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &command;
                VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
                VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
                if (stage.kind == StageKind::Logits) {
                    invalidate_buffer(runtime, cache0.logit_output);
                    std::memcpy(host_logits.data() + stage.first_row, cache0.logit_output.mapped,
                                static_cast<size_t>(stage.row_count) * sizeof(float));
                }
            }
        };

        auto run_token_async = [&](uint32_t token, uint32_t position) {
            const std::vector<Stage> stages = make_stages(token);
            struct PendingLogit { bool valid = false; uint32_t first = 0; uint32_t count = 0; };
            std::array<PendingLogit, 2> pending{};
            const auto wait_slot = [&](uint32_t slot) {
                VK_CHECK(vkfn::WaitForFences(runtime.device, 1, &fences[slot], VK_TRUE, UINT64_MAX));
                if (pending[slot].valid) {
                    invalidate_buffer(runtime, caches[slot]->logit_output);
                    std::memcpy(host_logits.data() + pending[slot].first,
                                caches[slot]->logit_output.mapped,
                                static_cast<size_t>(pending[slot].count) * sizeof(float));
                    pending[slot].valid = false;
                }
            };

            wait_slot(0);
            caches[0]->load(*stages[0].matrix, stages[0].first_row, stages[0].row_count);
            std::future<void> prefetch;
            for (size_t stage_index = 0; stage_index < stages.size(); ++stage_index) {
                const uint32_t slot = static_cast<uint32_t>(stage_index & 1u);
                if (prefetch.valid()) prefetch.get();
                VK_CHECK(vkfn::ResetFences(runtime.device, 1, &fences[slot]));
                const VkCommandBuffer command = record_stage(stages[stage_index], slot, position);
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &command;
                VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, fences[slot]));
                if (stages[stage_index].kind == StageKind::Logits) {
                    pending[slot] = {true, stages[stage_index].first_row,
                                     stages[stage_index].row_count};
                }
                if (stage_index + 1 < stages.size()) {
                    const uint32_t next_slot = slot ^ 1u;
                    wait_slot(next_slot);
                    const Stage next_stage = stages[stage_index + 1];
                    prefetch = std::async(std::launch::async, [&, next_slot, next_stage] {
                        caches[next_slot]->load(*next_stage.matrix, next_stage.first_row,
                                                next_stage.row_count);
                    });
                }
            }
            if (prefetch.valid()) prefetch.get();
            wait_slot(0);
            wait_slot(1);
        };

        const std::vector<uint32_t> prompt_tokens = tokenizer.encode(prompt, true);
        if (prompt_tokens.empty() || prompt_tokens.size() + generation_tokens > M5_CACHE_SEQUENCE) {
            throw std::runtime_error("Prompt and generation exceed the 256-token milestone cache");
        }
        auto run_mode = [&](bool asynchronous) {
            std::vector<uint32_t> tokens = prompt_tokens;
            for (uint32_t position = 0; position < tokens.size(); ++position) {
                if (asynchronous) run_token_async(tokens[position], position);
                else run_token_sync(tokens[position], position);
            }
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t generated = 0; generated < generation_tokens; ++generated) {
                const uint32_t next = greedy_token(host_logits);
                tokens.push_back(next);
                const uint32_t position = static_cast<uint32_t>(tokens.size() - 1);
                if (asynchronous) run_token_async(next, position);
                else run_token_sync(next, position);
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            return ModeResult{tokens, seconds, generation_tokens / seconds};
        };

        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0\n"
                  << "Parameters represented by INT8 matrices: " << std::fixed << std::setprecision(3)
                  << (index.matrix_bytes / (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
                  << "Double-buffered learned-weight budget: " << (M5_WEIGHT_BUDGET / (1024 * 1024))
                  << " MiB (actual cache + norms " << (learned_weight_capacity / (1024.0 * 1024.0))
                  << " MiB)\n"
                  << "Prompt tokens: " << prompt_tokens.size() << ", measured generated tokens: "
                  << generation_tokens << "\n";

        const ModeResult synchronous = run_mode(false);
        const ModeResult asynchronous = run_mode(true);
        const size_t prompt_count = prompt_tokens.size();
        const std::vector<uint32_t> sync_generated(synchronous.tokens.begin() + prompt_count,
                                                   synchronous.tokens.end());
        const std::vector<uint32_t> async_generated(asynchronous.tokens.begin() + prompt_count,
                                                    asynchronous.tokens.end());
        if (sync_generated != async_generated) {
            throw std::runtime_error("Synchronous and asynchronous greedy token sequences differ");
        }
        std::cout << "Synchronous queue-idle generation: " << synchronous.tokens_per_second
                  << " tok/s (" << synchronous.seconds << " s)\n"
                  << "Async prefetched double-buffer generation: " << asynchronous.tokens_per_second
                  << " tok/s (" << asynchronous.seconds << " s)\n"
                  << "Speedup: " << (asynchronous.tokens_per_second / synchronous.tokens_per_second)
                  << "x\nGenerated token IDs:";
        for (uint32_t token : async_generated) std::cout << " " << token;
        std::cout << "\n--- generated text ---\n" << tokenizer.decode(asynchronous.tokens)
                  << "\n--- end ---\n"
                  << "RESULT: PASS - 1.1B model streamed with async double buffering on Vulkan\n";

        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        for (VkFence fence : fences) vkfn::DestroyFence(runtime.device, fence, nullptr);
        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines) vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        for (VkShaderModule module : resources.shader_modules) vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        destroy_buffer(runtime, down);
        for (Buffer* buffer : auxiliary_buffers) destroy_buffer(runtime, *buffer);
        cache1.destroy();
        cache0.destroy();
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
