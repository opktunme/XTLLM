#define OVLLM_M7_RUNTIME_ONLY
#include "m6_moe.cpp"

#include <future>
#include <memory>

constexpr uint32_t M7_CACHE_SLOTS_PER_LAYER = 6;

struct FullExpertWeights {
    GpuMatrix gate_up;
    GpuMatrix down;
};

struct FullLayerWeights {
    GpuMatrix qkv;
    GpuMatrix output;
    std::vector<FullExpertWeights> experts;
};

struct FullWeights {
    GpuMatrix embedding;
    GpuMatrix lm_head;
    std::vector<FullLayerWeights> layers;
    uint64_t bytes = 0;
};

static uint32_t choose_device_memory_type(const Runtime& runtime, uint32_t allowed) {
    for (uint32_t pass = 0; pass < 2; ++pass) {
        for (uint32_t type = 0; type < runtime.memory_properties.memoryTypeCount; ++type) {
            if (!(allowed & (1u << type))) continue;
            const VkMemoryPropertyFlags flags =
                runtime.memory_properties.memoryTypes[type].propertyFlags;
            if (!(flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            const bool host_visible = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
            if ((pass == 0 && !host_visible) || pass == 1) return type;
        }
    }
    throw std::runtime_error("No device-local memory type is available");
}

static bool last_device_allocation_oom = false;

static Buffer create_device_buffer(const Runtime& runtime, VkDeviceSize size) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocation.pNext = &address_flags;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_device_memory_type(runtime, requirements.memoryTypeBits);
    const VkResult allocation_result =
        vkfn::AllocateMemory(runtime.device, &allocation, nullptr, &buffer.memory);
    if (allocation_result != VK_SUCCESS) {
        if (allocation_result == VK_ERROR_OUT_OF_DEVICE_MEMORY)
            last_device_allocation_oom = true;
        vkfn::DestroyBuffer(runtime.device, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
        throw std::runtime_error("device-local Vulkan allocation failed (" +
            std::to_string(static_cast<uint64_t>(size)) + " bytes, VkResult " +
            std::to_string(static_cast<int32_t>(allocation_result)) + ")");
    }
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes, active_vulkan_buffer_bytes);
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle, buffer.memory, 0));
    return buffer;
}

class DeviceUploader {
public:
    DeviceUploader(const Runtime& runtime) : runtime_(runtime) {
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = runtime.queue_family;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &pool_));
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command_));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkfn::BeginCommandBuffer(command_, &begin));
    }

    template <typename T>
    Buffer upload(const std::vector<T>& values) {
        Buffer staging = upload_vector(runtime_, values);
        Buffer device = create_device_buffer(runtime_, staging.size);
        VkBufferCopy region{};
        region.size = staging.size;
        vkfn::CmdCopyBuffer(command_, staging.handle, device.handle, 1, &region);
        staging_.push_back(staging);
        return device;
    }

    void finish() {
        VK_CHECK(vkfn::EndCommandBuffer(command_));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        VK_CHECK(vkfn::QueueSubmit(runtime_.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkfn::QueueWaitIdle(runtime_.queue));
        for (Buffer& staging : staging_) destroy_buffer(runtime_, staging);
        staging_.clear();
        vkfn::DestroyCommandPool(runtime_.device, pool_, nullptr);
        pool_ = VK_NULL_HANDLE;
    }

private:
    const Runtime& runtime_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    std::vector<Buffer> staging_;
};

static GpuMatrix load_indexed_matrix(std::ifstream& input, const Runtime& runtime,
                                     DeviceUploader& uploader, const MatrixIndex& source,
                                     uint64_t& bytes) {
    GpuMatrix matrix;
    matrix.rows = source.rows;
    matrix.columns = source.columns;
    matrix.packed_stride = source.packed_stride;
    std::vector<float> scales(source.rows);
    std::vector<uint32_t> packed(static_cast<size_t>(source.rows) * source.packed_stride);
    read_file_at(input, source.scale_offset, scales.data(), scales.size() * sizeof(float),
                 "resident matrix scales");
    read_file_at(input, source.weight_offset, packed.data(), packed.size() * sizeof(uint32_t),
                 "resident packed matrix");
    matrix.scales = uploader.upload(scales);
    matrix.weights = uploader.upload(packed);
    bytes += matrix.scales.size + matrix.weights.size;
    return matrix;
}

static FullWeights load_full_weights(const std::string& path, const Runtime& runtime,
                                     const MoEModelIndex& index) {
    std::ifstream input(path, std::ios::binary);
    DeviceUploader uploader(runtime);
    FullWeights weights;
    weights.embedding = load_indexed_matrix(input, runtime, uploader, index.token_embedding,
                                            weights.bytes);
    weights.lm_head = load_indexed_matrix(input, runtime, uploader, index.lm_head, weights.bytes);
    weights.layers.resize(index.header.layers);
    for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
        const MoELayerIndex& source = index.layers[layer_number];
        FullLayerWeights& layer = weights.layers[layer_number];
        layer.qkv = load_indexed_matrix(input, runtime, uploader, source.qkv, weights.bytes);
        layer.output = load_indexed_matrix(input, runtime, uploader, source.output, weights.bytes);
        layer.experts.resize(index.extension.experts);
        for (uint32_t expert = 0; expert < index.extension.experts; ++expert) {
            layer.experts[expert].gate_up = load_indexed_matrix(
                input, runtime, uploader, source.experts[expert].gate_up, weights.bytes);
            layer.experts[expert].down = load_indexed_matrix(
                input, runtime, uploader, source.experts[expert].down, weights.bytes);
        }
    }
    uploader.finish();
    return weights;
}

static void destroy_full_weights(const Runtime& runtime, FullWeights& weights) {
    for (FullLayerWeights& layer : weights.layers) {
        for (FullExpertWeights& expert : layer.experts) {
            destroy_matrix(runtime, expert.down);
            destroy_matrix(runtime, expert.gate_up);
        }
        destroy_matrix(runtime, layer.output);
        destroy_matrix(runtime, layer.qkv);
    }
    destroy_matrix(runtime, weights.lm_head);
    destroy_matrix(runtime, weights.embedding);
}

static void destroy_full_experts(const Runtime& runtime, FullWeights& weights) {
    for (FullLayerWeights& layer : weights.layers) {
        for (FullExpertWeights& expert : layer.experts) {
            destroy_matrix(runtime, expert.down);
            destroy_matrix(runtime, expert.gate_up);
        }
        layer.experts.clear();
    }
}

class MappedModelFile {
public:
    explicit MappedModelFile(const std::string& path) {
        file_ = CreateFileA(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("Could not map model file");
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file_, &length) || length.QuadPart <= 0) {
            throw std::runtime_error("Could not determine mapped model size");
        }
        size_ = static_cast<uint64_t>(length.QuadPart);
        // Vulkan requires an importable writable host view even though this runtime binds
        // it exclusively as a transfer source.  The shared mapping avoids COW duplication.
        mapping_ = CreateFileMappingA(file_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (!mapping_) throw std::runtime_error("CreateFileMapping failed for model");
        data_ = static_cast<const uint8_t*>(MapViewOfFile(
            mapping_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0));
        if (!data_) throw std::runtime_error("MapViewOfFile failed for model");
    }

    ~MappedModelFile() {
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }

private:
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
};

class ExpertSlot {
public:
    ExpertSlot(const MappedModelFile& source, const Runtime& runtime, uint32_t dimension,
               uint32_t hidden)
        : source_(source), runtime_(runtime) {
        const uint32_t gate_rows = 2 * hidden;
        const uint32_t gate_stride = (dimension + 3) / 4;
        const uint32_t down_stride = (hidden + 3) / 4;
        gate_scales = create_device_buffer(runtime, gate_rows * sizeof(float));
        gate_weights = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(gate_rows) * gate_stride * sizeof(uint32_t));
        down_scales = create_device_buffer(runtime, dimension * sizeof(float));
        down_weights = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(dimension) * down_stride * sizeof(uint32_t));
        upload_ = create_buffer(runtime, gate_scales.size + gate_weights.size +
                                down_scales.size + down_weights.size);
    }

    void load(const ExpertIndex& expert, uint32_t expert_number) {
        const uint64_t begin = expert.gate_up.scale_offset;
        const uint64_t end = expert.down.weight_offset +
            static_cast<uint64_t>(expert.down.rows) * expert.down.packed_stride * sizeof(uint32_t);
        if (end - begin != upload_.size) {
            throw std::runtime_error("Expert tensors are not contiguous in the streaming file");
        }
        if (end > source_.size()) throw std::runtime_error("Mapped expert read exceeds model file");
        std::memcpy(upload_.mapped, source_.data() + begin, static_cast<size_t>(upload_.size));
        flush_buffer(runtime_, upload_);
        resident_expert = static_cast<int32_t>(expert_number);
        dirty_ = true;
        ++loads;
        bytes_read += upload_.size;
    }

    bool record_upload(VkCommandBuffer command) {
        if (!dirty_) return false;
        VkDeviceSize source_offset = 0;
        const auto copy_to = [&](const Buffer& destination) {
            VkBufferCopy region{};
            region.srcOffset = source_offset;
            region.size = destination.size;
            vkfn::CmdCopyBuffer(command, upload_.handle, destination.handle, 1, &region);
            source_offset += destination.size;
        };
        copy_to(gate_scales);
        copy_to(gate_weights);
        copy_to(down_scales);
        copy_to(down_weights);
        dirty_ = false;
        return true;
    }

    uint64_t capacity() const {
        return gate_scales.size + gate_weights.size + down_scales.size + down_weights.size;
    }

    void destroy() {
        destroy_buffer(runtime_, upload_);
        destroy_buffer(runtime_, down_weights);
        destroy_buffer(runtime_, down_scales);
        destroy_buffer(runtime_, gate_weights);
        destroy_buffer(runtime_, gate_scales);
    }

    Buffer gate_scales;
    Buffer gate_weights;
    Buffer down_scales;
    Buffer down_weights;
    int32_t resident_expert = -1;
    uint64_t last_used = 0;
    uint64_t use_count = 0;
    uint64_t loads = 0;
    uint64_t bytes_read = 0;

private:
    const MappedModelFile& source_;
    const Runtime& runtime_;
    Buffer upload_;
    bool dirty_ = false;
};

struct FullExpertSets { VkDescriptorSet gate_up, down; };
struct SlotSets { VkDescriptorSet gate_up, down; };

struct M7LayerSets {
    VkDescriptorSet norm1;
    VkDescriptorSet qkv;
    VkDescriptorSet rope;
    VkDescriptorSet attention;
    VkDescriptorSet output;
    VkDescriptorSet residual1;
    VkDescriptorSet norm2;
    VkDescriptorSet router;
    VkDescriptorSet swiglu;
    VkDescriptorSet weighted;
    VkDescriptorSet residual2;
    std::vector<FullExpertSets> full_experts;
    std::vector<SlotSets> slots;
};

struct M7Result {
    std::vector<uint32_t> tokens;
    double seconds = 0.0;
    double tokens_per_second = 0.0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t streamed_bytes = 0;
};

#ifndef OVLLM_M8_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\tinymoe-100m-2x8-chat\\runtime";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "In a small village, there lived a friendly dragon named";
        const uint32_t generation_tokens = argc > 4 ?
            static_cast<uint32_t>(std::stoul(argv[4])) : 24;
        const std::string model_path = model_directory + "\\model.ovm";

        Runtime runtime = create_runtime();
        const MoEModelIndex index = index_moe_model(model_path);
        ResidentMoEModel model = load_moe_resident_state(model_path, runtime, index);
        FullWeights full = load_full_weights(model_path, runtime, index);
        MappedModelFile mapped_model(model_path);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        const uint32_t dimension = index.header.dimension;
        const uint32_t hidden = index.header.hidden_dimension;
        const uint32_t head_dimension = dimension / index.header.heads;
        const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
        const uint32_t qkv_dimension = dimension + 2 * kv_dimension;

        std::vector<std::vector<std::unique_ptr<ExpertSlot>>> caches(index.header.layers);
        uint64_t stream_cache_bytes = 0;
        for (auto& layer : caches) {
            layer.resize(M7_CACHE_SLOTS_PER_LAYER);
            for (auto& slot : layer) {
                slot = std::make_unique<ExpertSlot>(mapped_model, runtime, dimension, hidden);
                stream_cache_bytes += slot->capacity();
            }
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
        Buffer logits = create_buffer(runtime, index.header.vocabulary * sizeof(float));
        std::array<Buffer*, 13> auxiliary_buffers = {
            &rope_cos_buffer, &rope_sin_buffer, &dummy, &hidden_a, &hidden_b, &norm, &qkv,
            &context, &projection, &gate_up, &feed_forward, &expert_output, &moe_sum};

        ComputeResources resources = create_compute_resources(runtime, 512);
        const auto shader = [&](const char* name) { return shader_directory + "\\" + name + ".comp.spv"; };
        const VkPipeline embedding_pipeline = create_pipeline(runtime, resources, shader("embedding"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline qgemv_pipeline = create_pipeline(runtime, resources, shader("qgemv"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope_cache"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention_cache"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));
        const VkPipeline router_pipeline = create_pipeline(runtime, resources, shader("moe_router"));
        const VkPipeline weighted_pipeline = create_pipeline(runtime, resources,
                                                              shader("weighted_accumulate"));

        const VkDescriptorSet embedding_set = create_descriptor_set(runtime, resources,
            {&full.embedding.weights, &full.embedding.scales, &hidden_a, &dummy});
        const VkDescriptorSet logits_set = create_descriptor_set(runtime, resources,
            {&norm, &full.lm_head.weights, &full.lm_head.scales, &logits});
        const VkDescriptorSet final_norm_set = create_descriptor_set(runtime, resources,
            {&hidden_a, &model.final_norm, &norm, &dummy});
        std::vector<M7LayerSets> sets(index.header.layers);
        for (uint32_t number = 0; number < index.header.layers; ++number) {
            M7LayerSets& set = sets[number];
            ResidentMoELayer& layer = model.layers[number];
            FullLayerWeights& layer_weights = full.layers[number];
            set.norm1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &layer.attention_norm, &norm, &dummy});
            set.qkv = create_descriptor_set(runtime, resources,
                {&norm, &layer_weights.qkv.weights, &layer_weights.qkv.scales, &qkv});
            set.rope = create_descriptor_set(runtime, resources,
                {&qkv, &rope_cos_buffer, &rope_sin_buffer, &layer.attention_state});
            set.attention = create_descriptor_set(runtime, resources,
                {&layer.attention_state, &context, &dummy, &dummy});
            set.output = create_descriptor_set(runtime, resources,
                {&context, &layer_weights.output.weights, &layer_weights.output.scales, &projection});
            set.residual1 = create_descriptor_set(runtime, resources,
                {&hidden_a, &projection, &hidden_b, &dummy});
            set.norm2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &layer.feed_forward_norm, &norm, &dummy});
            set.router = create_descriptor_set(runtime, resources,
                {&norm, &layer.router, &routing, &dummy});
            set.swiglu = create_descriptor_set(runtime, resources,
                {&gate_up, &feed_forward, &dummy, &dummy});
            set.weighted = create_descriptor_set(runtime, resources,
                {&expert_output, &moe_sum, &dummy, &dummy});
            set.residual2 = create_descriptor_set(runtime, resources,
                {&hidden_b, &moe_sum, &hidden_a, &dummy});
            set.full_experts.resize(index.extension.experts);
            set.slots.resize(M7_CACHE_SLOTS_PER_LAYER);
            for (uint32_t expert = 0; expert < index.extension.experts; ++expert) {
                FullExpertWeights& expert_weights = layer_weights.experts[expert];
                set.full_experts[expert].gate_up = create_descriptor_set(runtime, resources,
                    {&norm, &expert_weights.gate_up.weights, &expert_weights.gate_up.scales, &gate_up});
                set.full_experts[expert].down = create_descriptor_set(runtime, resources,
                    {&feed_forward, &expert_weights.down.weights, &expert_weights.down.scales,
                     &expert_output});
            }
            for (uint32_t slot_number = 0; slot_number < M7_CACHE_SLOTS_PER_LAYER; ++slot_number) {
                ExpertSlot& slot = *caches[number][slot_number];
                set.slots[slot_number].gate_up = create_descriptor_set(runtime, resources,
                    {&norm, &slot.gate_weights, &slot.gate_scales, &gate_up});
                set.slots[slot_number].down = create_descriptor_set(runtime, resources,
                    {&feed_forward, &slot.down_weights, &slot.down_scales, &expert_output});
            }
        }

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
            completion.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
            VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        };

        const RmsPush rms_push{1, dimension, index.header.rms_epsilon, 0};
        const AddPush add_push{dimension, 0, 0, 0};
        const SwiGluPush swiglu_push{1, hidden, 0, 0};
        const RouterPush router_push{dimension, index.extension.experts, 2, 0};
        const auto qgemv = [&](VkCommandBuffer command, VkDescriptorSet descriptor,
                               uint32_t columns, uint32_t inner, uint32_t stride) {
            const LinearPush push{1, columns, inner, stride};
            dispatch(command, resources, qgemv_pipeline, descriptor, &push,
                     (columns + 7) / 8, 1);
        };
        const auto record_shared_layer = [&](VkCommandBuffer command, uint32_t layer_number,
                                             uint32_t position) {
            const MoELayerIndex& layer = index.layers[layer_number];
            const M7LayerSets& set = sets[layer_number];
            const CachePush cache{position, dimension, head_dimension, kv_dimension};
            qgemv(command, set.qkv, qkv_dimension, dimension, layer.qkv.packed_stride);
            compute_barrier(command);
            dispatch(command, resources, rope_pipeline, set.rope, &cache,
                     (dimension + 63) / 64, 1);
            compute_barrier(command);
            dispatch(command, resources, attention_pipeline, set.attention, &cache,
                     index.header.heads, 1);
            compute_barrier(command);
            qgemv(command, set.output, dimension, dimension, layer.output.packed_stride);
            compute_barrier(command);
            dispatch(command, resources, add_pipeline, set.residual1, &add_push,
                     (dimension + 63) / 64, 1);
            compute_barrier(command);
            dispatch(command, resources, rmsnorm_pipeline, set.norm2, &rms_push, 1, 1);
            compute_barrier(command);
            dispatch(command, resources, router_pipeline, set.router, &router_push, 1, 1);
        };
        const auto read_route = [&]() {
            invalidate_buffer(runtime, routing);
            const float* values = static_cast<const float*>(routing.mapped);
            std::array<uint32_t, 2> selected{
                static_cast<uint32_t>(std::lround(values[0])),
                static_cast<uint32_t>(std::lround(values[1]))};
            std::array<float, 2> route_weights{values[2], values[3]};
            if (selected[0] >= index.extension.experts || selected[1] >= index.extension.experts ||
                selected[0] == selected[1]) throw std::runtime_error("Invalid M7 router result");
            return std::make_pair(selected, route_weights);
        };

        const auto run_mode = [&](bool streamed) {
            uint64_t hits = 0, misses = 0, bytes_read = 0;
            uint64_t cache_clock = 0;
            const auto cache_counters = [&]() {
                uint64_t loads = 0, bytes = 0;
                for (const auto& layer : caches) for (const auto& slot : layer) {
                    loads += slot->loads;
                    bytes += slot->bytes_read;
                }
                return std::make_pair(loads, bytes);
            };
            const auto run_token = [&](uint32_t token, uint32_t position) {
                execute([&](VkCommandBuffer command) {
                    const EmbeddingPush embedding{token, dimension,
                                                  index.token_embedding.packed_stride, 0};
                    dispatch(command, resources, embedding_pipeline, embedding_set, &embedding,
                             (dimension + 63) / 64, 1);
                    compute_barrier(command);
                    dispatch(command, resources, rmsnorm_pipeline, sets[0].norm1,
                             &rms_push, 1, 1);
                    compute_barrier(command);
                    record_shared_layer(command, 0, position);
                });
                auto route = read_route();
                for (uint32_t layer_number = 0; layer_number < index.header.layers; ++layer_number) {
                    std::array<VkDescriptorSet, 2> gate_sets{};
                    std::array<VkDescriptorSet, 2> down_sets{};
                    std::array<int32_t, 2> stream_slots{-1, -1};
                    if (!streamed) {
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            const FullExpertSets& expert =
                                sets[layer_number].full_experts[route.first[rank]];
                            gate_sets[rank] = expert.gate_up;
                            down_sets[rank] = expert.down;
                        }
                    } else {
                        std::vector<bool> occupied(M7_CACHE_SLOTS_PER_LAYER, false);
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            for (uint32_t slot = 0; slot < M7_CACHE_SLOTS_PER_LAYER; ++slot) {
                                if (caches[layer_number][slot]->resident_expert ==
                                    static_cast<int32_t>(route.first[rank])) {
                                    stream_slots[rank] = static_cast<int32_t>(slot);
                                    occupied[slot] = true;
                                    caches[layer_number][slot]->last_used = ++cache_clock;
                                    ++hits;
                                    break;
                                }
                            }
                        }
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            if (stream_slots[rank] >= 0) continue;
                            uint32_t victim = 0;
                            uint64_t oldest = std::numeric_limits<uint64_t>::max();
                            for (uint32_t slot = 0; slot < M7_CACHE_SLOTS_PER_LAYER; ++slot) {
                                if (!occupied[slot] && caches[layer_number][slot]->last_used < oldest) {
                                    oldest = caches[layer_number][slot]->last_used;
                                    victim = slot;
                                }
                            }
                            stream_slots[rank] = static_cast<int32_t>(victim);
                            occupied[victim] = true;
                            caches[layer_number][victim]->last_used = ++cache_clock;
                            ++misses;
                        }
                        std::vector<std::future<void>> reads;
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            ExpertSlot& slot = *caches[layer_number][stream_slots[rank]];
                            if (slot.resident_expert != static_cast<int32_t>(route.first[rank])) {
                                const ExpertIndex expert =
                                    index.layers[layer_number].experts[route.first[rank]];
                                const uint32_t expert_number = route.first[rank];
                                reads.push_back(std::async(std::launch::async,
                                    [&slot, expert, expert_number] { slot.load(expert, expert_number); }));
                            }
                        }
                        for (auto& read : reads) read.get();
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            const SlotSets& slot = sets[layer_number].slots[stream_slots[rank]];
                            gate_sets[rank] = slot.gate_up;
                            down_sets[rank] = slot.down;
                        }
                    }

                    execute([&](VkCommandBuffer command) {
                        const M7LayerSets& set = sets[layer_number];
                        bool copied = false;
                        if (streamed) {
                            for (uint32_t rank = 0; rank < 2; ++rank) {
                                copied |= caches[layer_number][stream_slots[rank]]->record_upload(command);
                            }
                        }
                        if (copied) {
                            VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                            transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                            transfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                1, &transfer, 0, nullptr, 0, nullptr);
                        }
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            const LinearPush gate_push{1, 2 * hidden, dimension,
                                                      index.layers[layer_number]
                                                          .experts[route.first[rank]].gate_up.packed_stride};
                            dispatch(command, resources, qgemv_pipeline, gate_sets[rank], &gate_push,
                                     (2 * hidden + 7) / 8, 1);
                            compute_barrier(command);
                            dispatch(command, resources, swiglu_pipeline, set.swiglu,
                                     &swiglu_push, (hidden + 63) / 64, 1);
                            compute_barrier(command);
                            const LinearPush down_push{1, dimension, hidden,
                                                      index.layers[layer_number]
                                                          .experts[route.first[rank]].down.packed_stride};
                            dispatch(command, resources, qgemv_pipeline, down_sets[rank], &down_push,
                                     (dimension + 7) / 8, 1);
                            compute_barrier(command);
                            const WeightedPush weighted{dimension, route.second[rank],
                                                        rank == 0 ? 1u : 0u, 0};
                            dispatch(command, resources, weighted_pipeline, set.weighted,
                                     &weighted, (dimension + 63) / 64, 1);
                            compute_barrier(command);
                        }
                        dispatch(command, resources, add_pipeline, set.residual2,
                                 &add_push, (dimension + 63) / 64, 1);
                        compute_barrier(command);
                        if (layer_number + 1 < index.header.layers) {
                            dispatch(command, resources, rmsnorm_pipeline,
                                     sets[layer_number + 1].norm1, &rms_push, 1, 1);
                            compute_barrier(command);
                            record_shared_layer(command, layer_number + 1, position);
                        } else {
                            dispatch(command, resources, rmsnorm_pipeline, final_norm_set,
                                     &rms_push, 1, 1);
                            compute_barrier(command);
                            qgemv(command, logits_set, index.header.vocabulary, dimension,
                                  index.lm_head.packed_stride);
                        }
                    });
                    if (layer_number + 1 < index.header.layers) route = read_route();
                }
                invalidate_buffer(runtime, logits);
            };

            std::vector<uint32_t> tokens = tokenizer.encode(prompt, true);
            if (tokens.size() + generation_tokens > M6_CACHE_SEQUENCE) {
                throw std::runtime_error("M7 prompt exceeds KV cache");
            }
            for (uint32_t position = 0; position < tokens.size(); ++position) {
                run_token(tokens[position], position);
            }
            const auto before = cache_counters();
            hits = misses = 0;
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t generated = 0; generated < generation_tokens; ++generated) {
                const float* logit_values = static_cast<const float*>(logits.mapped);
                const uint32_t next = static_cast<uint32_t>(
                    std::max_element(logit_values, logit_values + index.header.vocabulary) -
                    logit_values);
                tokens.push_back(next);
                run_token(next, static_cast<uint32_t>(tokens.size() - 1));
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const auto after = cache_counters();
            bytes_read = after.second - before.second;
            return M7Result{tokens, seconds, generation_tokens / seconds,
                            hits, misses, bytes_read};
        };

        const M7Result resident = run_mode(false);
        // The streamed measurement must not retain a hidden full expert copy in
        // Vulkan memory. Shared attention/embedding/output weights stay resident;
        // all 80 expert matrices are released before demand-filled cache execution.
        destroy_full_experts(runtime, full);
        const M7Result streamed = run_mode(true);
        if (resident.tokens != streamed.tokens) {
            throw std::runtime_error("Resident and streamed deterministic tokens differ");
        }
        const std::vector<uint32_t> expected = {
            7240, 1916, 28725, 7240, 1916, 28723, 650, 403, 264, 16446, 1571, 28725,
            304, 5824, 28723, 1306, 6045, 298, 1156, 395, 652, 6656, 22170, 28723};
        const std::vector<uint32_t> generated(streamed.tokens.end() - generation_tokens,
                                              streamed.tokens.end());
        const bool baseline_identical = generation_tokens != expected.size() || generated == expected;
        if (!baseline_identical) {
            throw std::runtime_error("M7 deterministic tokens differ from the Milestone 6 baseline");
        }

        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << std::fixed << std::setprecision(3)
                  << "Fully resident: " << resident.tokens_per_second << " tok/s\n"
                  << "Expert streamed: " << streamed.tokens_per_second << " tok/s\n"
                  << "Speedup over M6: " << (streamed.tokens_per_second / 16.366) << "x\n"
                  << "Expert cache hits/misses during generation: " << streamed.cache_hits
                  << "/" << streamed.cache_misses << "\n"
                  << "Expert bytes read during generation: "
                  << (streamed.streamed_bytes / (1024.0 * 1024.0)) << " MiB\n"
                  << "Full matrix residency: " << (full.bytes / (1024.0 * 1024.0))
                  << " MiB; streamed device expert cache: "
                  << (stream_cache_bytes / (1024.0 * 1024.0))
                  << " MiB (+ equal reusable upload staging)\nGenerated token IDs:";
        for (uint32_t token : generated) std::cout << " " << token;
        std::cout << "\n--- generated text ---\n" << tokenizer.decode(streamed.tokens)
                  << "\n--- end ---\n"
                  << "RESULT: PASS - resident/streamed tokens identical; M6 sequence preserved\n";

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
        destroy_buffer(runtime, logits);
        destroy_buffer(runtime, routing);
        for (Buffer* buffer : auxiliary_buffers) destroy_buffer(runtime, *buffer);
        for (auto& layer : caches) for (auto& slot : layer) slot->destroy();
        destroy_full_weights(runtime, full);
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
