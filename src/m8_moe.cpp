#define OVLLM_M8_RUNTIME_ONLY
#include "m7_moe.cpp"

#include <numeric>

constexpr uint32_t M8_CACHE_SEQUENCE = 256;
constexpr uint32_t M8_CACHE_SLOTS_PER_LAYER = 16;
constexpr VkDeviceSize M8_UPLOAD_CHUNK = 64ull * 1024 * 1024;

#pragma pack(push, 1)
struct M8Extension {
    uint32_t experts;
    uint32_t top_k;
    uint32_t head_dimension;
    uint32_t query_dimension;
    uint32_t flags;
    uint32_t reserved0;
    uint32_t reserved1;
    float rope_factor;
    float router_jitter;
};
#pragma pack(pop)
static_assert(sizeof(M8Extension) == 36, "Unexpected M8 extension layout");

struct M8LayerIndex {
    uint64_t attention_gamma = 0;
    uint64_t attention_beta = 0;
    uint64_t feed_forward_gamma = 0;
    uint64_t feed_forward_beta = 0;
    MatrixIndex qkv;
    uint64_t qkv_bias = 0;
    MatrixIndex output;
    uint64_t output_bias = 0;
    uint64_t router = 0;
    std::vector<ExpertIndex> experts;
};

struct M8ModelIndex {
    ModelHeader header{};
    M8Extension extension{};
    MatrixIndex embedding;
    MatrixIndex lm_head;
    uint64_t lm_bias = 0;
    uint64_t final_gamma = 0;
    uint64_t final_beta = 0;
    std::vector<M8LayerIndex> layers;
    uint64_t file_bytes = 0;
    uint64_t matrix_bytes = 0;
    uint64_t matrix_rows = 0;
    uint64_t learned_float_bytes = 0;
    uint64_t total_parameters = 0;
    uint64_t expert_parameters = 0;
};

static void m8_add_vector(uint64_t& cursor, uint64_t count, M8ModelIndex& index,
                          uint64_t& offset) {
    offset = cursor;
    cursor += count * sizeof(float);
    index.learned_float_bytes += count * sizeof(float);
    index.total_parameters += count;
}

static MatrixIndex m8_add_matrix(uint64_t& cursor, uint32_t rows, uint32_t columns,
                                 M8ModelIndex& index, bool expert) {
    MatrixIndex result;
    result.rows = rows;
    result.columns = columns;
    const uint32_t weights_per_word = index.header.version == 5 ? 8 : 4;
    result.packed_stride = (columns + weights_per_word - 1) / weights_per_word;
    result.scale_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * sizeof(float);
    result.weight_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * result.packed_stride * sizeof(uint32_t);
    index.matrix_bytes += result.byte_size();
    index.matrix_rows += rows;
    const uint64_t count = static_cast<uint64_t>(rows) * columns;
    index.total_parameters += count;
    if (expert) index.expert_parameters += count;
    return result;
}

static M8ModelIndex index_m8_model(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open converted Phi MoE model: " + path);
    M8ModelIndex index;
    index.file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    input.read(reinterpret_cast<char*>(&index.extension), sizeof(index.extension));
    if (!input || std::memcmp(index.header.magic, "OVLLM3\0\0", 8) != 0 ||
        (index.header.version != 4 && index.header.version != 5) ||
        index.extension.experts != 16 ||
        index.extension.top_k != 2 || index.extension.head_dimension == 0 ||
        index.extension.query_dimension !=
            index.header.heads * index.extension.head_dimension) {
        throw std::runtime_error("Unsupported Phi MoE runtime format");
    }
    uint64_t cursor = sizeof(ModelHeader) + sizeof(M8Extension);
    const uint32_t dimension = index.header.dimension;
    const uint32_t hidden = index.header.hidden_dimension;
    const uint32_t query = index.extension.query_dimension;
    const uint32_t kv = index.header.kv_heads * index.extension.head_dimension;
    index.embedding = m8_add_matrix(cursor, index.header.vocabulary, dimension, index, false);
    index.lm_head = m8_add_matrix(cursor, index.header.vocabulary, dimension, index, false);
    m8_add_vector(cursor, index.header.vocabulary, index, index.lm_bias);
    m8_add_vector(cursor, dimension, index, index.final_gamma);
    m8_add_vector(cursor, dimension, index, index.final_beta);
    index.layers.resize(index.header.layers);
    for (M8LayerIndex& layer : index.layers) {
        m8_add_vector(cursor, dimension, index, layer.attention_gamma);
        m8_add_vector(cursor, dimension, index, layer.attention_beta);
        m8_add_vector(cursor, dimension, index, layer.feed_forward_gamma);
        m8_add_vector(cursor, dimension, index, layer.feed_forward_beta);
        layer.qkv = m8_add_matrix(cursor, query + 2 * kv, dimension, index, false);
        m8_add_vector(cursor, query + 2 * kv, index, layer.qkv_bias);
        layer.output = m8_add_matrix(cursor, dimension, query, index, false);
        m8_add_vector(cursor, dimension, index, layer.output_bias);
        m8_add_vector(cursor, static_cast<uint64_t>(index.extension.experts) * dimension,
                      index, layer.router);
        layer.experts.resize(index.extension.experts);
        for (ExpertIndex& expert : layer.experts) {
            expert.gate_up = m8_add_matrix(cursor, 2 * hidden, dimension, index, true);
            expert.down = m8_add_matrix(cursor, dimension, hidden, index, true);
        }
    }
    if (cursor != index.file_bytes) {
        throw std::runtime_error("Phi MoE model size does not match indexed layout");
    }
    return index;
}

struct M8SharedLayer {
    Buffer attention_gamma;
    Buffer attention_beta;
    Buffer feed_forward_gamma;
    Buffer feed_forward_beta;
    GpuMatrix qkv;
    Buffer qkv_bias;
    GpuMatrix output;
    Buffer output_bias;
    Buffer router;
};

struct M8SharedWeights {
    GpuMatrix embedding;
    GpuMatrix lm_head;
    Buffer lm_bias;
    Buffer final_gamma;
    Buffer final_beta;
    std::vector<M8SharedLayer> layers;
    uint64_t bytes = 0;
};

struct M8BiasWeights {
    Buffer lm;
    Buffer final_norm;
    std::vector<Buffer> qkv;
    std::vector<Buffer> output;
    std::vector<Buffer> attention_norm;
    std::vector<Buffer> feed_norm;
};

static Buffer load_m8_scale_bias(std::ifstream& input, const Runtime& runtime,
                                 const MatrixIndex& matrix, uint64_t bias_offset,
                                 const char* description) {
    std::vector<float> values(static_cast<size_t>(matrix.rows) * 2);
    read_file_at(input, matrix.scale_offset, values.data(),
                 static_cast<uint64_t>(matrix.rows) * sizeof(float), description);
    read_file_at(input, bias_offset, values.data() + matrix.rows,
                 static_cast<uint64_t>(matrix.rows) * sizeof(float), description);
    return upload_vector(runtime, values);
}

static Buffer load_m8_norm_pair(std::ifstream& input, const Runtime& runtime,
                                uint64_t offset, uint32_t dimension,
                                const char* description) {
    std::vector<float> values(static_cast<size_t>(dimension) * 2);
    read_file_at(input, offset, values.data(), values.size() * sizeof(float), description);
    return upload_vector(runtime, values);
}

static M8BiasWeights load_m8_bias_weights(const std::string& path, const Runtime& runtime,
                                          const M8ModelIndex& index) {
    std::ifstream input(path, std::ios::binary);
    M8BiasWeights result;
    result.lm = load_m8_scale_bias(input, runtime, index.lm_head, index.lm_bias, "LM scale/bias");
    result.final_norm = load_m8_norm_pair(input, runtime, index.final_gamma,
        index.header.dimension, "final norm gamma/beta");
    result.qkv.resize(index.header.layers);
    result.output.resize(index.header.layers);
    result.attention_norm.resize(index.header.layers);
    result.feed_norm.resize(index.header.layers);
    for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
        result.qkv[layer] = load_m8_scale_bias(input, runtime, index.layers[layer].qkv,
            index.layers[layer].qkv_bias, "QKV scale/bias");
        result.output[layer] = load_m8_scale_bias(input, runtime, index.layers[layer].output,
            index.layers[layer].output_bias, "output scale/bias");
        result.attention_norm[layer] = load_m8_norm_pair(input, runtime,
            index.layers[layer].attention_gamma, index.header.dimension,
            "attention norm gamma/beta");
        result.feed_norm[layer] = load_m8_norm_pair(input, runtime,
            index.layers[layer].feed_forward_gamma, index.header.dimension,
            "feed norm gamma/beta");
    }
    return result;
}

static void destroy_m8_bias_weights(const Runtime& runtime, M8BiasWeights& weights) {
    for (Buffer& buffer : weights.feed_norm) destroy_buffer(runtime, buffer);
    for (Buffer& buffer : weights.attention_norm) destroy_buffer(runtime, buffer);
    for (Buffer& buffer : weights.output) destroy_buffer(runtime, buffer);
    for (Buffer& buffer : weights.qkv) destroy_buffer(runtime, buffer);
    destroy_buffer(runtime, weights.final_norm);
    destroy_buffer(runtime, weights.lm);
}

static Buffer load_m8_float(std::ifstream& input, const Runtime& runtime, uint64_t offset,
                            uint64_t count, uint64_t& bytes, const char* description) {
    std::vector<float> values(static_cast<size_t>(count));
    read_file_at(input, offset, values.data(), count * sizeof(float), description);
    Buffer result = upload_vector(runtime, values);
    bytes += result.size;
    return result;
}

static M8SharedWeights load_m8_shared(const std::string& path, const Runtime& runtime,
                                     const M8ModelIndex& index) {
    std::ifstream input(path, std::ios::binary);
    DeviceUploader uploader(runtime);
    M8SharedWeights weights;
    weights.embedding = load_indexed_matrix(input, runtime, uploader, index.embedding,
                                             weights.bytes);
    weights.lm_head = load_indexed_matrix(input, runtime, uploader, index.lm_head,
                                           weights.bytes);
    weights.lm_bias = load_m8_float(input, runtime, index.lm_bias, index.header.vocabulary,
                                    weights.bytes, "LM bias");
    weights.final_gamma = load_m8_float(input, runtime, index.final_gamma,
                                        index.header.dimension, weights.bytes, "final gamma");
    weights.final_beta = load_m8_float(input, runtime, index.final_beta,
                                       index.header.dimension, weights.bytes, "final beta");
    weights.layers.resize(index.header.layers);
    for (uint32_t number = 0; number < index.header.layers; ++number) {
        const M8LayerIndex& source = index.layers[number];
        M8SharedLayer& layer = weights.layers[number];
        layer.attention_gamma = load_m8_float(input, runtime, source.attention_gamma,
            index.header.dimension, weights.bytes, "attention LayerNorm gamma");
        layer.attention_beta = load_m8_float(input, runtime, source.attention_beta,
            index.header.dimension, weights.bytes, "attention LayerNorm beta");
        layer.feed_forward_gamma = load_m8_float(input, runtime, source.feed_forward_gamma,
            index.header.dimension, weights.bytes, "MoE LayerNorm gamma");
        layer.feed_forward_beta = load_m8_float(input, runtime, source.feed_forward_beta,
            index.header.dimension, weights.bytes, "MoE LayerNorm beta");
        layer.qkv = load_indexed_matrix(input, runtime, uploader, source.qkv, weights.bytes);
        layer.qkv_bias = load_m8_float(input, runtime, source.qkv_bias, source.qkv.rows,
                                       weights.bytes, "QKV bias");
        layer.output = load_indexed_matrix(input, runtime, uploader, source.output,
                                           weights.bytes);
        layer.output_bias = load_m8_float(input, runtime, source.output_bias,
            index.header.dimension, weights.bytes, "attention output bias");
        layer.router = load_m8_float(input, runtime, source.router,
            static_cast<uint64_t>(index.extension.experts) * index.header.dimension,
            weights.bytes, "router weights");
    }
    uploader.finish();
    return weights;
}

static void destroy_m8_shared(const Runtime& runtime, M8SharedWeights& weights) {
    for (M8SharedLayer& layer : weights.layers) {
        destroy_buffer(runtime, layer.router);
        destroy_buffer(runtime, layer.output_bias);
        destroy_matrix(runtime, layer.output);
        destroy_buffer(runtime, layer.qkv_bias);
        destroy_matrix(runtime, layer.qkv);
        destroy_buffer(runtime, layer.feed_forward_beta);
        destroy_buffer(runtime, layer.feed_forward_gamma);
        destroy_buffer(runtime, layer.attention_beta);
        destroy_buffer(runtime, layer.attention_gamma);
    }
    destroy_buffer(runtime, weights.final_beta);
    destroy_buffer(runtime, weights.final_gamma);
    destroy_buffer(runtime, weights.lm_bias);
    destroy_matrix(runtime, weights.lm_head);
    destroy_matrix(runtime, weights.embedding);
}

static Buffer upload_m8_full_arena(const Runtime& runtime, const MappedModelFile& mapped) {
    Buffer device = create_device_buffer(runtime, mapped.size());
    Buffer staging = create_buffer(runtime, std::min<uint64_t>(M8_UPLOAD_CHUNK, mapped.size()));
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = runtime.queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &pool));
    for (uint64_t offset = 0; offset < mapped.size(); offset += staging.size) {
        const VkDeviceSize bytes = std::min<uint64_t>(staging.size, mapped.size() - offset);
        std::memcpy(staging.mapped, mapped.data() + offset, static_cast<size_t>(bytes));
        flush_buffer(runtime, staging);
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkfn::BeginCommandBuffer(command, &begin));
        VkBufferCopy copy{};
        copy.dstOffset = offset;
        copy.size = bytes;
        vkfn::CmdCopyBuffer(command, staging.handle, device.handle, 1, &copy);
        VK_CHECK(vkfn::EndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
    }
    vkfn::DestroyCommandPool(runtime.device, pool, nullptr);
    destroy_buffer(runtime, staging);
    return device;
}

struct DescriptorRange {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize range = 0;
};

static DescriptorRange whole(const Buffer& buffer) {
    return DescriptorRange{buffer.handle, 0, buffer.size};
}

static DescriptorRange arena_range(const Buffer& arena, uint64_t offset, uint64_t bytes) {
    return DescriptorRange{arena.handle, offset, bytes};
}

static VkDescriptorSet create_range_set(const Runtime& runtime,
                                        const ComputeResources& resources,
                                        const std::array<DescriptorRange, 4>& ranges) {
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = resources.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &resources.descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkfn::AllocateDescriptorSets(runtime.device, &allocation, &set));
    VkDescriptorBufferInfo infos[4]{};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t binding = 0; binding < 4; ++binding) {
        infos[binding].buffer = ranges[binding].buffer;
        infos[binding].offset = ranges[binding].offset;
        infos[binding].range = ranges[binding].range;
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkfn::UpdateDescriptorSets(runtime.device, 4, writes, 0, nullptr);
    return set;
}

static DescriptorRange matrix_scales(const Buffer& arena, const MatrixIndex& matrix) {
    return arena_range(arena, matrix.scale_offset,
                       static_cast<uint64_t>(matrix.rows) * sizeof(float));
}

static DescriptorRange matrix_weights(const Buffer& arena, const MatrixIndex& matrix) {
    return arena_range(arena, matrix.weight_offset,
        static_cast<uint64_t>(matrix.rows) * matrix.packed_stride * sizeof(uint32_t));
}

struct M8Auxiliary {
    Buffer dummy;
    Buffer hidden_a;
    Buffer hidden_b;
    Buffer norm;
    Buffer quant_norm;
    Buffer quant_temp;
    Buffer qkv;
    Buffer context;
    Buffer projection;
    Buffer gate_up;
    Buffer feed_forward;
    Buffer expert_output;
    Buffer moe_sum;
    Buffer routing;
    Buffer router_logits;
    Buffer token_parameter;
    Buffer token_history;
    Buffer logits;
    Buffer rope_cos;
    Buffer rope_sin;
    std::vector<Buffer> attention_states;
};

struct M8ExpertSets { VkDescriptorSet gate_up = VK_NULL_HANDLE, down = VK_NULL_HANDLE; };

struct M8LayerSets {
    VkDescriptorSet norm1 = VK_NULL_HANDLE;
    VkDescriptorSet qkv = VK_NULL_HANDLE;
    VkDescriptorSet qkv_bias = VK_NULL_HANDLE;
    VkDescriptorSet rope = VK_NULL_HANDLE;
    VkDescriptorSet attention = VK_NULL_HANDLE;
    VkDescriptorSet output = VK_NULL_HANDLE;
    VkDescriptorSet output_residual = VK_NULL_HANDLE;
    VkDescriptorSet output_bias = VK_NULL_HANDLE;
    VkDescriptorSet residual1 = VK_NULL_HANDLE;
    VkDescriptorSet norm2 = VK_NULL_HANDLE;
    VkDescriptorSet router = VK_NULL_HANDLE;
    VkDescriptorSet router_select = VK_NULL_HANDLE;
    VkDescriptorSet swiglu = VK_NULL_HANDLE;
    VkDescriptorSet weighted = VK_NULL_HANDLE;
    VkDescriptorSet residual2 = VK_NULL_HANDLE;
    VkDescriptorSet residual1_norm = VK_NULL_HANDLE;
    VkDescriptorSet residual2_norm = VK_NULL_HANDLE;
    VkDescriptorSet resident_gate = VK_NULL_HANDLE;
    VkDescriptorSet resident_down = VK_NULL_HANDLE;
    std::vector<M8ExpertSets> experts;
};

struct M8Sets {
    VkDescriptorSet embedding = VK_NULL_HANDLE;
    VkDescriptorSet quant_norm = VK_NULL_HANDLE;
    VkDescriptorSet quant_context = VK_NULL_HANDLE;
    VkDescriptorSet quant_feed_forward = VK_NULL_HANDLE;
    VkDescriptorSet final_norm = VK_NULL_HANDLE;
    VkDescriptorSet logits = VK_NULL_HANDLE;
    VkDescriptorSet logits_bias = VK_NULL_HANDLE;
    VkDescriptorSet greedy = VK_NULL_HANDLE;
    std::vector<M8LayerSets> layers;
};

static M8Sets create_m8_resident_sets(const Runtime& runtime,
                                      const ComputeResources& resources,
                                      const M8ModelIndex& index, const Buffer& arena,
                                      M8BiasWeights& bias, M8Auxiliary& aux,
                                      bool create_individual_expert_sets = false) {
    const DescriptorRange dummy = whole(aux.dummy);
    M8Sets sets;
    sets.quant_norm = create_range_set(runtime, resources,
        {whole(aux.norm), whole(aux.quant_norm), dummy, dummy});
    sets.quant_context = create_range_set(runtime, resources,
        {whole(aux.context), whole(aux.quant_temp), dummy, dummy});
    sets.quant_feed_forward = create_range_set(runtime, resources,
        {whole(aux.feed_forward), whole(aux.quant_temp), dummy, dummy});
    sets.embedding = create_range_set(runtime, resources,
        {matrix_weights(arena, index.embedding), matrix_scales(arena, index.embedding),
         whole(aux.hidden_a), whole(aux.token_parameter)});
    sets.final_norm = create_range_set(runtime, resources,
        {whole(aux.hidden_a), arena_range(arena, index.final_gamma,
         index.header.dimension * sizeof(float)), whole(aux.norm),
         arena_range(arena, index.final_beta, index.header.dimension * sizeof(float))});
    sets.logits = create_range_set(runtime, resources,
        {whole(aux.norm), matrix_weights(arena, index.lm_head),
         whole(bias.lm), whole(aux.logits)});
    sets.logits_bias = create_range_set(runtime, resources,
        {whole(aux.logits), arena_range(arena, index.lm_bias,
         index.header.vocabulary * sizeof(float)), whole(aux.logits), dummy});
    sets.greedy = create_range_set(runtime, resources,
        {whole(aux.logits), whole(aux.token_parameter), whole(aux.token_history), dummy});
    sets.layers.resize(index.header.layers);
    for (uint32_t number = 0; number < index.header.layers; ++number) {
        const M8LayerIndex& source = index.layers[number];
        M8LayerSets& set = sets.layers[number];
        set.norm1 = create_range_set(runtime, resources,
            {whole(aux.hidden_a), arena_range(arena, source.attention_gamma,
             index.header.dimension * sizeof(float)), whole(aux.norm),
             arena_range(arena, source.attention_beta,
             index.header.dimension * sizeof(float))});
        set.qkv = create_range_set(runtime, resources,
            {whole(aux.norm), matrix_weights(arena, source.qkv),
             whole(bias.qkv[number]), whole(aux.qkv)});
        set.qkv_bias = create_range_set(runtime, resources,
            {whole(aux.qkv), arena_range(arena, source.qkv_bias,
             source.qkv.rows * sizeof(float)), whole(aux.qkv), dummy});
        set.rope = create_range_set(runtime, resources,
            {whole(aux.qkv), whole(aux.rope_cos), whole(aux.rope_sin),
             whole(aux.attention_states[number])});
        set.attention = create_range_set(runtime, resources,
            {whole(aux.attention_states[number]), whole(aux.context), dummy, dummy});
        set.output = create_range_set(runtime, resources,
            {whole(aux.context), matrix_weights(arena, source.output),
             whole(bias.output[number]), whole(aux.projection)});
        set.output_residual = create_range_set(runtime, resources,
            {whole(aux.context), matrix_weights(arena, source.output),
             whole(bias.output[number]), whole(aux.hidden_a)});
        set.output_bias = create_range_set(runtime, resources,
            {whole(aux.projection), arena_range(arena, source.output_bias,
             index.header.dimension * sizeof(float)), whole(aux.projection), dummy});
        set.residual1 = create_range_set(runtime, resources,
            {whole(aux.hidden_a), whole(aux.projection), whole(aux.hidden_b), dummy});
        set.norm2 = create_range_set(runtime, resources,
            {whole(aux.hidden_b), arena_range(arena, source.feed_forward_gamma,
             index.header.dimension * sizeof(float)), whole(aux.norm),
             arena_range(arena, source.feed_forward_beta,
             index.header.dimension * sizeof(float))});
        set.router = create_range_set(runtime, resources,
            {whole(aux.norm), arena_range(arena, source.router,
             static_cast<uint64_t>(index.extension.experts) * index.header.dimension *
             sizeof(float)), whole(aux.routing), dummy});
        set.router_select = create_range_set(runtime, resources,
            {whole(aux.router_logits), whole(aux.routing), dummy, dummy});
        set.swiglu = create_range_set(runtime, resources,
            {whole(aux.gate_up), whole(aux.feed_forward), dummy, dummy});
        set.weighted = create_range_set(runtime, resources,
            {whole(aux.expert_output), whole(aux.moe_sum), dummy, dummy});
        set.residual2 = create_range_set(runtime, resources,
            {whole(aux.hidden_b), whole(aux.moe_sum), whole(aux.hidden_a), dummy});
        const DescriptorRange residual1_left = number == 0 ? whole(aux.hidden_a) : whole(aux.moe_sum);
        set.residual1_norm = create_range_set(runtime, resources,
            {residual1_left, whole(aux.projection),
             arena_range(arena, source.feed_forward_gamma,
                         2ull * index.header.dimension * sizeof(float)), whole(aux.norm)});
        const DescriptorRange next_norm = number + 1 < index.header.layers ?
            arena_range(arena, index.layers[number + 1].attention_gamma,
                        2ull * index.header.dimension * sizeof(float)) :
            arena_range(arena, index.final_gamma,
                        2ull * index.header.dimension * sizeof(float));
        set.residual2_norm = create_range_set(runtime, resources,
            {whole(aux.projection), whole(aux.moe_sum), next_norm, whole(aux.norm)});
        const ExpertIndex& first_expert = source.experts.front();
        const ExpertIndex& last_expert = source.experts.back();
        const uint64_t expert_begin = first_expert.gate_up.scale_offset;
        const uint64_t expert_end = last_expert.down.weight_offset +
            static_cast<uint64_t>(last_expert.down.rows) *
            last_expert.down.packed_stride * sizeof(uint32_t);
        const DescriptorRange expert_block = arena_range(arena, expert_begin,
                                                          expert_end - expert_begin);
        set.resident_gate = create_range_set(runtime, resources,
            {whole(aux.norm), expert_block, whole(aux.routing), whole(aux.feed_forward)});
        set.resident_down = create_range_set(runtime, resources,
            {whole(aux.feed_forward), expert_block, whole(aux.routing), whole(aux.moe_sum)});
        if (create_individual_expert_sets) {
            set.experts.resize(index.extension.experts);
            for (uint32_t expert = 0; expert < index.extension.experts; ++expert) {
                const ExpertIndex& source_expert = source.experts[expert];
                set.experts[expert].gate_up = create_range_set(runtime, resources,
                    {whole(aux.norm), matrix_weights(arena, source_expert.gate_up),
                     matrix_scales(arena, source_expert.gate_up), whole(aux.feed_forward)});
                set.experts[expert].down = create_range_set(runtime, resources,
                    {whole(aux.feed_forward), matrix_weights(arena, source_expert.down),
                     matrix_scales(arena, source_expert.down), whole(aux.moe_sum)});
            }
        }
    }
    return sets;
}

static M8Sets create_m8_streamed_sets(const Runtime& runtime,
                                      const ComputeResources& resources,
                                      const M8ModelIndex& index, M8SharedWeights& shared,
                                      M8BiasWeights& bias,
                                      std::vector<std::vector<std::unique_ptr<ExpertSlot>>>& caches,
                                      M8Auxiliary& aux) {
    const DescriptorRange dummy = whole(aux.dummy);
    M8Sets sets;
    sets.quant_norm = create_descriptor_set(runtime, resources,
        {&aux.norm, &aux.quant_norm, &aux.dummy, &aux.dummy});
    sets.quant_context = create_descriptor_set(runtime, resources,
        {&aux.context, &aux.quant_temp, &aux.dummy, &aux.dummy});
    sets.quant_feed_forward = create_descriptor_set(runtime, resources,
        {&aux.feed_forward, &aux.quant_temp, &aux.dummy, &aux.dummy});
    sets.embedding = create_descriptor_set(runtime, resources,
        {&shared.embedding.weights, &shared.embedding.scales, &aux.hidden_a,
         &aux.token_parameter});
    sets.final_norm = create_descriptor_set(runtime, resources,
        {&aux.hidden_a, &shared.final_gamma, &aux.norm, &shared.final_beta});
    sets.logits = create_descriptor_set(runtime, resources,
        {&aux.norm, &shared.lm_head.weights, &bias.lm, &aux.logits});
    sets.logits_bias = create_descriptor_set(runtime, resources,
        {&aux.logits, &shared.lm_bias, &aux.logits, &aux.dummy});
    sets.layers.resize(index.header.layers);
    for (uint32_t number = 0; number < index.header.layers; ++number) {
        M8SharedLayer& source = shared.layers[number];
        M8LayerSets& set = sets.layers[number];
        set.norm1 = create_descriptor_set(runtime, resources,
            {&aux.hidden_a, &source.attention_gamma, &aux.norm, &source.attention_beta});
        set.qkv = create_descriptor_set(runtime, resources,
            {&aux.norm, &source.qkv.weights, &bias.qkv[number], &aux.qkv});
        set.qkv_bias = create_descriptor_set(runtime, resources,
            {&aux.qkv, &source.qkv_bias, &aux.qkv, &aux.dummy});
        set.rope = create_descriptor_set(runtime, resources,
            {&aux.qkv, &aux.rope_cos, &aux.rope_sin, &aux.attention_states[number]});
        set.attention = create_descriptor_set(runtime, resources,
            {&aux.attention_states[number], &aux.context, &aux.dummy, &aux.dummy});
        set.output = create_descriptor_set(runtime, resources,
            {&aux.context, &source.output.weights, &bias.output[number], &aux.projection});
        set.output_residual = create_descriptor_set(runtime, resources,
            {&aux.context, &source.output.weights, &bias.output[number], &aux.hidden_a});
        set.output_bias = create_descriptor_set(runtime, resources,
            {&aux.projection, &source.output_bias, &aux.projection, &aux.dummy});
        set.residual1 = create_descriptor_set(runtime, resources,
            {&aux.hidden_a, &aux.projection, &aux.hidden_b, &aux.dummy});
        set.norm2 = create_descriptor_set(runtime, resources,
            {&aux.hidden_b, &source.feed_forward_gamma, &aux.norm,
             &source.feed_forward_beta});
        set.router = create_descriptor_set(runtime, resources,
            {&aux.norm, &source.router, &aux.routing, &aux.dummy});
        set.router_select = create_descriptor_set(runtime, resources,
            {&aux.router_logits, &aux.routing, &aux.dummy, &aux.dummy});
        set.swiglu = create_descriptor_set(runtime, resources,
            {&aux.gate_up, &aux.feed_forward, &aux.dummy, &aux.dummy});
        set.weighted = create_descriptor_set(runtime, resources,
            {&aux.expert_output, &aux.moe_sum, &aux.dummy, &aux.dummy});
        set.residual2 = create_descriptor_set(runtime, resources,
            {&aux.hidden_b, &aux.moe_sum, &aux.hidden_a, &aux.dummy});
        Buffer& residual1_left = number == 0 ? aux.hidden_a : aux.moe_sum;
        set.residual1_norm = create_descriptor_set(runtime, resources,
            {&residual1_left, &aux.projection, &bias.feed_norm[number], &aux.norm});
        Buffer& next_norm = number + 1 < index.header.layers ?
            bias.attention_norm[number + 1] : bias.final_norm;
        set.residual2_norm = create_descriptor_set(runtime, resources,
            {&aux.projection, &aux.moe_sum, &next_norm, &aux.norm});
        set.experts.resize(M8_CACHE_SLOTS_PER_LAYER);
        for (uint32_t slot_number = 0; slot_number < M8_CACHE_SLOTS_PER_LAYER; ++slot_number) {
            ExpertSlot& slot = *caches[number][slot_number];
            set.experts[slot_number].gate_up = create_descriptor_set(runtime, resources,
                {&aux.norm, &slot.gate_weights, &slot.gate_scales, &aux.feed_forward});
            set.experts[slot_number].down = create_descriptor_set(runtime, resources,
                {&aux.feed_forward, &slot.down_weights, &slot.down_scales, &aux.moe_sum});
        }
    }
    (void)dummy;
    return sets;
}

struct M8Pipelines {
    VkPipeline embedding = VK_NULL_HANDLE;
    VkPipeline quantize = VK_NULL_HANDLE;
    VkPipeline layernorm = VK_NULL_HANDLE;
    VkPipeline qgemv = VK_NULL_HANDLE;
    VkPipeline qgemv_bias = VK_NULL_HANDLE;
    VkPipeline qgemv_residual = VK_NULL_HANDLE;
    VkPipeline qgemv_swiglu = VK_NULL_HANDLE;
    VkPipeline qgemv_weighted = VK_NULL_HANDLE;
    VkPipeline resident_expert_swiglu = VK_NULL_HANDLE;
    VkPipeline resident_expert_down = VK_NULL_HANDLE;
    VkPipeline rope = VK_NULL_HANDLE;
    VkPipeline attention = VK_NULL_HANDLE;
    VkPipeline add = VK_NULL_HANDLE;
    VkPipeline swiglu = VK_NULL_HANDLE;
    VkPipeline router = VK_NULL_HANDLE;
    VkPipeline router_select = VK_NULL_HANDLE;
    VkPipeline weighted = VK_NULL_HANDLE;
    VkPipeline add_norm = VK_NULL_HANDLE;
    VkPipeline greedy = VK_NULL_HANDLE;
};

struct SparseRouterPush { uint32_t dimension, experts; float jitter; uint32_t unused; };
struct QuantizePush { uint32_t count, group_size, unused0, unused1; };
struct WeightedLinearPush { float signed_weight; uint32_t columns, inner, packed_stride; };
struct ResidentExpertPush { uint32_t rank, unused0, unused1, unused2; };
struct GreedyPush { uint32_t count, position, unused0, unused1; };

struct M8Result {
    std::vector<uint32_t> tokens;
    std::vector<float> final_logits;
    double seconds = 0.0;
    double tokens_per_second = 0.0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t transferred_bytes = 0;
};

#ifndef OVLLM_GPTOSS_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\phi-tiny-moe-3.8b\\runtime-int8";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "Write one short sentence about Vulkan.";
        const uint32_t generation_tokens = argc > 4 ?
            static_cast<uint32_t>(std::stoul(argv[4])) : 16;
        const std::string model_path = model_directory + "\\model.ovm";

        const M8ModelIndex index = index_m8_model(model_path);
        const bool q4 = index.header.version == 5;
        Runtime runtime = create_runtime();
        MappedModelFile mapped(model_path);
        Tokenizer tokenizer(model_directory + "\\tokenizer.ovt");
        if (tokenizer.vocabulary_size() != index.header.vocabulary) {
            throw std::runtime_error("Tokenizer/model vocabulary mismatch");
        }
        const uint32_t dimension = index.header.dimension;
        const uint32_t hidden = index.header.hidden_dimension;
        const uint32_t head_dimension = index.extension.head_dimension;
        const uint32_t query_dimension = index.extension.query_dimension;
        const uint32_t kv_dimension = index.header.kv_heads * head_dimension;
        const uint32_t qkv_dimension = query_dimension + 2 * kv_dimension;

        M8BiasWeights bias_weights = load_m8_bias_weights(model_path, runtime, index);
        Buffer full_arena = upload_m8_full_arena(runtime, mapped);
        M8Auxiliary aux;
        aux.dummy = create_buffer(runtime, sizeof(float));
        aux.hidden_a = create_buffer(runtime, dimension * sizeof(float));
        aux.hidden_b = create_buffer(runtime, dimension * sizeof(float));
        aux.norm = create_buffer(runtime, dimension * sizeof(float));
        aux.quant_norm = create_buffer(runtime, 2176 * sizeof(uint32_t));
        aux.quant_temp = create_buffer(runtime, 2176 * sizeof(uint32_t));
        aux.qkv = create_buffer(runtime, qkv_dimension * sizeof(float));
        aux.context = create_buffer(runtime, query_dimension * sizeof(float));
        aux.projection = create_buffer(runtime, dimension * sizeof(float));
        aux.gate_up = create_buffer(runtime, 2 * hidden * sizeof(float));
        aux.feed_forward = create_buffer(runtime, 2ull * hidden * sizeof(float));
        aux.expert_output = create_buffer(runtime, dimension * sizeof(float));
        aux.moe_sum = create_buffer(runtime, dimension * sizeof(float));
        aux.routing = create_buffer(runtime, 4 * sizeof(float));
        aux.router_logits = create_buffer(runtime, index.extension.experts * sizeof(float));
        aux.token_parameter = create_buffer(runtime, sizeof(uint32_t));
        aux.token_history = create_buffer(runtime, M8_CACHE_SEQUENCE * sizeof(uint32_t));
        aux.logits = create_buffer(runtime, index.header.vocabulary * sizeof(float));
        std::vector<float> rope_cos(static_cast<size_t>(M8_CACHE_SEQUENCE) * head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t position = 0; position < M8_CACHE_SEQUENCE; ++position) {
            for (uint32_t frequency = 0; frequency < head_dimension / 2; ++frequency) {
                const float inverse_frequency = std::pow(index.header.rope_theta,
                    -2.0f * static_cast<float>(frequency) / static_cast<float>(head_dimension));
                const float angle = (static_cast<float>(position) / index.extension.rope_factor) *
                                    inverse_frequency;
                const size_t offset = static_cast<size_t>(position) * (head_dimension / 2) +
                                      frequency;
                rope_cos[offset] = std::cos(angle);
                rope_sin[offset] = std::sin(angle);
            }
        }
        aux.rope_cos = upload_vector(runtime, rope_cos);
        aux.rope_sin = upload_vector(runtime, rope_sin);
        aux.attention_states.resize(index.header.layers);
        for (Buffer& state : aux.attention_states) {
            state = create_buffer(runtime, static_cast<VkDeviceSize>(query_dimension +
                2ull * M8_CACHE_SEQUENCE * kv_dimension) * sizeof(float));
        }

        ComputeResources resources = create_compute_resources(runtime, 4000);
        const auto shader = [&](const char* name) { return shader_directory + "\\" + name +
                                                           ".comp.spv"; };
        M8Pipelines pipelines;
        pipelines.embedding = create_pipeline(runtime, resources,
            shader(q4 ? "embedding_q4" : "embedding_dynamic"));
        pipelines.quantize = create_pipeline(runtime, resources, shader("quantize_q8x2"));
        pipelines.layernorm = create_pipeline(runtime, resources, shader("layernorm_wave"));
        pipelines.qgemv = create_pipeline(runtime, resources, shader("qgemv_wave"));
        pipelines.qgemv_bias = create_pipeline(runtime, resources,
            shader(q4 ? "qgemv_bias_q4" : "qgemv_bias_uvec4"));
        pipelines.qgemv_residual = create_pipeline(runtime, resources,
            shader("qgemv_bias_residual_uvec4"));
        pipelines.qgemv_swiglu = create_pipeline(runtime, resources, shader("qgemv_swiglu"));
        pipelines.qgemv_weighted = create_pipeline(runtime, resources, shader("qgemv_weighted"));
        pipelines.resident_expert_swiglu = create_pipeline(runtime, resources,
            shader(q4 ? "phi_resident_expert_swiglu_q4" :
                        "phi_resident_expert_swiglu_uvec4"));
        pipelines.resident_expert_down = create_pipeline(runtime, resources,
            shader(q4 ? "phi_resident_expert_down_q4" :
                        "phi_resident_expert_down_uvec2"));
        pipelines.rope = create_pipeline(runtime, resources, shader("rope_cache"));
        pipelines.attention = create_pipeline(runtime, resources,
            shader("attention_cache_wave8"));
        pipelines.add = create_pipeline(runtime, resources, shader("add"));
        pipelines.swiglu = create_pipeline(runtime, resources, shader("swiglu"));
        pipelines.router = create_pipeline(runtime, resources,
            shader("sparsemixer_router_wave"));
        pipelines.router_select = create_pipeline(runtime, resources,
            shader("sparsemixer_select"));
        pipelines.weighted = create_pipeline(runtime, resources, shader("weighted_accumulate"));
        pipelines.add_norm = create_pipeline(runtime, resources, shader("add_layernorm"));
        pipelines.greedy = create_pipeline(runtime, resources, shader("greedy_argmax"));

        M8Sets resident_sets = create_m8_resident_sets(runtime, resources, index,
            full_arena, bias_weights, aux);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = runtime.queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &command_pool));
        VkCommandBufferAllocateInfo execute_allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        execute_allocation.commandPool = command_pool;
        execute_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        execute_allocation.commandBufferCount = 1;
        VkCommandBuffer execute_command = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &execute_allocation,
                                               &execute_command));
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence execute_fence = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateFence(runtime.device, &fence_info, nullptr, &execute_fence));
        VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2048;
        VK_CHECK(vkfn::CreateQueryPool(runtime.device, &query_info, nullptr,
                                       &dispatch_profile_pool));
        const auto record_command = [&](VkCommandBuffer command, const auto& record,
                                        bool reset) {
            if (reset) VK_CHECK(vkfn::ResetCommandBuffer(command, 0));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkfn::BeginCommandBuffer(command, &begin));
            if (dispatch_profile_enabled) {
                dispatch_profile_next = 0;
                dispatch_profile_pipelines.clear();
                vkfn::CmdResetQueryPool(command, dispatch_profile_pool, 0, 2048);
            }
            VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &upload, 0, nullptr, 0, nullptr);
            record(command);
            VkMemoryBarrier completion{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            completion.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            completion.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command));
        };
        const auto submit_command = [&](VkCommandBuffer command) {
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, execute_fence));
            VK_CHECK(vkfn::WaitForFences(runtime.device, 1, &execute_fence, VK_TRUE,
                                         UINT64_MAX));
            VK_CHECK(vkfn::ResetFences(runtime.device, 1, &execute_fence));
        };
        const auto submit_commands = [&](const VkCommandBuffer* commands, uint32_t count) {
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = count;
            submit.pCommandBuffers = commands;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, execute_fence));
            VK_CHECK(vkfn::WaitForFences(runtime.device, 1, &execute_fence, VK_TRUE,
                                         UINT64_MAX));
            VK_CHECK(vkfn::ResetFences(runtime.device, 1, &execute_fence));
        };
        const auto execute = [&](const auto& record) {
            record_command(execute_command, record, true);
            submit_command(execute_command);
        };

        const RmsPush norm_push{1, dimension, index.header.rms_epsilon, 0};
        const AddPush add_dimension{dimension, 0, 0, 0};
        const AddPush add_qkv{qkv_dimension, 0, 0, 0};
        const AddPush add_logits{index.header.vocabulary, 0, 0, 0};
        const SwiGluPush swiglu_push{1, hidden, 0, 0};
        const SparseRouterPush router_push{dimension, index.extension.experts,
                                           index.extension.router_jitter, 0};
        const auto quantize = [&](VkCommandBuffer command, VkDescriptorSet set, uint32_t count) {
            const QuantizePush push{count, 256, 0, 0};
            dispatch(command, resources, pipelines.quantize, set, &push, 1, 1);
        };
        const auto qgemv = [&](VkCommandBuffer command, VkDescriptorSet set,
                               uint32_t columns, uint32_t inner, uint32_t stride) {
            const LinearPush push{1, columns, inner, stride};
            dispatch(command, resources, pipelines.qgemv, set, &push,
                     (columns + 3) / 4, 1);
        };
        const auto qgemv_bias = [&](VkCommandBuffer command, VkDescriptorSet set,
                                    uint32_t columns, uint32_t inner, uint32_t stride) {
            const LinearPush push{1, columns, inner, stride};
            dispatch(command, resources, pipelines.qgemv_bias, set, &push,
                     (columns + 3) / 4, 1);
        };
        const auto record_shared_layer = [&](VkCommandBuffer command, const M8Sets& sets,
                                             uint32_t number, uint32_t position) {
            const M8LayerIndex& layer = index.layers[number];
            const M8LayerSets& set = sets.layers[number];
            const CachePush cache{position, query_dimension, head_dimension, kv_dimension};
            qgemv_bias(command, set.qkv, qkv_dimension, dimension, layer.qkv.packed_stride);
            compute_buffer_barrier(command, aux.qkv);
            dispatch(command, resources, pipelines.rope, set.rope, &cache,
                     (query_dimension + 63) / 64, 1);
            compute_buffer_barrier(command, aux.attention_states[number]);
            dispatch(command, resources, pipelines.attention, set.attention, &cache,
                     index.header.heads, 1);
            compute_buffer_barrier(command, aux.context);
            qgemv_bias(command, set.output, dimension, query_dimension,
                       layer.output.packed_stride);
            compute_buffer_barrier(command, aux.projection);
            dispatch(command, resources, pipelines.add, set.residual1, &add_dimension,
                     (dimension + 63) / 64, 1);
            compute_buffer_barrier(command, aux.hidden_b);
            dispatch(command, resources, pipelines.layernorm, set.norm2, &norm_push, 1, 1);
            compute_buffer_barrier(command, aux.norm);
            dispatch(command, resources, pipelines.router, set.router, &router_push, 1, 1);
        };
        const auto read_route = [&]() {
            invalidate_buffer(runtime, aux.routing);
            const float* values = static_cast<const float*>(aux.routing.mapped);
            std::array<uint32_t, 2> experts{
                static_cast<uint32_t>(std::lround(values[0])),
                static_cast<uint32_t>(std::lround(values[1]))};
            std::array<float, 2> weights{values[2], values[3]};
            if (experts[0] >= index.extension.experts ||
                experts[1] >= index.extension.experts || experts[0] == experts[1] ||
                !std::isfinite(weights[0]) || !std::isfinite(weights[1])) {
                throw std::runtime_error("Invalid SparseMixer route");
            }
            return std::make_pair(experts, weights);
        };

        std::vector<std::vector<std::unique_ptr<ExpertSlot>>> caches;
        uint64_t cache_clock = 0;
        const auto run_mode = [&](bool streamed, const M8Sets& sets) {
            uint64_t hits = 0, misses = 0;
            const auto counters = [&]() {
                uint64_t loads = 0, bytes = 0;
                for (const auto& layer : caches) for (const auto& slot : layer) {
                    loads += slot->loads;
                    bytes += slot->bytes_read;
                }
                return std::make_pair(loads, bytes);
            };
            std::array<VkCommandBuffer, M8_CACHE_SEQUENCE> resident_commands{};
            const auto prepare_resident = [&](uint32_t position) {
                if (resident_commands[position] != VK_NULL_HANDLE) return;
                VkCommandBufferAllocateInfo allocation{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                allocation.commandPool = command_pool;
                allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                allocation.commandBufferCount = 1;
                VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation,
                                                       &resident_commands[position]));
                record_command(resident_commands[position], [&](VkCommandBuffer command) {
                        const EmbeddingPush embedding{0, dimension,
                                                      index.embedding.packed_stride, 0};
                        dispatch(command, resources, pipelines.embedding, sets.embedding,
                                 &embedding, (dimension + 63) / 64, 1);
                        compute_buffer_barrier(command, aux.hidden_a);
                        dispatch(command, resources, pipelines.layernorm,
                                 sets.layers[0].norm1, &norm_push, 1, 1);
                        compute_buffer_barrier(command, aux.norm);
                        record_shared_layer(command, sets, 0, position);
                        compute_buffer_barrier(command, aux.routing);
                        for (uint32_t number = 0; number < index.header.layers; ++number) {
                            const M8LayerSets& layer_sets = sets.layers[number];
                            const ResidentExpertPush expert_push{0, 0, 0, 0};
                            dispatch(command, resources, pipelines.resident_expert_swiglu,
                                     layer_sets.resident_gate, &expert_push,
                                     (hidden + 3) / 4, 1);
                            compute_buffer_barrier(command, aux.feed_forward);
                            dispatch(command, resources, pipelines.resident_expert_down,
                                     layer_sets.resident_down, &expert_push,
                                     (dimension + 3) / 4, 1);
                            compute_buffer_barrier(command, aux.moe_sum);
                            dispatch(command, resources, pipelines.add,
                                     layer_sets.residual2, &add_dimension,
                                     (dimension + 63) / 64, 1);
                            compute_buffer_barrier(command, aux.hidden_a);
                            if (number + 1 < index.header.layers) {
                                dispatch(command, resources, pipelines.layernorm,
                                         sets.layers[number + 1].norm1, &norm_push, 1, 1);
                                compute_buffer_barrier(command, aux.norm);
                                record_shared_layer(command, sets, number + 1, position);
                                compute_buffer_barrier(command, aux.routing);
                            } else {
                                dispatch(command, resources, pipelines.layernorm,
                                         sets.final_norm, &norm_push, 1, 1);
                                compute_buffer_barrier(command, aux.norm);
                                qgemv_bias(command, sets.logits, index.header.vocabulary,
                                           dimension, index.lm_head.packed_stride);
                                if (q4) {
                                    compute_buffer_barrier(command, aux.logits);
                                    const GreedyPush greedy_push{
                                        index.header.vocabulary, position, 0, 0};
                                    dispatch(command, resources, pipelines.greedy, sets.greedy,
                                             &greedy_push, 1, 1);
                                }
                            }
                        }
                    }, false);
            };
            const auto run_token = [&](uint32_t token, uint32_t position) {
                *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
                flush_buffer(runtime, aux.token_parameter);
                if (!streamed) {
                    prepare_resident(position);
                    submit_command(resident_commands[position]);
                    invalidate_buffer(runtime, aux.logits);
                    return;
                }
                execute([&](VkCommandBuffer command) {
                    const EmbeddingPush embedding{0, dimension,
                                                  index.embedding.packed_stride, 0};
                    dispatch(command, resources, pipelines.embedding, sets.embedding,
                             &embedding, (dimension + 63) / 64, 1);
                    compute_buffer_barrier(command, aux.hidden_a);
                    dispatch(command, resources, pipelines.layernorm, sets.layers[0].norm1,
                             &norm_push, 1, 1);
                    compute_buffer_barrier(command, aux.norm);
                    record_shared_layer(command, sets, 0, position);
                });
                auto route = read_route();
                for (uint32_t number = 0; number < index.header.layers; ++number) {
                    std::array<uint32_t, 2> selected_sets = route.first;
                    std::array<int32_t, 2> selected_slots{-1, -1};
                    if (streamed) {
                        std::array<bool, M8_CACHE_SLOTS_PER_LAYER> occupied{};
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            for (uint32_t slot = 0; slot < M8_CACHE_SLOTS_PER_LAYER; ++slot) {
                                if (caches[number][slot]->resident_expert ==
                                    static_cast<int32_t>(route.first[rank])) {
                                    selected_slots[rank] = static_cast<int32_t>(slot);
                                    occupied[slot] = true;
                                    caches[number][slot]->last_used = ++cache_clock;
                                    ++caches[number][slot]->use_count;
                                    ++hits;
                                    break;
                                }
                            }
                        }
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            if (selected_slots[rank] >= 0) continue;
                            uint32_t victim = 0;
                            uint64_t least_used = std::numeric_limits<uint64_t>::max();
                            uint64_t oldest = std::numeric_limits<uint64_t>::max();
                            for (uint32_t slot = 0; slot < M8_CACHE_SLOTS_PER_LAYER; ++slot) {
                                if (!occupied[slot]) {
                                    const uint64_t used = caches[number][slot]->use_count;
                                    const uint64_t age = caches[number][slot]->last_used;
                                    if (used < least_used || (used == least_used && age < oldest)) {
                                        least_used = used;
                                        oldest = age;
                                        victim = slot;
                                    }
                                }
                            }
                            selected_slots[rank] = static_cast<int32_t>(victim);
                            occupied[victim] = true;
                            caches[number][victim]->last_used = ++cache_clock;
                            caches[number][victim]->use_count = 1;
                            ++misses;
                        }
                        std::vector<std::future<void>> reads;
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            ExpertSlot& slot = *caches[number][selected_slots[rank]];
                            if (slot.resident_expert != static_cast<int32_t>(route.first[rank])) {
                                const ExpertIndex expert = index.layers[number].experts[
                                    route.first[rank]];
                                const uint32_t expert_number = route.first[rank];
                                reads.push_back(std::async(std::launch::async,
                                    [&slot, expert, expert_number] {
                                        slot.load(expert, expert_number);
                                    }));
                            }
                        }
                        for (auto& read : reads) read.get();
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            selected_sets[rank] = static_cast<uint32_t>(selected_slots[rank]);
                        }
                    }

                    execute([&](VkCommandBuffer command) {
                        bool copied = false;
                        if (streamed) {
                            for (uint32_t rank = 0; rank < 2; ++rank) {
                                copied |= caches[number][selected_slots[rank]]->record_upload(command);
                            }
                        }
                        if (copied) {
                            VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                            transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                            transfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &transfer,
                                0, nullptr, 0, nullptr);
                        }
                        const M8LayerSets& layer_sets = sets.layers[number];
                        for (uint32_t rank = 0; rank < 2; ++rank) {
                            const M8ExpertSets& expert = layer_sets.experts[selected_sets[rank]];
                            const ExpertIndex& expert_index =
                                index.layers[number].experts[route.first[rank]];
                            const LinearPush gate_push{1, hidden, dimension,
                                                       expert_index.gate_up.packed_stride};
                            dispatch(command, resources, pipelines.qgemv_swiglu,
                                     expert.gate_up, &gate_push, (hidden + 3) / 4, 1);
                            compute_buffer_barrier(command, aux.feed_forward);
                            const WeightedLinearPush down_push{
                                rank == 0 ? -route.second[rank] : route.second[rank],
                                dimension, hidden, expert_index.down.packed_stride};
                            dispatch(command, resources, pipelines.qgemv_weighted,
                                     expert.down, &down_push, (dimension + 3) / 4, 1);
                            compute_buffer_barrier(command, aux.moe_sum);
                        }
                        dispatch(command, resources, pipelines.add, layer_sets.residual2,
                                 &add_dimension, (dimension + 63) / 64, 1);
                        compute_buffer_barrier(command, aux.hidden_a);
                        if (number + 1 < index.header.layers) {
                            dispatch(command, resources, pipelines.layernorm,
                                     sets.layers[number + 1].norm1, &norm_push, 1, 1);
                            compute_buffer_barrier(command, aux.norm);
                            record_shared_layer(command, sets, number + 1, position);
                        } else {
                            dispatch(command, resources, pipelines.layernorm, sets.final_norm,
                                     &norm_push, 1, 1);
                            compute_buffer_barrier(command, aux.norm);
                            qgemv_bias(command, sets.logits, index.header.vocabulary,
                                       dimension, index.lm_head.packed_stride);
                        }
                    });
                    if (number + 1 < index.header.layers) route = read_route();
                }
                invalidate_buffer(runtime, aux.logits);
            };

            std::vector<uint32_t> tokens{tokenizer.token_id("<|user|>")};
            const std::vector<uint32_t> content = tokenizer.encode(prompt, false, true);
            tokens.insert(tokens.end(), content.begin(), content.end());
            tokens.push_back(tokenizer.token_id("<|end|>"));
            tokens.push_back(tokenizer.token_id("<|assistant|>"));
            if (tokens.size() + generation_tokens > M8_CACHE_SEQUENCE) {
                throw std::runtime_error("Phi MoE prompt exceeds KV cache");
            }
            for (uint32_t position = 0; position < tokens.size(); ++position) {
                run_token(tokens[position], position);
            }
            if (!streamed) {
                for (uint32_t position = static_cast<uint32_t>(tokens.size());
                     position < tokens.size() + generation_tokens; ++position) {
                    prepare_resident(position);
                }
            }
            const auto before = counters();
            hits = misses = 0;
            const auto start = std::chrono::steady_clock::now();
            if (!streamed && q4) {
                const uint32_t first_position = static_cast<uint32_t>(tokens.size());
                submit_commands(resident_commands.data() + first_position,
                                generation_tokens);
                invalidate_buffer(runtime, aux.token_history);
                invalidate_buffer(runtime, aux.logits);
                const uint32_t* history = static_cast<const uint32_t*>(aux.token_history.mapped);
                for (uint32_t generated = 0; generated < generation_tokens; ++generated) {
                    tokens.push_back(history[first_position - 1u + generated]);
                }
            } else for (uint32_t generated = 0; generated < generation_tokens; ++generated) {
                const float* values = static_cast<const float*>(aux.logits.mapped);
                const uint32_t next = static_cast<uint32_t>(
                    std::max_element(values, values + index.header.vocabulary) - values);
                tokens.push_back(next);
                run_token(next, static_cast<uint32_t>(tokens.size() - 1));
                if (dispatch_profile_enabled) {
                    dispatch_profile_enabled = false;
                    std::vector<uint64_t> stamps(dispatch_profile_next);
                    VK_CHECK(vkfn::GetQueryPoolResults(runtime.device, dispatch_profile_pool,
                        0, dispatch_profile_next, stamps.size() * sizeof(uint64_t),
                        stamps.data(), sizeof(uint64_t),
                        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                    std::array<double, 10> elapsed{};
                    for (size_t item = 0; item < dispatch_profile_pipelines.size(); ++item) {
                        const VkPipeline pipeline = dispatch_profile_pipelines[item];
                        size_t category = 9;
                        if (pipeline == pipelines.embedding) category = 0;
                        else if (pipeline == pipelines.layernorm) category = 1;
                        else if (pipeline == pipelines.qgemv_bias) category = 2;
                        else if (pipeline == pipelines.rope) category = 3;
                        else if (pipeline == pipelines.attention) category = 4;
                        else if (pipeline == pipelines.add) category = 5;
                        else if (pipeline == pipelines.router ||
                                 pipeline == pipelines.router_select) category = 6;
                        else if (pipeline == pipelines.resident_expert_swiglu) category = 7;
                        else if (pipeline == pipelines.resident_expert_down) category = 8;
                        elapsed[category] += (stamps[item * 2 + 1] - stamps[item * 2]) *
                                             runtime.properties.limits.timestampPeriod * 1e-6;
                    }
                    const std::array<const char*, 10> names{"embed", "norm", "shared_qgemv",
                        "rope", "attention", "add", "router", "expert_gate", "expert_down",
                        "other"};
                    std::cerr << "GPU PROFILE (ms):";
                    for (size_t item = 0; item < names.size(); ++item)
                        std::cerr << " " << names[item] << "=" << elapsed[item];
                    const double span = (stamps.back() - stamps.front()) *
                        runtime.properties.limits.timestampPeriod * 1e-6;
                    std::cerr << " span=" << span << "\n";
                }
            }
            const double seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            const auto after = counters();
            std::vector<float> final_logits(index.header.vocabulary);
            std::memcpy(final_logits.data(), aux.logits.mapped,
                        final_logits.size() * sizeof(float));
            return M8Result{tokens, std::move(final_logits), seconds,
                            generation_tokens / seconds, hits, misses,
                            after.second - before.second};
        };

        const M8Result resident = run_mode(false, resident_sets);
        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        destroy_buffer(runtime, full_arena);

        if (q4) {
            invalidate_buffer(runtime, aux.norm);
            const float* final_input = static_cast<const float*>(aux.norm.mapped);
            const float* scales = reinterpret_cast<const float*>(
                mapped.data() + index.lm_head.scale_offset);
            const uint32_t* packed = reinterpret_cast<const uint32_t*>(
                mapped.data() + index.lm_head.weight_offset);
            const float* lm_bias = reinterpret_cast<const float*>(
                mapped.data() + index.lm_bias);
            float cpu_gpu_max_error = 0.0f;
            const auto check_row = [&](uint32_t row) {
                double sum = 0.0;
                const uint32_t* row_weights = packed +
                    static_cast<uint64_t>(row) * index.lm_head.packed_stride;
                for (uint32_t column = 0; column < dimension; ++column) {
                    const uint32_t word = row_weights[column >> 3u];
                    const uint32_t nibble = (word >> ((column & 7u) * 4u)) & 15u;
                    const int quantized = nibble >= 8u ?
                        static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
                    sum += static_cast<double>(final_input[column]) * quantized;
                }
                const float reference = static_cast<float>(sum * scales[row] + lm_bias[row]);
                cpu_gpu_max_error = std::max(cpu_gpu_max_error,
                    std::abs(reference - resident.final_logits[row]));
            };
            for (uint32_t row = 0; row < index.header.vocabulary; row += 251u) check_row(row);
            check_row(static_cast<uint32_t>(std::max_element(resident.final_logits.begin(),
                resident.final_logits.end()) - resident.final_logits.begin()));
            if (!std::isfinite(cpu_gpu_max_error) || cpu_gpu_max_error > 0.02f) {
                throw std::runtime_error("Packed Q4 GPU/CPU reference check failed");
            }
            const std::vector<uint32_t> generated(resident.tokens.end() - generation_tokens,
                                                  resident.tokens.end());
            const uint64_t matrix_parameters = index.total_parameters -
                index.learned_float_bytes / sizeof(float);
            const uint64_t int8_matrix_bytes = matrix_parameters +
                index.matrix_rows * sizeof(float);
            const double effective_bits = index.matrix_bytes * 8.0 / matrix_parameters;
            const double memory_reduction = 100.0 *
                (1.0 - index.matrix_bytes / static_cast<double>(int8_matrix_bytes));
            std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                      << std::fixed << std::setprecision(3)
                      << "Q4 resident: " << resident.tokens_per_second << " tok/s\n"
                      << "Effective matrix storage: " << effective_bits << " bits/weight\n"
                      << "Matrix memory reduction vs INT8: " << memory_reduction << "%\n"
                      << "Sampled final LM-head CPU/GPU max abs delta: "
                      << cpu_gpu_max_error << "\nGenerated token IDs:";
            for (uint32_t token : generated) std::cout << " " << token;
            std::cout << "\n--- generated text ---\n" << tokenizer.decode(generated)
                      << "\n--- end ---\nRESULT: PASS - native packed Q4 Vulkan inference\n";

            vkfn::DestroyFence(runtime.device, execute_fence, nullptr);
            vkfn::DestroyQueryPool(runtime.device, dispatch_profile_pool, nullptr);
            vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
            for (VkPipeline pipeline : resources.pipelines)
                vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
            for (VkShaderModule module : resources.shader_modules)
                vkfn::DestroyShaderModule(runtime.device, module, nullptr);
            vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
            vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
            vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout,
                                              nullptr);
            destroy_m8_bias_weights(runtime, bias_weights);
            for (Buffer& state : aux.attention_states) destroy_buffer(runtime, state);
            std::array<Buffer*, 20> buffers = {&aux.rope_sin, &aux.rope_cos, &aux.logits,
                &aux.token_history, &aux.token_parameter, &aux.router_logits, &aux.routing, &aux.moe_sum,
                &aux.expert_output, &aux.feed_forward, &aux.gate_up, &aux.projection,
                &aux.context, &aux.qkv, &aux.norm, &aux.quant_temp, &aux.quant_norm,
                &aux.hidden_b, &aux.hidden_a, &aux.dummy};
            for (Buffer* buffer : buffers) destroy_buffer(runtime, *buffer);
            vkfn::DestroyDevice(runtime.device, nullptr);
            vkfn::DestroyInstance(runtime.instance, nullptr);
            FreeLibrary(runtime.loader);
            return 0;
        }

        M8SharedWeights shared = load_m8_shared(model_path, runtime, index);
        caches.resize(index.header.layers);
        uint64_t expert_cache_bytes = 0;
        for (auto& layer : caches) {
            layer.resize(M8_CACHE_SLOTS_PER_LAYER);
            for (auto& slot : layer) {
                slot = std::make_unique<ExpertSlot>(mapped, runtime, dimension, hidden);
                expert_cache_bytes += slot->capacity();
            }
        }
        M8Sets streamed_sets = create_m8_streamed_sets(runtime, resources, index, shared,
                                                        bias_weights,
                                                        caches, aux);
        const M8Result streamed_cold = run_mode(true, streamed_sets);
        if (resident.tokens != streamed_cold.tokens) {
            throw std::runtime_error("Resident and streamed Phi MoE tokens differ");
        }

        // Repack the demand-populated working set into one canonical model-layout arena.
        // This keeps inactive expert regions empty while allowing cache hits and all shared
        // matrices to consume the GPU router result without host queue round trips.
        const uint64_t expert_block_bytes = caches[0][0]->capacity();
        Buffer hot_arena = create_device_buffer(runtime, mapped.size());
        execute([&](VkCommandBuffer command) {
            const auto copy_to = [&](const Buffer& source, uint64_t destination) {
                VkBufferCopy copy{};
                copy.dstOffset = destination;
                copy.size = source.size;
                vkfn::CmdCopyBuffer(command, source.handle, hot_arena.handle, 1, &copy);
            };
            const auto copy_matrix = [&](const GpuMatrix& source, const MatrixIndex& target) {
                copy_to(source.scales, target.scale_offset);
                copy_to(source.weights, target.weight_offset);
            };
            copy_matrix(shared.embedding, index.embedding);
            copy_matrix(shared.lm_head, index.lm_head);
            copy_to(shared.lm_bias, index.lm_bias);
            copy_to(shared.final_gamma, index.final_gamma);
            copy_to(shared.final_beta, index.final_beta);
            for (uint32_t number = 0; number < index.header.layers; ++number) {
                const M8SharedLayer& source_layer = shared.layers[number];
                const M8LayerIndex& target_layer = index.layers[number];
                copy_to(source_layer.attention_gamma, target_layer.attention_gamma);
                copy_to(source_layer.attention_beta, target_layer.attention_beta);
                copy_to(source_layer.feed_forward_gamma, target_layer.feed_forward_gamma);
                copy_to(source_layer.feed_forward_beta, target_layer.feed_forward_beta);
                copy_matrix(source_layer.qkv, target_layer.qkv);
                copy_to(source_layer.qkv_bias, target_layer.qkv_bias);
                copy_matrix(source_layer.output, target_layer.output);
                copy_to(source_layer.output_bias, target_layer.output_bias);
                copy_to(source_layer.router, target_layer.router);
                for (const auto& slot_ptr : caches[number]) {
                    const ExpertSlot& slot = *slot_ptr;
                    if (slot.resident_expert < 0) continue;
                    const ExpertIndex& target = target_layer.experts[
                        static_cast<uint32_t>(slot.resident_expert)];
                    const std::array<const Buffer*, 4> tensors{
                        &slot.gate_scales, &slot.gate_weights,
                        &slot.down_scales, &slot.down_weights};
                    copy_to(*tensors[0], target.gate_up.scale_offset);
                    copy_to(*tensors[1], target.gate_up.weight_offset);
                    copy_to(*tensors[2], target.down.scale_offset);
                    copy_to(*tensors[3], target.down.weight_offset);
                }
            }
        });
        M8Sets hot_sets = create_m8_resident_sets(runtime, resources, index,
            hot_arena, bias_weights, aux);
        for (auto& layer : caches) for (auto& slot : layer) slot->destroy();
        destroy_m8_shared(runtime, shared);
        const M8Result streamed = run_mode(false, hot_sets);
        if (resident.tokens != streamed.tokens) {
            throw std::runtime_error("GPU-routed hot streamed tokens differ");
        }
        float logit_max_error = 0.0f;
        for (size_t value = 0; value < resident.final_logits.size(); ++value) {
            logit_max_error = std::max(logit_max_error,
                std::abs(resident.final_logits[value] - streamed.final_logits[value]));
        }
        const std::vector<uint32_t> generated(streamed.tokens.end() - generation_tokens,
                                              streamed.tokens.end());
        static const std::vector<uint32_t> expected_m8{
            478, 352, 11052, 338, 263, 4482, 29899, 957, 2813, 29892, 4891, 29899,
            12120, 29871, 29941, 29928, 18533, 322, 10272, 3450, 8688, 304, 11157, 4180};
        if (generation_tokens == expected_m8.size() && generated != expected_m8) {
            throw std::runtime_error("Milestone 8 deterministic token sequence changed");
        }
        const uint64_t shared_learned_bytes = shared.bytes;
        uint64_t populated_cache_bytes = 0;
        uint64_t populated_slots = 0;
        for (const auto& layer : caches) for (const auto& slot : layer) {
            if (slot->resident_expert >= 0) {
                populated_cache_bytes += expert_block_bytes;
                ++populated_slots;
            }
        }
        const uint64_t streamed_residency = shared_learned_bytes + populated_cache_bytes;
        const double hit_rate = (streamed_cold.hits + streamed_cold.misses) == 0 ? 0.0 :
            100.0 * streamed_cold.hits /
                static_cast<double>(streamed_cold.hits + streamed_cold.misses);
        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Model parameters: " << index.total_parameters << " (active/token "
                  << (index.total_parameters - index.expert_parameters +
                      index.expert_parameters * index.extension.top_k /
                      index.extension.experts) << ")\n"
                  << std::fixed << std::setprecision(3)
                  << "Fully resident: " << resident.tokens_per_second << " tok/s\n"
                  << "Expert streamed: " << streamed.tokens_per_second << " tok/s\n"
                  << "Streamed learned residency: "
                  << (streamed_residency / (1024.0 * 1024.0)) << " MiB (shared "
                  << (shared_learned_bytes / (1024.0 * 1024.0)) << " + populated expert cache "
                  << (populated_cache_bytes / (1024.0 * 1024.0)) << ")\n"
                  << "Expert-cache capacity: "
                  << (expert_cache_bytes / (1024.0 * 1024.0)) << " MiB; populated slots: "
                  << populated_slots << "/"
                  << (index.header.layers * M8_CACHE_SLOTS_PER_LAYER) << "\n"
                  << "Cache warm-up hits/misses: " << streamed_cold.hits << "/"
                  << streamed_cold.misses
                  << " (" << hit_rate << "% hit)\n"
                  << "Expert transfer/token: "
                  << (streamed_cold.transferred_bytes / static_cast<double>(generation_tokens) /
                      (1024.0 * 1024.0)) << " MiB\n"
                  << "Resident/streamed final-logit max abs delta: " << logit_max_error
                  << "\nGenerated token IDs:";
        for (uint32_t token : generated) std::cout << " " << token;
        std::cout << "\n--- generated text ---\n" << tokenizer.decode(generated)
                  << "\n--- end ---\n"
                  << "RESULT: PASS - Phi MoE resident/streamed deterministic output identical\n";

        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        vkfn::DestroyFence(runtime.device, execute_fence, nullptr);
        vkfn::DestroyQueryPool(runtime.device, dispatch_profile_pool, nullptr);
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
        for (auto& layer : caches) for (auto& slot : layer) slot->destroy();
        destroy_buffer(runtime, hot_arena);
        destroy_m8_shared(runtime, shared);
        destroy_m8_bias_weights(runtime, bias_weights);
        for (Buffer& state : aux.attention_states) destroy_buffer(runtime, state);
        std::array<Buffer*, 20> buffers = {&aux.rope_sin, &aux.rope_cos, &aux.logits,
            &aux.token_history, &aux.token_parameter, &aux.router_logits, &aux.routing, &aux.moe_sum,
            &aux.expert_output, &aux.feed_forward,
            &aux.gate_up, &aux.projection, &aux.context, &aux.qkv, &aux.norm,
            &aux.quant_temp, &aux.quant_norm, &aux.hidden_b, &aux.hidden_a, &aux.dummy};
        for (Buffer* buffer : buffers) destroy_buffer(runtime, *buffer);
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
