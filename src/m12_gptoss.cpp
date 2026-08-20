#define OVLLM_GPTOSS_RUNTIME_ONLY
#include "m8_moe.cpp"

#include <array>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>

constexpr uint32_t GOSS_CACHE_SLOTS = 20;
constexpr uint32_t GOSS_HOT_IMPORTS = 64;
constexpr uint32_t GOSS_STATIC_HOT_IMPORTS = GOSS_HOT_IMPORTS;
// Group-32 signed Q4 experts with BF16 scales, repacked from the checkpoint's
// native MXFP4 weights for direct packed integer-dot execution.
constexpr uint64_t GOSS_Q3_EXPERT_HEADER = 4096ull;
constexpr uint64_t GOSS_Q3_EXPERT_BLOCK = 13236480ull;
constexpr uint64_t GOSS_UPLOAD_CHUNK = 64ull * 1024 * 1024;
constexpr uint64_t GOSS_GPU_CONTROL_STRIDE = 4096;
constexpr uint32_t GOSS_MAX_INDIRECT_COMMANDS = 512;
constexpr uint32_t GOSS_ROUTE_FLOATS = 160;

#pragma pack(push, 1)
struct GossHeader {
    char magic[8];
    uint32_t version;
    uint32_t dimension;
    uint32_t hidden_dimension;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t vocabulary;
    uint32_t experts;
    uint32_t top_k;
    uint32_t head_dimension;
    uint32_t query_dimension;
    uint32_t qkv_dimension;
    uint32_t max_sequence;
    uint32_t sliding_window;
    uint32_t reserved;
    float rope_theta;
    float rms_epsilon;
    float swiglu_limit;
    float rope_scaling_factor;
    float rope_ntk_alpha;
    float rope_ntk_beta;
    uint64_t total_parameters;
    uint64_t active_parameters;
    uint64_t expert_block_bytes;
    uint8_t padding[12];
};
#pragma pack(pop)
static_assert(sizeof(GossHeader) == 128, "Unexpected GPT-OSS header layout");

struct GossMatrix {
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

struct GossLayerIndex {
    uint64_t attention_norm = 0;
    GossMatrix qkv;
    uint64_t qkv_bias = 0;
    uint64_t sinks = 0;
    GossMatrix output;
    uint64_t output_bias = 0;
    uint64_t mlp_norm = 0;
    uint64_t router_weight = 0;
    uint64_t router_bias = 0;
};

struct GossIndex {
    GossHeader header{};
    GossMatrix embedding;
    GossMatrix unembedding;
    uint64_t final_norm = 0;
    std::vector<GossLayerIndex> layers;
    uint64_t shared_bytes = 0;
    uint64_t file_bytes = 0;
};

static GossMatrix goss_matrix(uint64_t& cursor, uint32_t rows, uint32_t columns) {
    GossMatrix result;
    result.rows = rows;
    result.columns = columns;
    result.packed_stride = columns / 4;
    result.scale_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * sizeof(float);
    result.weight_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * columns;
    return result;
}

static GossMatrix goss_matrix_q4(uint64_t& cursor, uint32_t rows, uint32_t columns) {
    GossMatrix result;
    result.rows = rows;
    result.columns = columns;
    result.packed_stride = columns / 8;
    result.scale_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * sizeof(float);
    result.weight_offset = cursor;
    cursor += static_cast<uint64_t>(rows) * result.packed_stride * sizeof(uint32_t);
    return result;
}

static uint64_t goss_vector(uint64_t& cursor, uint64_t count) {
    const uint64_t result = cursor;
    cursor += count * sizeof(float);
    return result;
}

static GossIndex index_goss(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open GPT-OSS runtime model: " + path);
    GossIndex index;
    index.file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    const GossHeader& h = index.header;
    if (!input || std::memcmp(h.magic, "OVGOSS1\0", 8) != 0 || h.version != 1 ||
        h.dimension != 2880 || h.hidden_dimension != 2880 || h.layers != 36 ||
        h.heads != 64 || h.kv_heads != 8 || h.experts != 128 || h.top_k != 4 ||
        h.head_dimension != 64 || h.query_dimension != 4096 || h.qkv_dimension != 5120 ||
        h.max_sequence != 256 || h.expert_block_bytes != 13236480) {
        throw std::runtime_error("Unsupported GPT-OSS-120B runtime layout");
    }
    uint64_t cursor = sizeof(GossHeader);
    index.embedding = goss_matrix(cursor, h.vocabulary, h.dimension);
    index.unembedding = goss_matrix(cursor, h.vocabulary, h.dimension);
    index.final_norm = goss_vector(cursor, h.dimension);
    index.layers.resize(h.layers);
    for (GossLayerIndex& layer : index.layers) {
        layer.attention_norm = goss_vector(cursor, h.dimension);
        layer.qkv = goss_matrix(cursor, h.qkv_dimension, h.dimension);
        layer.qkv_bias = goss_vector(cursor, h.qkv_dimension);
        layer.sinks = goss_vector(cursor, h.heads);
        layer.output = goss_matrix(cursor, h.dimension, h.query_dimension);
        layer.output_bias = goss_vector(cursor, h.dimension);
        layer.mlp_norm = goss_vector(cursor, h.dimension);
        layer.router_weight = goss_vector(cursor,
            static_cast<uint64_t>(h.experts) * h.dimension);
        layer.router_bias = goss_vector(cursor, h.experts);
    }
    cursor = (cursor + 4095u) & ~4095ull;
    index.shared_bytes = cursor;
    const uint64_t expected = cursor + static_cast<uint64_t>(h.layers) * h.experts *
                                      h.expert_block_bytes;
    if (expected != index.file_bytes) {
        throw std::runtime_error("GPT-OSS runtime file does not match indexed layout");
    }
    return index;
}

static GossIndex index_goss_q4_shared(const std::string& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open GPT-OSS Q4 shared weights: " + path);
    GossIndex index;
    index.file_bytes = static_cast<uint64_t>(input.tellg());
    input.seekg(0);
    input.read(reinterpret_cast<char*>(&index.header), sizeof(index.header));
    const GossHeader& h = index.header;
    if (!input || std::memcmp(h.magic, "OVGQ4R8\0", 8) != 0 || h.version != 1 ||
        h.dimension != 2880 || h.layers != 36 || h.experts != 128) {
        throw std::runtime_error("Unsupported GPT-OSS Q4 shared layout");
    }
    uint64_t cursor = sizeof(GossHeader);
    index.embedding = goss_matrix_q4(cursor, h.vocabulary, h.dimension);
    index.unembedding = goss_matrix_q4(cursor, h.vocabulary, h.dimension);
    index.final_norm = goss_vector(cursor, h.dimension);
    index.layers.resize(h.layers);
    for (GossLayerIndex& layer : index.layers) {
        layer.attention_norm = goss_vector(cursor, h.dimension);
        layer.qkv = goss_matrix_q4(cursor, h.qkv_dimension, h.dimension);
        layer.qkv_bias = goss_vector(cursor, h.qkv_dimension);
        layer.sinks = goss_vector(cursor, h.heads);
        layer.output = goss_matrix_q4(cursor, h.dimension, h.query_dimension);
        layer.output_bias = goss_vector(cursor, h.dimension);
        layer.mlp_norm = goss_vector(cursor, h.dimension);
        layer.router_weight = cursor;
        cursor += static_cast<uint64_t>(h.experts) * sizeof(float) +
                  static_cast<uint64_t>(h.experts) * h.dimension;
        layer.router_bias = goss_vector(cursor, h.experts);
    }
    index.shared_bytes = (cursor + 4095u) & ~4095ull;
    if (index.shared_bytes != index.file_bytes)
        throw std::runtime_error("GPT-OSS Q4 shared file size mismatch");
    return index;
}

class GossTokenizer {
public:
    explicit GossTokenizer(const std::string& path) {
        std::ifstream input(path, std::ios::binary);
        char magic[8]{};
        uint32_t version = 0, vocabulary = 0;
        input.read(magic, 8);
        input.read(reinterpret_cast<char*>(&version), 4);
        input.read(reinterpret_cast<char*>(&vocabulary), 4);
        input.read(reinterpret_cast<char*>(&bos_), 4);
        input.read(reinterpret_cast<char*>(&eos_), 4);
        input.read(reinterpret_cast<char*>(&unknown_), 4);
        if (!input || std::memcmp(magic, "OVBPE1\0\0", 8) != 0 || version != 1) {
            throw std::runtime_error("Unsupported GPT-OSS tokenizer");
        }
        pieces_.resize(vocabulary);
        special_.resize(vocabulary);
        for (uint32_t token = 0; token < vocabulary; ++token) {
            uint32_t flags = 0, length = 0;
            input.read(reinterpret_cast<char*>(&flags), 4);
            input.read(reinterpret_cast<char*>(&length), 4);
            pieces_[token].resize(length);
            input.read(pieces_[token].data(), length);
            if (!input) throw std::runtime_error("Truncated GPT-OSS tokenizer");
            special_[token] = (flags & 1u) != 0;
            lookup_[pieces_[token]] = token;
        }
    }

    uint32_t token_id(const std::string& piece) const {
        const auto found = lookup_.find(piece);
        if (found == lookup_.end()) throw std::runtime_error("Missing tokenizer piece: " + piece);
        return found->second;
    }

    std::vector<uint32_t> encode(const std::string& text) const {
        std::vector<uint32_t> result;
        for (const std::string& segment : split_ascii(text)) {
            std::vector<std::string> symbols;
            symbols.reserve(segment.size());
            for (unsigned char byte : segment) symbols.emplace_back(1, static_cast<char>(byte));
            while (symbols.size() > 1) {
                uint32_t best_rank = std::numeric_limits<uint32_t>::max();
                size_t best = symbols.size();
                for (size_t index = 0; index + 1 < symbols.size(); ++index) {
                    const auto found = lookup_.find(symbols[index] + symbols[index + 1]);
                    if (found != lookup_.end() && !special_[found->second] &&
                        found->second < best_rank) {
                        best_rank = found->second;
                        best = index;
                    }
                }
                if (best == symbols.size()) break;
                symbols[best] += symbols[best + 1];
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
            }
            for (const std::string& symbol : symbols) {
                const auto found = lookup_.find(symbol);
                result.push_back(found == lookup_.end() ? unknown_ : found->second);
            }
        }
        return result;
    }

    std::string decode(const std::vector<uint32_t>& tokens) const {
        std::string result;
        for (uint32_t token : tokens) {
            if (token < pieces_.size() && !special_[token]) result += pieces_[token];
        }
        return result;
    }

    size_t vocabulary_size() const { return pieces_.size(); }
    uint32_t eos() const { return eos_; }

private:
    static bool letter(unsigned char c) { return std::isalpha(c) != 0; }
    static bool digit(unsigned char c) { return std::isdigit(c) != 0; }
    static std::vector<std::string> split_ascii(const std::string& text) {
        std::vector<std::string> segments;
        size_t position = 0;
        while (position < text.size()) {
            const size_t begin = position;
            if (text[position] == ' ' && position + 1 < text.size() &&
                text[position + 1] != ' ') ++position;
            if (position < text.size() && letter(static_cast<unsigned char>(text[position]))) {
                while (position < text.size() &&
                       letter(static_cast<unsigned char>(text[position]))) ++position;
                if (position < text.size() && text[position] == '\'' &&
                    position + 1 < text.size()) {
                    size_t end = position + 1;
                    while (end < text.size() && letter(static_cast<unsigned char>(text[end]))) ++end;
                    const std::string suffix = text.substr(position, end - position);
                    if (suffix == "'s" || suffix == "'t" || suffix == "'re" ||
                        suffix == "'ve" || suffix == "'m" || suffix == "'ll" || suffix == "'d")
                        position = end;
                }
            } else if (position < text.size() && digit(static_cast<unsigned char>(text[position]))) {
                uint32_t count = 0;
                while (position < text.size() && digit(static_cast<unsigned char>(text[position])) &&
                       count++ < 3) ++position;
            } else if (position < text.size() && std::isspace(
                           static_cast<unsigned char>(text[position]))) {
                while (position < text.size() && std::isspace(
                           static_cast<unsigned char>(text[position]))) ++position;
            } else {
                while (position < text.size() && !letter(static_cast<unsigned char>(text[position])) &&
                       !digit(static_cast<unsigned char>(text[position])) &&
                       !std::isspace(static_cast<unsigned char>(text[position]))) ++position;
                while (position < text.size() && (text[position] == '\r' ||
                       text[position] == '\n' || text[position] == '/')) ++position;
            }
            if (position == begin) ++position;
            segments.push_back(text.substr(begin, position - begin));
        }
        return segments;
    }

    std::vector<std::string> pieces_;
    std::vector<bool> special_;
    std::unordered_map<std::string, uint32_t> lookup_;
    uint32_t bos_ = 199998, eos_ = 200002, unknown_ = 199999;
};

class GossReadPool {
public:
    GossReadPool() {
        for (uint32_t rank = 0; rank < 4; ++rank)
            workers_[rank] = std::thread([this, rank] { worker(rank); });
    }
    ~GossReadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (std::thread& thread : workers_) thread.join();
    }
    void copy(const std::array<bool, 4>& needed,
              const std::array<const uint8_t*, 4>& sources,
              const std::array<void*, 4>& destinations, size_t bytes) {
        std::unique_lock<std::mutex> lock(mutex_);
        pending_ = 0;
        for (uint32_t rank = 0; rank < 4; ++rank) {
            if (!needed[rank]) continue;
            jobs_[rank] = Job{sources[rank], destinations[rank], bytes, true};
            ++pending_;
        }
        if (pending_ == 0) return;
        ready_.notify_all();
        complete_.wait(lock, [&] { return pending_ == 0; });
    }
private:
    struct Job {
        const uint8_t* source = nullptr;
        void* destination = nullptr;
        size_t bytes = 0;
        bool ready = false;
    };
    void worker(uint32_t rank) {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stop_ || jobs_[rank].ready; });
            if (stop_) return;
            const Job job = jobs_[rank];
            lock.unlock();
            std::memcpy(job.destination, job.source, job.bytes);
            lock.lock();
            jobs_[rank].ready = false;
            if (--pending_ == 0) complete_.notify_one();
        }
    }
    std::array<Job, 4> jobs_{};
    std::array<std::thread, 4> workers_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable complete_;
    uint32_t pending_ = 0;
    bool stop_ = false;
};

class GossPrefetchPool {
public:
    explicit GossPrefetchPool(uint32_t layers) : pending_(layers, 0) {
        for (std::thread& thread : workers_)
            thread = std::thread([this] { worker(); });
    }
    ~GossPrefetchPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        ready_.notify_all();
        for (std::thread& thread : workers_) thread.join();
    }
    void schedule(uint32_t layer, const uint8_t* source, void* destination, size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push_back(Task{layer, source, destination, bytes});
            ++pending_[layer];
        }
        ready_.notify_one();
    }
    void wait(uint32_t layer) {
        std::unique_lock<std::mutex> lock(mutex_);
        complete_.wait(lock, [&] { return pending_[layer] == 0; });
    }
private:
    struct Task {
        uint32_t layer;
        const uint8_t* source;
        void* destination;
        size_t bytes;
    };
    void worker() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return stop_ || !tasks_.empty(); });
            if (stop_ && tasks_.empty()) return;
            const Task task = tasks_.front();
            tasks_.pop_front();
            lock.unlock();
            std::memcpy(task.destination, task.source, task.bytes);
            lock.lock();
            if (--pending_[task.layer] == 0) complete_.notify_all();
        }
    }
    std::array<std::thread, 8> workers_;
    std::vector<uint32_t> pending_;
    std::deque<Task> tasks_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable complete_;
    bool stop_ = false;
};

// Host-visible Vulkan storage used as the second expert-cache tier.  Unlike
// create_buffer(), this deliberately does not touch every page at allocation:
// slots become resident only when populated from the mapped model.
static Buffer create_goss_host_cache_buffer(const Runtime& runtime, VkDeviceSize size) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    allocation.pNext = &address_flags;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_memory_type(
        runtime, requirements.memoryTypeBits, buffer.coherent);
    VK_CHECK(vkfn::AllocateMemory(runtime.device, &allocation, nullptr, &buffer.memory));
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes,
                                        active_vulkan_buffer_bytes);
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle, buffer.memory, 0));
    VK_CHECK(vkfn::MapMemory(runtime.device, buffer.memory, 0, VK_WHOLE_SIZE, 0,
                            &buffer.mapped));
    std::memset(buffer.mapped, 0, static_cast<size_t>(size));
    return buffer;
}

struct GossImportedRange {
    Buffer buffer;
    uint64_t file_offset = 0;
};

struct GossHotImport {
    int32_t expert = -1;
    uint64_t last_used = 0;
    GossImportedRange imported;
};

static GossImportedRange import_goss_model_range(const Runtime& runtime,
                                                  const MappedModelFile& mapped,
                                                  uint64_t begin, uint64_t end) {
    if (!external_host_memory_enabled)
        throw std::runtime_error("AMD Vulkan driver lacks VK_EXT_external_memory_host");
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &host_properties;
    vkfn::GetPhysicalDeviceProperties2(runtime.physical, &properties);
    const uint64_t alignment = host_properties.minImportedHostPointerAlignment;
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::runtime_error("Invalid Vulkan external host pointer alignment");
    const uint64_t aligned_begin = begin & ~(alignment - 1);
    const uint64_t aligned_end = (end + alignment - 1) & ~(alignment - 1);
    const void* pointer = mapped.data() + aligned_begin;

    VkExternalMemoryHandleTypeFlagBits handle_type =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT;
    VkMemoryHostPointerPropertiesEXT pointer_properties{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult query = vkfn::GetMemoryHostPointerPropertiesEXT(runtime.device, handle_type,
                                                              pointer, &pointer_properties);
    if (query != VK_SUCCESS) {
        handle_type = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        pointer_properties = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        query = vkfn::GetMemoryHostPointerPropertiesEXT(runtime.device, handle_type,
                                                         pointer, &pointer_properties);
    }
    VK_CHECK(query);

    GossImportedRange result;
    result.file_offset = aligned_begin;
    Buffer& buffer = result.buffer;
    buffer.size = aligned_end - aligned_begin;
    buffer.coherent = true;
    VkExternalMemoryBufferCreateInfo external{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = handle_type;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &external;
    info.size = buffer.size;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    const uint32_t allowed = requirements.memoryTypeBits & pointer_properties.memoryTypeBits;
    if (allowed == 0) throw std::runtime_error("No memory type for imported model mapping");
    bool coherent = false;
    VkImportMemoryHostPointerInfoEXT import{
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import.handleType = handle_type;
    import.pHostPointer = const_cast<void*>(pointer);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    address_flags.pNext = &import;
    allocation.pNext = &address_flags;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_memory_type(runtime, allowed, coherent);
    VK_CHECK(vkfn::AllocateMemory(runtime.device, &allocation, nullptr, &buffer.memory));
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes,
                                        active_vulkan_buffer_bytes);
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle, buffer.memory, 0));
    return result;
}

static Buffer upload_goss_shared(const Runtime& runtime, const MappedModelFile& mapped,
                                 uint64_t bytes) {
    Buffer device = create_device_buffer(runtime, bytes);
    Buffer staging = create_buffer(runtime, std::min<uint64_t>(GOSS_UPLOAD_CHUNK, bytes));
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = runtime.queue_family;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &pool));
    for (uint64_t offset = 0; offset < bytes; offset += staging.size) {
        const VkDeviceSize count = std::min<uint64_t>(staging.size, bytes - offset);
        std::memcpy(staging.mapped, mapped.data() + offset, static_cast<size_t>(count));
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
        copy.srcOffset = 0;
        copy.dstOffset = offset;
        copy.size = count;
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

static uint64_t goss_buffer_address(const Runtime& runtime, const Buffer& buffer) {
    VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    info.buffer = buffer.handle;
    return vkfn::GetBufferDeviceAddress(runtime.device, &info);
}

struct GossAuxiliary {
    Buffer dummy;
    Buffer token_parameter;
    Buffer hidden_a;
    Buffer hidden_b;
    Buffer norm;
    Buffer qkv;
    Buffer context;
    Buffer projection;
    Buffer feed_forward;
    Buffer moe_sum;
    Buffer routing;
    Buffer prediction_routing;
    Buffer slot_maps;
    Buffer cache_control;
    Buffer indirect_commands;
    Buffer speculative_norm;
    Buffer speculative_hidden_b;
    Buffer quantized_expert_input;
    Buffer logits;
    Buffer rope_cos;
    Buffer rope_sin;
    std::vector<Buffer> attention_states;
};

struct GossSlot {
    int32_t expert = -1;
    uint64_t last_used = 0;
};

struct GossLayerCache {
    Buffer arena;
    std::array<GossSlot, GOSS_CACHE_SLOTS> slots;
};

struct GossExpertSets {
    VkDescriptorSet gate = VK_NULL_HANDLE;
    VkDescriptorSet down = VK_NULL_HANDLE;
};

struct GossLayerSets {
    VkDescriptorSet attention_norm = VK_NULL_HANDLE;
    VkDescriptorSet qkv = VK_NULL_HANDLE;
    VkDescriptorSet rope = VK_NULL_HANDLE;
    VkDescriptorSet attention = VK_NULL_HANDLE;
    VkDescriptorSet output = VK_NULL_HANDLE;
    VkDescriptorSet attention_residual = VK_NULL_HANDLE;
    VkDescriptorSet mlp_norm = VK_NULL_HANDLE;
    VkDescriptorSet router = VK_NULL_HANDLE;
    VkDescriptorSet prediction_norm = VK_NULL_HANDLE;
    VkDescriptorSet prediction_router = VK_NULL_HANDLE;
    std::array<GossExpertSets, GOSS_CACHE_SLOTS> experts;
    GossExpertSets fused_experts;
    GossExpertSets speculative_experts;
    GossExpertSets gpu_experts;
    VkDescriptorSet cache_resolve = VK_NULL_HANDLE;
    VkDescriptorSet cache_copy = VK_NULL_HANDLE;
    VkDescriptorSet mlp_residual = VK_NULL_HANDLE;
};

struct GossSets {
    VkDescriptorSet embedding = VK_NULL_HANDLE;
    VkDescriptorSet final_norm = VK_NULL_HANDLE;
    VkDescriptorSet logits = VK_NULL_HANDLE;
    VkDescriptorSet greedy = VK_NULL_HANDLE;
    VkDescriptorSet quantize_norm = VK_NULL_HANDLE;
    VkDescriptorSet quantize_feed_forward = VK_NULL_HANDLE;
    std::vector<GossLayerSets> layers;
};

static DescriptorRange goss_matrix_block(const Buffer& arena, const GossMatrix& matrix) {
    return arena_range(arena, matrix.scale_offset, matrix.byte_size());
}

static VkDescriptorSet create_goss_dynamic_set(
    const Runtime& runtime, const ComputeResources& resources,
    const std::array<DescriptorRange, 4>& core,
    const std::array<DescriptorRange, 8>& hosts,
    const DescriptorRange& slot_map) {
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = resources.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &resources.descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkfn::AllocateDescriptorSets(runtime.device, &allocation, &set));
    VkDescriptorBufferInfo infos[13]{};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t binding = 0; binding < 4; ++binding) {
        infos[binding].buffer = core[binding].buffer;
        infos[binding].offset = core[binding].offset;
        infos[binding].range = core[binding].range;
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    for (uint32_t chunk = 0; chunk < 8; ++chunk) {
        infos[4 + chunk].buffer = hosts[chunk].buffer;
        infos[4 + chunk].offset = hosts[chunk].offset;
        infos[4 + chunk].range = hosts[chunk].range;
    }
    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = set;
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 8;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[4].pBufferInfo = &infos[4];
    infos[12].buffer = slot_map.buffer;
    infos[12].offset = slot_map.offset;
    infos[12].range = slot_map.range;
    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = set;
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[5].pBufferInfo = &infos[12];
    vkfn::UpdateDescriptorSets(runtime.device, 6, writes, 0, nullptr);
    return set;
}

static VkDescriptorSet create_goss_five_range_set(
    const Runtime& runtime, const ComputeResources& resources,
    const std::array<DescriptorRange, 5>& ranges) {
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = resources.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &resources.descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkfn::AllocateDescriptorSets(runtime.device, &allocation, &set));
    VkDescriptorBufferInfo infos[5]{};
    VkWriteDescriptorSet writes[5]{};
    for (uint32_t binding = 0; binding < 5; ++binding) {
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
    vkfn::UpdateDescriptorSets(runtime.device, 5, writes, 0, nullptr);
    return set;
}

static GossSets create_goss_sets(const Runtime& runtime, const ComputeResources& resources,
                                 const GossIndex& index, const Buffer& shared,
                                 std::vector<GossLayerCache>& caches, GossAuxiliary& aux) {
    const DescriptorRange dummy = whole(aux.dummy);
    GossSets sets;
    sets.embedding = create_range_set(runtime, resources,
        {goss_matrix_block(shared, index.embedding), whole(aux.token_parameter),
         whole(aux.hidden_a), dummy});
    sets.final_norm = create_range_set(runtime, resources,
        {whole(aux.hidden_a), arena_range(shared, index.final_norm,
         index.header.dimension * sizeof(float)), whole(aux.norm), dummy});
    sets.logits = create_range_set(runtime, resources,
        {whole(aux.norm), goss_matrix_block(shared, index.unembedding), dummy,
         whole(aux.logits)});
    sets.greedy = create_range_set(runtime, resources,
        {whole(aux.logits), whole(aux.token_parameter), whole(aux.feed_forward), dummy});
    sets.quantize_norm = create_range_set(runtime, resources,
        {whole(aux.norm), whole(aux.quantized_expert_input), dummy, dummy});
    sets.quantize_feed_forward = create_range_set(runtime, resources,
        {whole(aux.feed_forward), whole(aux.quantized_expert_input), dummy, dummy});
    sets.layers.resize(index.header.layers);
    for (uint32_t number = 0; number < index.header.layers; ++number) {
        const GossLayerIndex& source = index.layers[number];
        GossLayerSets& set = sets.layers[number];
        set.attention_norm = create_range_set(runtime, resources,
            {whole(aux.hidden_a), arena_range(shared, source.attention_norm,
             index.header.dimension * sizeof(float)), whole(aux.norm), dummy});
        set.qkv = create_range_set(runtime, resources,
            {whole(aux.norm), goss_matrix_block(shared, source.qkv),
             arena_range(shared, source.qkv_bias,
                         index.header.qkv_dimension * sizeof(float)), whole(aux.qkv)});
        set.rope = create_range_set(runtime, resources,
            {whole(aux.qkv), whole(aux.rope_cos), whole(aux.rope_sin),
             whole(aux.attention_states[number])});
        set.attention = create_range_set(runtime, resources,
            {whole(aux.attention_states[number]),
             arena_range(shared, source.sinks, index.header.heads * sizeof(float)),
             whole(aux.context), dummy});
        set.output = create_goss_five_range_set(runtime, resources,
            {whole(aux.context), goss_matrix_block(shared, source.output),
             arena_range(shared, source.output_bias,
                         index.header.dimension * sizeof(float)), whole(aux.hidden_b),
             whole(aux.hidden_a)});
        set.attention_residual = create_range_set(runtime, resources,
            {whole(aux.hidden_a), whole(aux.projection), whole(aux.hidden_b), dummy});
        set.mlp_norm = create_range_set(runtime, resources,
            {whole(aux.hidden_b), arena_range(shared, source.mlp_norm,
             index.header.dimension * sizeof(float)), whole(aux.norm), dummy});
        const DescriptorRange layer_routing = arena_range(aux.routing,
            static_cast<uint64_t>(number) * GOSS_ROUTE_FLOATS * sizeof(float),
            GOSS_ROUTE_FLOATS * sizeof(float));
        set.router = create_range_set(runtime, resources,
            {whole(aux.norm), arena_range(shared, source.router_weight,
             static_cast<uint64_t>(index.header.experts) * sizeof(float) +
             static_cast<uint64_t>(index.header.experts) * index.header.dimension),
             arena_range(shared, source.router_bias,
              index.header.experts * sizeof(float)), layer_routing});
        const DescriptorRange prediction_routing = arena_range(aux.prediction_routing,
            static_cast<uint64_t>(number) * GOSS_ROUTE_FLOATS * sizeof(float),
            GOSS_ROUTE_FLOATS * sizeof(float));
        set.prediction_norm = create_range_set(runtime, resources,
            {whole(aux.hidden_a), arena_range(shared, source.mlp_norm,
             index.header.dimension * sizeof(float)), whole(aux.speculative_norm), dummy});
        set.prediction_router = create_range_set(runtime, resources,
            {whole(aux.speculative_norm), arena_range(shared, source.router_weight,
             static_cast<uint64_t>(index.header.experts) * sizeof(float) +
             static_cast<uint64_t>(index.header.experts) * index.header.dimension),
             arena_range(shared, source.router_bias,
             index.header.experts * sizeof(float)), prediction_routing});
        set.mlp_residual = create_range_set(runtime, resources,
            {whole(aux.hidden_b), whole(aux.moe_sum), whole(aux.hidden_a), dummy});
        for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
            const DescriptorRange expert = arena_range(caches[number].arena,
                static_cast<uint64_t>(slot) * GOSS_Q3_EXPERT_BLOCK,
                GOSS_Q3_EXPERT_BLOCK);
            set.experts[slot].gate = create_range_set(runtime, resources,
                {whole(aux.norm), expert, layer_routing, whole(aux.feed_forward)});
            set.experts[slot].down = create_range_set(runtime, resources,
                {whole(aux.feed_forward), expert, layer_routing, whole(aux.moe_sum)});
        }
        const DescriptorRange expert_arena = whole(caches[number].arena);
        set.fused_experts.gate = create_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), expert_arena, layer_routing,
             whole(aux.feed_forward)});
        set.fused_experts.down = create_goss_five_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), expert_arena, layer_routing,
             whole(aux.hidden_a), whole(aux.hidden_b)});
        const DescriptorRange address_table = arena_range(aux.slot_maps,
            static_cast<uint64_t>(number) * (index.header.experts + 4) * sizeof(uint64_t),
            (index.header.experts + 4) * sizeof(uint64_t));
        set.speculative_experts.gate = create_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), address_table, layer_routing,
             whole(aux.feed_forward)});
        set.speculative_experts.down = create_goss_five_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), address_table, layer_routing,
             whole(aux.hidden_a), whole(aux.hidden_b)});
        const DescriptorRange cache_control = arena_range(aux.cache_control,
            static_cast<uint64_t>(number) * GOSS_GPU_CONTROL_STRIDE,
            GOSS_GPU_CONTROL_STRIDE);
        set.cache_resolve = create_range_set(runtime, resources,
            {layer_routing, cache_control, whole(aux.indirect_commands), dummy});
        set.cache_copy = create_range_set(runtime, resources,
            {whole(caches[number].arena), cache_control, dummy, dummy});
        set.gpu_experts.gate = create_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), cache_control, layer_routing,
             whole(aux.feed_forward)});
        set.gpu_experts.down = create_goss_five_range_set(runtime, resources,
            {whole(aux.quantized_expert_input), cache_control, layer_routing,
             whole(aux.hidden_a), whole(aux.hidden_b)});
    }
    return sets;
}

struct GossPipelines {
    VkPipeline embedding = VK_NULL_HANDLE;
    VkPipeline rmsnorm = VK_NULL_HANDLE;
    VkPipeline qgemv = VK_NULL_HANDLE;
    VkPipeline qgemv_residual = VK_NULL_HANDLE;
    VkPipeline rope = VK_NULL_HANDLE;
    VkPipeline attention = VK_NULL_HANDLE;
    VkPipeline add = VK_NULL_HANDLE;
    VkPipeline router = VK_NULL_HANDLE;
    VkPipeline router_select = VK_NULL_HANDLE;
    VkPipeline expert_gate = VK_NULL_HANDLE;
    VkPipeline expert_down = VK_NULL_HANDLE;
    VkPipeline expert_gate_dynamic = VK_NULL_HANDLE;
    VkPipeline expert_down_dynamic = VK_NULL_HANDLE;
    VkPipeline expert_gate_speculative = VK_NULL_HANDLE;
    VkPipeline expert_down_speculative = VK_NULL_HANDLE;
    VkPipeline expert_gate_table = VK_NULL_HANDLE;
    VkPipeline expert_down_table = VK_NULL_HANDLE;
    VkPipeline quantize_expert = VK_NULL_HANDLE;
    VkPipeline cache_resolve = VK_NULL_HANDLE;
    VkPipeline cache_copy = VK_NULL_HANDLE;
    VkPipeline cache_copy_done = VK_NULL_HANDLE;
    VkPipeline cache_resolve_stop = VK_NULL_HANDLE;
    VkPipeline cache_copy_finite = VK_NULL_HANDLE;
    VkPipeline greedy = VK_NULL_HANDLE;
};

struct GossEmbeddingPush { uint32_t rows, columns, packed_stride, unused; };
struct GossLinearPush { uint32_t has_bias, columns, inner, packed_stride; };
struct GossRouterPush { uint32_t dimension, experts, top_k, unused; };
struct GossExpertPush { uint32_t rank, unused0, unused1, unused2; };
struct GossFusedExpertPush { uint32_t slots[4]; };
struct GossGreedyPush { uint32_t count, unused0, unused1, unused2; };
struct GossQuantizePush { uint32_t count, ranks, unused0, unused1; };
struct GossResolvePush { uint32_t copy_command, stop_command, command_limit, unused; };

static void goss_dispatch_indirect(VkCommandBuffer command_buffer,
                                   const ComputeResources& resources,
                                   VkPipeline pipeline, VkDescriptorSet descriptor_set,
                                   const void* push_data, const Buffer& commands,
                                   uint32_t command_index) {
    vkfn::CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkfn::CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        resources.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    vkfn::CmdPushConstants(command_buffer, resources.pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push_data);
    vkfn::CmdDispatchIndirect(command_buffer, commands.handle,
        static_cast<VkDeviceSize>(command_index) * sizeof(VkDispatchIndirectCommand));
}

struct GossRunResult {
    std::vector<uint32_t> generated;
    double seconds = 0.0;
    double tokens_per_second = 0.0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t transferred_bytes = 0;
    uint64_t peak_populated_slots = 0;
    float lm_cpu_gpu_error = 0.0f;
};

int main(int argc, char** argv) {
    try {
        const std::string model_directory = argc > 1 ? argv[1] :
            "A:\\amd-vulkan-llm-m1\\models\\gpt-oss-120b\\runtime-q8-mxfp4";
        const std::string shader_directory = argc > 2 ? argv[2] :
            "A:\\amd-vulkan-llm-m1\\build";
        const std::string prompt = argc > 3 ? argv[3] :
            "In one short sentence, explain what Vulkan is.";
        const uint32_t generation_limit = argc > 4 ?
            static_cast<uint32_t>(std::stoul(argv[4])) : 32;
        const std::string model_path = model_directory + "\\model.ovm";
        const std::string shared_q4_path = model_directory + "\\shared-q4-router-q8.ovs";
        const std::string expert_q3_path = model_directory + "\\experts-q4g64t.ovx";

        const GossIndex index = index_goss(model_path);
        MappedModelFile mapped(model_path);
        const GossIndex shared_index = index_goss_q4_shared(shared_q4_path);
        MappedModelFile shared_mapped(shared_q4_path);
        MappedModelFile expert_storage(expert_q3_path);
        const MappedModelFile& expert_mapped = expert_storage;
        if (expert_mapped.size() != GOSS_Q3_EXPERT_HEADER +
                static_cast<uint64_t>(index.header.layers) * index.header.experts *
                GOSS_Q3_EXPERT_BLOCK ||
            std::memcmp(expert_mapped.data(), "OVGQ4T1\0", 8) != 0)
            throw std::runtime_error("Invalid GPT-OSS packed-Q4 expert source");
        GossTokenizer tokenizer(model_directory + "\\tokenizer.ovb");
        if (tokenizer.vocabulary_size() != index.header.vocabulary) {
            throw std::runtime_error("GPT-OSS tokenizer/model vocabulary mismatch");
        }
        Runtime runtime = create_runtime();
        Buffer shared = upload_goss_shared(runtime, shared_mapped, shared_index.shared_bytes);

        GossAuxiliary aux;
        aux.dummy = create_buffer(runtime, index.header.vocabulary * sizeof(float));
        aux.token_parameter = create_buffer(runtime, sizeof(uint32_t));
        aux.hidden_a = create_device_buffer(runtime, index.header.dimension * sizeof(float));
        aux.hidden_b = create_device_buffer(runtime, index.header.dimension * sizeof(float));
        aux.norm = create_device_buffer(runtime, index.header.dimension * sizeof(float));
        aux.qkv = create_device_buffer(runtime, index.header.qkv_dimension * sizeof(float));
        aux.context = create_device_buffer(runtime, index.header.query_dimension * sizeof(float));
        aux.projection = create_device_buffer(runtime, index.header.dimension * sizeof(float));
        aux.feed_forward = create_device_buffer(runtime,
            static_cast<uint64_t>(index.header.top_k) * index.header.hidden_dimension *
            sizeof(float));
        aux.moe_sum = create_device_buffer(runtime, index.header.dimension * sizeof(float));
        aux.routing = create_buffer(runtime, static_cast<uint64_t>(index.header.layers) *
            GOSS_ROUTE_FLOATS * sizeof(float));
        aux.prediction_routing = create_buffer(runtime,
            static_cast<uint64_t>(index.header.layers) * GOSS_ROUTE_FLOATS * sizeof(float));
        aux.slot_maps = create_buffer(runtime, static_cast<uint64_t>(index.header.layers) *
            (index.header.experts + 4) * sizeof(uint64_t));
        aux.cache_control = create_buffer(runtime,
            static_cast<uint64_t>(index.header.layers) * GOSS_GPU_CONTROL_STRIDE);
        aux.indirect_commands = create_buffer(runtime,
            static_cast<uint64_t>(GOSS_MAX_INDIRECT_COMMANDS) *
            sizeof(VkDispatchIndirectCommand));
        aux.speculative_norm = create_device_buffer(
            runtime, index.header.dimension * sizeof(float));
        aux.speculative_hidden_b = create_device_buffer(
            runtime, index.header.dimension * sizeof(float));
        aux.quantized_expert_input = create_device_buffer(runtime,
            static_cast<uint64_t>(index.header.top_k) *
            (index.header.hidden_dimension / 4 + 1) * sizeof(uint32_t));
        aux.logits = create_buffer(runtime, index.header.vocabulary * sizeof(float));

        std::vector<float> rope_cos(static_cast<size_t>(index.header.max_sequence) *
                                    index.header.head_dimension / 2);
        std::vector<float> rope_sin(rope_cos.size());
        const float half = index.header.head_dimension / 2.0f;
        const float low = half * std::log(4096.0f /
            (index.header.rope_ntk_beta * 2.0f * 3.14159265358979323846f)) /
            std::log(index.header.rope_theta);
        const float high = half * std::log(4096.0f /
            (index.header.rope_ntk_alpha * 2.0f * 3.14159265358979323846f)) /
            std::log(index.header.rope_theta);
        const float concentration = 0.1f * std::log(index.header.rope_scaling_factor) + 1.0f;
        for (uint32_t position = 0; position < index.header.max_sequence; ++position) {
            for (uint32_t frequency = 0; frequency < index.header.head_dimension / 2;
                 ++frequency) {
                const float base_frequency = std::pow(index.header.rope_theta,
                    2.0f * frequency / index.header.head_dimension);
                const float ramp = std::clamp((frequency - low) / (high - low), 0.0f, 1.0f);
                const float mask = 1.0f - ramp;
                const float inverse = (1.0f / (index.header.rope_scaling_factor *
                                               base_frequency)) * (1.0f - mask) +
                                      (1.0f / base_frequency) * mask;
                const float angle = position * inverse;
                const size_t offset = static_cast<size_t>(position) *
                                      (index.header.head_dimension / 2) + frequency;
                rope_cos[offset] = std::cos(angle) * concentration;
                rope_sin[offset] = std::sin(angle) * concentration;
            }
        }
        DeviceUploader rope_uploader(runtime);
        aux.rope_cos = rope_uploader.upload(rope_cos);
        aux.rope_sin = rope_uploader.upload(rope_sin);
        rope_uploader.finish();
        aux.attention_states.resize(index.header.layers);
        for (Buffer& state : aux.attention_states) {
            constexpr uint32_t cache_sequence = 256;
            state = create_device_buffer(runtime,
                static_cast<uint64_t>(index.header.query_dimension +
                    2ull * cache_sequence *
                    index.header.kv_heads * index.header.head_dimension) * sizeof(float));
        }

        std::vector<GossLayerCache> caches(index.header.layers);
        for (GossLayerCache& cache : caches) {
            cache.arena = create_device_buffer(runtime,
                static_cast<uint64_t>(GOSS_CACHE_SLOTS) * GOSS_Q3_EXPERT_BLOCK);
        }
        std::array<Buffer, 4> staging;
        for (Buffer& buffer : staging)
            buffer = create_goss_host_cache_buffer(runtime, GOSS_Q3_EXPERT_BLOCK);
        Buffer zero_expert = create_goss_host_cache_buffer(runtime, GOSS_Q3_EXPERT_BLOCK);
        const uint64_t zero_expert_address = goss_buffer_address(runtime, zero_expert);
        if (!aux.cache_control.coherent)
            throw std::runtime_error(
                "GPU/CPU cooperative cache requires host-coherent Vulkan control memory");
        GossPrefetchPool acquisition_pool(index.header.layers);
        std::vector<std::array<GossHotImport, GOSS_HOT_IMPORTS>> hot_imports(
            index.header.layers);

        ComputeResources resources = create_compute_resources(runtime, 3000);
        const auto shader = [&](const char* name) {
            return shader_directory + "\\" + name + ".comp.spv";
        };
        GossPipelines pipelines;
        pipelines.embedding = create_pipeline(runtime, resources,
            shader("gptoss_embedding_q4"));
        pipelines.rmsnorm = create_pipeline(runtime, resources, shader("rmsnorm"));
        pipelines.qgemv = create_pipeline(runtime, resources, shader("gptoss_qgemv_q4"));
        pipelines.qgemv_residual = create_pipeline(runtime, resources,
            shader("gptoss_qgemv_q4_residual"));
        pipelines.rope = create_pipeline(runtime, resources, shader("rope_cache"));
        pipelines.attention = create_pipeline(runtime, resources, shader("gptoss_attention"));
        pipelines.add = create_pipeline(runtime, resources, shader("add"));
        pipelines.router = create_pipeline(runtime, resources,
            shader("gptoss_router_q8_parallel"));
        pipelines.router_select = create_pipeline(runtime, resources,
            shader("gptoss_router_top4_select"));
        pipelines.expert_gate = create_pipeline(runtime, resources,
            shader("gptoss_expert_gate_q4t_fused"));
        pipelines.expert_down = create_pipeline(runtime, resources,
            shader("gptoss_expert_down_q4t_fused"));
        pipelines.expert_gate_speculative = create_pipeline(runtime, resources,
            shader("gptoss_expert_gate_q4t_bda"));
        pipelines.expert_down_speculative = create_pipeline(runtime, resources,
            shader("gptoss_expert_down_q4t_bda"));
        pipelines.expert_gate_table = create_pipeline(runtime, resources,
            shader("gptoss_expert_gate_q4t_table"));
        pipelines.expert_down_table = create_pipeline(runtime, resources,
            shader("gptoss_expert_down_q4t_table"));
        pipelines.quantize_expert = create_pipeline(runtime, resources,
            shader("gptoss_quantize_q8"));
        pipelines.cache_resolve = create_pipeline(runtime, resources,
            shader("gptoss_resolve_hybrid"));
        pipelines.cache_copy = create_pipeline(runtime, resources,
            shader("gptoss_cache_copy"));
        pipelines.cache_copy_done = create_pipeline(runtime, resources,
            shader("gptoss_cache_copy_done"));
        pipelines.cache_resolve_stop = create_pipeline(runtime, resources,
            shader("gptoss_resolve_stop"));
        pipelines.cache_copy_finite = create_pipeline(runtime, resources,
            shader("gptoss_cache_copy_finite"));
        pipelines.greedy = create_pipeline(runtime, resources,
            shader("gptoss_greedy_argmax"));
        GossSets sets = create_goss_sets(runtime, resources, shared_index, shared, caches, aux);

        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.queueFamilyIndex = runtime.queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &command_pool));
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = command_pool;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command));
        std::vector<VkCommandBuffer> layer_commands(index.header.layers * 2 + 2);
        VkCommandBufferAllocateInfo layer_allocation{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        layer_allocation.commandPool = command_pool;
        layer_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        layer_allocation.commandBufferCount =
            static_cast<uint32_t>(layer_commands.size());
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &layer_allocation,
                                               layer_commands.data()));
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VkFence fence = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateFence(runtime.device, &fence_info, nullptr, &fence));
        VkSemaphoreTypeCreateInfo timeline_type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_type.initialValue = 0;
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_type;
        VkSemaphore timeline = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateSemaphore(runtime.device, &semaphore_info, nullptr, &timeline));
        VkSemaphore gpu_layer_timeline = VK_NULL_HANDLE;
        VkSemaphore host_layer_timeline = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateSemaphore(runtime.device, &semaphore_info, nullptr,
                                       &gpu_layer_timeline));
        VK_CHECK(vkfn::CreateSemaphore(runtime.device, &semaphore_info, nullptr,
                                       &host_layer_timeline));
        VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 256;
        VK_CHECK(vkfn::CreateQueryPool(runtime.device, &query_info, nullptr,
                                       &dispatch_profile_pool));
        std::array<double, 12> profile_ms{};
        double profile_span_ms = 0.0;
        uint64_t profile_submissions = 0;
        uint64_t timeline_value = 0;
        uint64_t layer_timeline_base = 0;
        double gpu_submission_seconds = 0.0;
        const auto execute = [&](const auto& record) {
            const auto execute_start = std::chrono::steady_clock::now();
            VK_CHECK(vkfn::ResetCommandBuffer(command, 0));
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkfn::BeginCommandBuffer(command, &begin));
            if (dispatch_profile_enabled) {
                dispatch_profile_next = 0;
                dispatch_profile_pipelines.clear();
                vkfn::CmdResetQueryPool(command, dispatch_profile_pool, 0, 256);
            }
            VkMemoryBarrier upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &upload, 0, nullptr, 0, nullptr);
            record(command);
            flush_buffer(runtime, aux.indirect_commands);
            flush_buffer(runtime, aux.cache_control);
            VkMemoryBarrier completion{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            completion.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            completion.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &completion, 0, nullptr, 0, nullptr);
            VK_CHECK(vkfn::EndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            const uint64_t signal_value = ++timeline_value;
            VkTimelineSemaphoreSubmitInfo timeline_submit{
                VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline_submit.signalSemaphoreValueCount = 1;
            timeline_submit.pSignalSemaphoreValues = &signal_value;
            submit.pNext = &timeline_submit;
            submit.signalSemaphoreCount = 1;
            submit.pSignalSemaphores = &timeline;
            VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
            uint64_t completed = 0;
            do {
                VK_CHECK(vkfn::GetSemaphoreCounterValue(runtime.device, timeline, &completed));
            } while (completed < signal_value);
            if (dispatch_profile_enabled && dispatch_profile_next >= 2u) {
                std::vector<uint64_t> stamps(dispatch_profile_next);
                VK_CHECK(vkfn::GetQueryPoolResults(runtime.device, dispatch_profile_pool,
                    0, dispatch_profile_next, stamps.size() * sizeof(uint64_t),
                    stamps.data(), sizeof(uint64_t),
                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                for (size_t item = 0; item < dispatch_profile_pipelines.size(); ++item) {
                    const VkPipeline pipeline = dispatch_profile_pipelines[item];
                    size_t category = 11;
                    if (pipeline == pipelines.embedding) category = 0;
                    else if (pipeline == pipelines.rmsnorm) category = 1;
                    else if (pipeline == pipelines.qgemv ||
                             pipeline == pipelines.qgemv_residual) category = 2;
                    else if (pipeline == pipelines.rope) category = 3;
                    else if (pipeline == pipelines.attention) category = 4;
                    else if (pipeline == pipelines.router ||
                             pipeline == pipelines.router_select) category = 5;
                    else if (pipeline == pipelines.quantize_expert) category = 6;
                    else if (pipeline == pipelines.expert_gate) category = 7;
                    else if (pipeline == pipelines.expert_down) category = 8;
                    else if (pipeline == pipelines.add) category = 9;
                    else if (pipeline == pipelines.greedy) category = 10;
                    profile_ms[category] +=
                        (stamps[item * 2u + 1u] - stamps[item * 2u]) *
                        runtime.properties.limits.timestampPeriod * 1e-6;
                }
                profile_span_ms += (stamps.back() - stamps.front()) *
                    runtime.properties.limits.timestampPeriod * 1e-6;
                ++profile_submissions;
            }
            gpu_submission_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - execute_start).count();
        };

        const RmsPush norm_push{1, index.header.dimension, index.header.rms_epsilon, 0};
        const AddPush add_push{index.header.dimension, 0, 0, 0};
        const GossRouterPush router_push{index.header.dimension, index.header.experts,
                                         index.header.top_k, 0};
        const auto qgemv = [&](VkCommandBuffer cb, VkDescriptorSet set,
                               const GossMatrix& matrix, bool bias) {
            const GossLinearPush push{bias ? 1u : 0u, matrix.rows, matrix.columns,
                                      matrix.packed_stride};
            dispatch(cb, resources, pipelines.qgemv, set, &push,
                     (matrix.rows + 3u) / 4u, 1);
        };
        const auto record_shared_layer = [&](VkCommandBuffer cb, uint32_t number,
                                             uint32_t position) {
            const GossLayerIndex& layer = shared_index.layers[number];
            const GossLayerSets& set = sets.layers[number];
            dispatch(cb, resources, pipelines.rmsnorm, set.attention_norm,
                     &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            qgemv(cb, set.qkv, layer.qkv, true);
            compute_buffer_barrier(cb, aux.qkv);
            const CachePush rope_push{position, index.header.query_dimension,
                                      index.header.head_dimension,
                                      index.header.kv_heads * index.header.head_dimension};
            dispatch(cb, resources, pipelines.rope, set.rope, &rope_push,
                     (index.header.query_dimension + 63u) / 64u, 1);
            compute_buffer_barrier(cb, aux.attention_states[number]);
            const uint32_t sliding = (number % 2u == 0u) ? 0x10000u : 0u;
            const CachePush attention_push{position, index.header.query_dimension,
                index.header.head_dimension,
                index.header.kv_heads * index.header.head_dimension | sliding};
            dispatch(cb, resources, pipelines.attention, set.attention,
                     &attention_push, index.header.heads, 1);
            compute_buffer_barrier(cb, aux.context);
            const GossLinearPush output_push{
                1, layer.output.rows, layer.output.columns, layer.output.packed_stride};
            dispatch(cb, resources, pipelines.qgemv_residual, set.output, &output_push,
                     (layer.output.rows + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.hidden_b);
            dispatch(cb, resources, pipelines.rmsnorm, set.mlp_norm,
                     &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            dispatch(cb, resources, pipelines.router, set.router,
                     &router_push, index.header.experts, 1);
            compute_buffer_barrier(cb, aux.routing);
            dispatch(cb, resources, pipelines.router_select, set.router,
                     &router_push, 1, 1);
        };

        auto* indirect_commands = static_cast<VkDispatchIndirectCommand*>(
            aux.indirect_commands.mapped);
        const auto indirect_dispatch = [&](VkCommandBuffer cb, uint32_t& cursor,
                                           VkPipeline pipeline, VkDescriptorSet set,
                                           const void* push, uint32_t groups_x,
                                           uint32_t groups_y) {
            if (cursor >= GOSS_MAX_INDIRECT_COMMANDS)
                throw std::runtime_error("GPT-OSS indirect command chain overflow");
            indirect_commands[cursor] = {groups_x, groups_y, 1};
            goss_dispatch_indirect(cb, resources, pipeline, set, push,
                                   aux.indirect_commands, cursor++);
        };
        const auto record_shared_layer_indirect = [&](VkCommandBuffer cb, uint32_t& cursor,
                                                      uint32_t number,
                                                      uint32_t position) {
            const GossLayerIndex& layer = shared_index.layers[number];
            const GossLayerSets& set = sets.layers[number];
            indirect_dispatch(cb, cursor, pipelines.rmsnorm, set.attention_norm,
                              &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            const GossLinearPush qkv_push{
                1, layer.qkv.rows, layer.qkv.columns, layer.qkv.packed_stride};
            indirect_dispatch(cb, cursor, pipelines.qgemv, set.qkv, &qkv_push,
                              (layer.qkv.rows + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.qkv);
            const CachePush rope_push{position, index.header.query_dimension,
                                      index.header.head_dimension,
                                      index.header.kv_heads * index.header.head_dimension};
            indirect_dispatch(cb, cursor, pipelines.rope, set.rope, &rope_push,
                              (index.header.query_dimension + 63u) / 64u, 1);
            compute_buffer_barrier(cb, aux.attention_states[number]);
            const uint32_t sliding = (number % 2u == 0u) ? 0x10000u : 0u;
            const CachePush attention_push{position, index.header.query_dimension,
                index.header.head_dimension,
                index.header.kv_heads * index.header.head_dimension | sliding};
            indirect_dispatch(cb, cursor, pipelines.attention, set.attention,
                              &attention_push, index.header.heads, 1);
            compute_buffer_barrier(cb, aux.context);
            const GossLinearPush output_push{
                1, layer.output.rows, layer.output.columns, layer.output.packed_stride};
            indirect_dispatch(cb, cursor, pipelines.qgemv_residual, set.output,
                              &output_push, (layer.output.rows + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.hidden_b);
            indirect_dispatch(cb, cursor, pipelines.rmsnorm, set.mlp_norm,
                              &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            indirect_dispatch(cb, cursor, pipelines.router, set.router,
                              &router_push, index.header.experts, 1);
            compute_buffer_barrier(cb, aux.routing);
            indirect_dispatch(cb, cursor, pipelines.router_select, set.router,
                              &router_push, 1, 1);
            compute_buffer_barrier(cb, aux.routing);
        };

        std::vector<std::array<uint32_t, 4>> last_routes(index.header.layers);
        std::vector<uint32_t> route_streak(index.header.layers, 0);
        std::vector<bool> route_seen(index.header.layers, false);
        const auto read_route = [&](uint32_t layer) {
            invalidate_buffer(runtime, aux.routing);
            const float* values = static_cast<const float*>(aux.routing.mapped) +
                static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS;
            std::array<uint32_t, 4> experts{};
            std::array<float, 4> weights{};
            for (uint32_t rank = 0; rank < 4; ++rank) {
                experts[rank] = static_cast<uint32_t>(std::lround(values[rank]));
                weights[rank] = values[4 + rank];
                if (experts[rank] >= index.header.experts || !std::isfinite(weights[rank]))
                    throw std::runtime_error("Invalid GPT-OSS router output");
                for (uint32_t prior = 0; prior < rank; ++prior) {
                    if (experts[prior] == experts[rank])
                        throw std::runtime_error("GPT-OSS router selected a duplicate expert");
                }
            }
            if (route_seen[layer] && last_routes[layer] == experts)
                ++route_streak[layer];
            else
                route_streak[layer] = 0;
            last_routes[layer] = experts;
            route_seen[layer] = true;
            return std::make_pair(experts, weights);
        };

        uint64_t cache_clock = 0;
        uint64_t hits = 0, misses = 0, transferred_bytes = 0;
        uint64_t hot_import_hits = 0, hot_import_misses = 0;
        uint64_t populated_slots = 0, peak_populated_slots = 0;
        std::vector<std::vector<uint32_t>> expert_frequency(index.header.layers,
            std::vector<uint32_t>(index.header.experts, 0));
        double expert_read_seconds = 0.0;
        bool dynamic_imports_enabled = false;
        uint64_t hot_import_clock = 0;
        auto* expert_addresses = static_cast<uint64_t*>(aux.slot_maps.mapped);
        std::fill(expert_addresses,
            expert_addresses + static_cast<uint64_t>(index.header.layers) *
            (index.header.experts + 4), 0ull);
        std::vector<uint64_t> cache_addresses(index.header.layers);
        std::vector<std::array<uint64_t, GOSS_HOT_IMPORTS>> hot_addresses(
            index.header.layers);
        for (uint32_t number = 0; number < index.header.layers; ++number)
            cache_addresses[number] = goss_buffer_address(runtime, caches[number].arena);
        bool speculative_addresses_ready = false;
        bool restore_speculative_state = false;
        const auto hot_address_for = [&](uint32_t layer, int32_t expert) {
            if (expert < 0) return uint64_t{0};
            for (uint32_t hot = 0; hot < GOSS_HOT_IMPORTS; ++hot)
                if (hot_imports[layer][hot].expert == expert)
                    return hot_addresses[layer][hot];
            return uint64_t{0};
        };
        const auto run_token = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            execute([&](VkCommandBuffer cb) {
                const GossEmbeddingPush push{shared_index.header.vocabulary,
                    shared_index.header.dimension, shared_index.embedding.packed_stride, 0};
                dispatch(cb, resources, pipelines.embedding, sets.embedding,
                         &push, (index.header.dimension + 63u) / 64u, 1);
                compute_buffer_barrier(cb, aux.hidden_a);
                record_shared_layer(cb, 0, position);
            });
            auto route = read_route(0);

            for (uint32_t number = 0; number < index.header.layers; ++number) {
                GossLayerCache& cache = caches[number];
                std::array<uint32_t, 4> selected_slots{};
                std::array<bool, GOSS_CACHE_SLOTS> occupied{};
                std::array<bool, 4> needs_upload{};
                std::array<bool, 4> direct_hot{};
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    bool found = false;
                    for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                        if (cache.slots[slot].expert ==
                            static_cast<int32_t>(route.first[rank])) {
                            selected_slots[rank] = slot;
                            occupied[slot] = true;
                            cache.slots[slot].last_used = ++cache_clock;
                            ++hits;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                    if (hot_address_for(number,
                            static_cast<int32_t>(route.first[rank])) != 0) {
                        direct_hot[rank] = true;
                        ++hits;
                        ++hot_import_hits;
                        continue;
                    }
                    uint32_t victim = 0;
                    uint64_t oldest = std::numeric_limits<uint64_t>::max();
                    for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                        if (!occupied[slot] && cache.slots[slot].last_used < oldest) {
                            oldest = cache.slots[slot].last_used;
                            victim = slot;
                        }
                    }
                    selected_slots[rank] = victim;
                    occupied[victim] = true;
                    needs_upload[rank] = true;
                    ++misses;
                }
                const auto read_start = std::chrono::steady_clock::now();
                std::array<int32_t, 4> selected_hot{{-1, -1, -1, -1}};
                std::array<bool, 4> needs_staging{};
                std::array<bool, GOSS_HOT_IMPORTS> occupied_hot{};
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    if (!needs_upload[rank]) continue;
                    for (uint32_t hot = 0; hot < GOSS_HOT_IMPORTS; ++hot) {
                        if (hot_imports[number][hot].expert ==
                            static_cast<int32_t>(route.first[rank])) {
                            selected_hot[rank] = static_cast<int32_t>(hot);
                            occupied_hot[hot] = true;
                            hot_imports[number][hot].last_used = ++hot_import_clock;
                            ++hot_import_hits;
                            break;
                        }
                    }
                    if (selected_hot[rank] >= 0) continue;
                    ++hot_import_misses;
                    if (dynamic_imports_enabled) {
                        uint32_t victim = GOSS_STATIC_HOT_IMPORTS;
                        uint64_t oldest = std::numeric_limits<uint64_t>::max();
                        for (uint32_t hot = GOSS_STATIC_HOT_IMPORTS;
                             hot < GOSS_HOT_IMPORTS; ++hot) {
                            if (!occupied_hot[hot] &&
                                hot_imports[number][hot].last_used < oldest) {
                                oldest = hot_imports[number][hot].last_used;
                                victim = hot;
                            }
                        }
                        GossHotImport& entry = hot_imports[number][victim];
                        destroy_buffer(runtime, entry.imported.buffer);
                        const uint64_t begin = GOSS_Q3_EXPERT_HEADER +
                            (static_cast<uint64_t>(number) * index.header.experts +
                             route.first[rank]) * GOSS_Q3_EXPERT_BLOCK;
                        entry.expert = static_cast<int32_t>(route.first[rank]);
                        entry.last_used = ++hot_import_clock;
                        entry.imported = import_goss_model_range(
                            runtime, expert_mapped, begin, begin + GOSS_Q3_EXPERT_BLOCK);
                        selected_hot[rank] = static_cast<int32_t>(victim);
                        occupied_hot[victim] = true;
                        continue;
                    }
                    needs_staging[rank] = true;
                    const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                        (static_cast<uint64_t>(number) * index.header.experts +
                         route.first[rank]) * GOSS_Q3_EXPERT_BLOCK;
                    constexpr uint32_t stripes = 8;
                    const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                    for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                        const uint64_t offset = stripe * stripe_bytes;
                        acquisition_pool.schedule(number, expert_mapped.data() + file_offset + offset,
                            static_cast<uint8_t*>(staging[rank].mapped) + offset,
                            static_cast<size_t>(stripe_bytes));
                    }
                }
                acquisition_pool.wait(number);
                for (uint32_t rank = 0; rank < 4; ++rank)
                    if (needs_staging[rank]) flush_buffer(runtime, staging[rank]);
                expert_read_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - read_start).count();
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    if (!needs_upload[rank]) continue;
                    GossSlot& slot = cache.slots[selected_slots[rank]];
                    if (slot.expert < 0) ++populated_slots;
                    if (speculative_addresses_ready && slot.expert >= 0)
                        expert_addresses[static_cast<uint64_t>(number) *
                            (index.header.experts + 4) + slot.expert] =
                            hot_address_for(number, slot.expert);
                    slot.expert = static_cast<int32_t>(route.first[rank]);
                    slot.last_used = ++cache_clock;
                    if (speculative_addresses_ready)
                        expert_addresses[static_cast<uint64_t>(number) *
                            (index.header.experts + 4) + route.first[rank]] =
                            cache_addresses[number] +
                            static_cast<uint64_t>(selected_slots[rank]) *
                            GOSS_Q3_EXPERT_BLOCK;
                    transferred_bytes += GOSS_Q3_EXPERT_BLOCK;
                }
                for (uint32_t expert : route.first)
                    ++expert_frequency[number][expert];
                peak_populated_slots = std::max(peak_populated_slots, populated_slots);

                const bool speculate = speculative_addresses_ready &&
                    number + 2u < index.header.layers;
                if (speculate) {
                    for (uint32_t rank = 0; rank < 4; ++rank)
                        expert_addresses[static_cast<uint64_t>(number + 1) *
                            (index.header.experts + 4) + index.header.experts + rank] = 0;
                }
                flush_buffer(runtime, aux.slot_maps);
                execute([&](VkCommandBuffer cb) {
                    bool copied = false;
                    if (restore_speculative_state) {
                        VkBufferCopy restore{};
                        restore.size = index.header.dimension * sizeof(float);
                        vkfn::CmdCopyBuffer(cb, aux.speculative_norm.handle,
                            aux.norm.handle, 1, &restore);
                        vkfn::CmdCopyBuffer(cb, aux.speculative_hidden_b.handle,
                            aux.hidden_b.handle, 1, &restore);
                        copied = true;
                        restore_speculative_state = false;
                    }
                    for (uint32_t rank = 0; rank < 4; ++rank) {
                        if (!needs_upload[rank]) continue;
                        VkBufferCopy copy{};
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        copy.dstOffset = static_cast<uint64_t>(selected_slots[rank]) *
                                         GOSS_Q3_EXPERT_BLOCK;
                        if (selected_hot[rank] >= 0) {
                            const GossImportedRange& imported =
                                hot_imports[number][selected_hot[rank]].imported;
                            const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                                (static_cast<uint64_t>(number) * index.header.experts +
                                 route.first[rank]) * GOSS_Q3_EXPERT_BLOCK;
                            copy.srcOffset = file_offset - imported.file_offset;
                            vkfn::CmdCopyBuffer(cb, imported.buffer.handle,
                                                cache.arena.handle, 1, &copy);
                        } else {
                            vkfn::CmdCopyBuffer(cb, staging[rank].handle,
                                                cache.arena.handle, 1, &copy);
                        }
                        copied = true;
                    }
                    if (copied) {
                        VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        transfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &transfer,
                            0, nullptr, 0, nullptr);
                    }
                    const GossLayerSets& layer_sets = sets.layers[number];
                    const GossFusedExpertPush expert_push{{selected_slots[0],
                        selected_slots[1], selected_slots[2], selected_slots[3]}};
                    const GossFusedExpertPush no_push{{0, 0, 0, 0}};
                    const GossQuantizePush norm_quantize{
                        index.header.dimension, 1, 0, 0};
                    dispatch(cb, resources, pipelines.quantize_expert,
                        sets.quantize_norm, &norm_quantize, 1, 1);
                    compute_buffer_barrier(cb, aux.quantized_expert_input);
                    if (speculative_addresses_ready)
                        dispatch(cb, resources, pipelines.expert_gate_table,
                            sets.layers[number].speculative_experts.gate, &no_push,
                            (index.header.hidden_dimension + 3u) / 4u, 4);
                    else
                        dispatch(cb, resources, pipelines.expert_gate,
                            layer_sets.fused_experts.gate, &expert_push,
                            (index.header.hidden_dimension + 3u) / 4u, 4);
                    compute_buffer_barrier(cb, aux.feed_forward);
                    const GossQuantizePush down_quantize{
                        index.header.hidden_dimension, index.header.top_k, 0, 0};
                    dispatch(cb, resources, pipelines.quantize_expert,
                        sets.quantize_feed_forward, &down_quantize, 1, 1);
                    compute_buffer_barrier(cb, aux.quantized_expert_input);
                    if (speculative_addresses_ready)
                        dispatch(cb, resources, pipelines.expert_down_table,
                            sets.layers[number].speculative_experts.down, &no_push,
                            (index.header.dimension + 3u) / 4u, 1);
                    else
                        dispatch(cb, resources, pipelines.expert_down,
                            layer_sets.fused_experts.down, &expert_push,
                            (index.header.dimension + 3u) / 4u, 1);
                    compute_buffer_barrier(cb, aux.hidden_a);
                    if (speculate) {
                        record_shared_layer(cb, number + 1, position);
                        VkMemoryBarrier save_source{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        save_source.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        save_source.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                        vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &save_source,
                            0, nullptr, 0, nullptr);
                        VkBufferCopy save{};
                        save.size = index.header.dimension * sizeof(float);
                        vkfn::CmdCopyBuffer(cb, aux.norm.handle,
                            aux.speculative_norm.handle, 1, &save);
                        vkfn::CmdCopyBuffer(cb, aux.hidden_b.handle,
                            aux.speculative_hidden_b.handle, 1, &save);
                        VkMemoryBarrier save_complete{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        save_complete.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        save_complete.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_SHADER_WRITE_BIT;
                        vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &save_complete,
                            0, nullptr, 0, nullptr);
                        const GossFusedExpertPush no_push{{0, 0, 0, 0}};
                        const GossQuantizePush norm_quantize{
                            index.header.dimension, 1, 0, 0};
                        dispatch(cb, resources, pipelines.quantize_expert,
                            sets.quantize_norm, &norm_quantize, 1, 1);
                        compute_buffer_barrier(cb, aux.quantized_expert_input);
                        dispatch(cb, resources, pipelines.expert_gate_table,
                            sets.layers[number + 1].speculative_experts.gate, &no_push,
                            (index.header.hidden_dimension + 3u) / 4u, 4);
                        compute_buffer_barrier(cb, aux.feed_forward);
                        const GossQuantizePush down_quantize{
                            index.header.hidden_dimension, index.header.top_k, 0, 0};
                        dispatch(cb, resources, pipelines.quantize_expert,
                            sets.quantize_feed_forward, &down_quantize, 1, 1);
                        compute_buffer_barrier(cb, aux.quantized_expert_input);
                        dispatch(cb, resources, pipelines.expert_down_table,
                            sets.layers[number + 1].speculative_experts.down, &no_push,
                            (index.header.dimension + 3u) / 4u, 1);
                        compute_buffer_barrier(cb, aux.hidden_a);
                        record_shared_layer(cb, number + 2, position);
                    } else if (number + 1 < index.header.layers) {
                        record_shared_layer(cb, number + 1, position);
                    } else {
                        dispatch(cb, resources, pipelines.rmsnorm, sets.final_norm,
                                 &norm_push, 1, 1);
                        compute_buffer_barrier(cb, aux.norm);
                        qgemv(cb, sets.logits, shared_index.unembedding, false);
                        compute_buffer_barrier(cb, aux.logits);
                        constexpr uint32_t greedy_groups = 256;
                        const GossGreedyPush greedy_push{
                            index.header.vocabulary, greedy_groups, 0, 0};
                        dispatch(cb, resources, pipelines.greedy, sets.greedy,
                                 &greedy_push, greedy_groups, 1);
                        compute_buffer_barrier(cb, aux.feed_forward);
                        const GossGreedyPush greedy_finish{
                            index.header.vocabulary, greedy_groups, 1, 0};
                        dispatch(cb, resources, pipelines.greedy, sets.greedy,
                                 &greedy_finish, 1, 1);
                    }
                });
                if (speculate) {
                    invalidate_buffer(runtime, aux.slot_maps);
                    uint64_t missed = 0;
                    for (uint32_t rank = 0; rank < 4; ++rank)
                        missed |= expert_addresses[static_cast<uint64_t>(number + 1) *
                            (index.header.experts + 4) + index.header.experts + rank];
                    if (missed == 0) {
                        restore_speculative_state = false;
                        const auto skipped_route = read_route(number + 1);
                        GossLayerCache& skipped_cache = caches[number + 1];
                        for (uint32_t expert : skipped_route.first) {
                            ++expert_frequency[number + 1][expert];
                            for (GossSlot& cached : skipped_cache.slots)
                                if (cached.expert == static_cast<int32_t>(expert)) {
                                    cached.last_used = ++cache_clock;
                                    break;
                                }
                        }
                        route = read_route(number + 2);
                        ++number;
                    } else {
                        restore_speculative_state = true;
                        route = read_route(number + 1);
                    }
                } else if (number + 1 < index.header.layers) {
                    route = read_route(number + 1);
                }
            }
            invalidate_buffer(runtime, aux.token_parameter);
        };

        std::vector<uint32_t> tokens;
        const auto special = [&](const char* piece) { tokens.push_back(tokenizer.token_id(piece)); };
        const auto text = [&](const std::string& value) {
            const std::vector<uint32_t> encoded = tokenizer.encode(value);
            tokens.insert(tokens.end(), encoded.begin(), encoded.end());
        };
        special("<|start|>"); text("system"); special("<|message|>");
        text("You are ChatGPT, a large language model trained by OpenAI.\n"
             "Knowledge cutoff: 2024-06\nCurrent date: 2026-08-07\n\n"
             "Reasoning: low\n\nValid channels: analysis, final.");
        special("<|end|>"); special("<|start|>"); text("user"); special("<|message|>");
        text(prompt); special("<|end|>"); special("<|start|>"); text("assistant");
        special("<|channel|>"); text("analysis"); special("<|message|>");
        if (tokens.size() + generation_limit > index.header.max_sequence) {
            throw std::runtime_error("GPT-OSS prompt exceeds the milestone KV cache");
        }
        for (uint32_t position = 0; position < tokens.size(); ++position) {
            if ((position % 8u) == 0u)
                std::cerr << "prompt position=" << position << "/" << tokens.size() << "\n";
            run_token(tokens[position], position);
        }
        std::cerr << "prompt complete\n";

        std::vector<std::array<GossImportedRange, 8>> layer_imports(index.header.layers);
        std::vector<GossExpertSets> dynamic_experts(index.header.layers);

        for (uint32_t number = 0; number < index.header.layers; ++number) {
            std::vector<uint32_t> order(index.header.experts);
            for (uint32_t expert = 0; expert < index.header.experts; ++expert)
                order[expert] = expert;
            std::stable_sort(order.begin(), order.end(), [&](uint32_t left, uint32_t right) {
                return expert_frequency[number][left] > expert_frequency[number][right];
            });
            for (uint32_t hot = 0; hot < GOSS_STATIC_HOT_IMPORTS; ++hot) {
                const uint32_t expert = order[hot];
                const uint64_t begin = GOSS_Q3_EXPERT_HEADER +
                    (static_cast<uint64_t>(number) * index.header.experts + expert) *
                    GOSS_Q3_EXPERT_BLOCK;
                hot_imports[number][hot].expert = static_cast<int32_t>(expert);
                hot_imports[number][hot].last_used = ++hot_import_clock;
                hot_imports[number][hot].imported = import_goss_model_range(
                    runtime, expert_mapped, begin, begin + GOSS_Q3_EXPERT_BLOCK);
            }
        }

        for (uint32_t number = 0; number < index.header.layers; ++number) {
            const uint64_t table_base = static_cast<uint64_t>(number) *
                (index.header.experts + 4);
            for (uint32_t hot = 0; hot < GOSS_HOT_IMPORTS; ++hot) {
                const GossHotImport& entry = hot_imports[number][hot];
                if (entry.expert < 0) continue;
                const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                    (static_cast<uint64_t>(number) * index.header.experts +
                     entry.expert) * GOSS_Q3_EXPERT_BLOCK;
                hot_addresses[number][hot] =
                    goss_buffer_address(runtime, entry.imported.buffer) +
                    file_offset - entry.imported.file_offset;
                expert_addresses[table_base +
                    static_cast<uint32_t>(entry.expert)] = hot_addresses[number][hot];
            }
            for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                const int32_t expert = caches[number].slots[slot].expert;
                if (expert >= 0)
                    expert_addresses[table_base + expert] = cache_addresses[number] +
                        static_cast<uint64_t>(slot) * GOSS_Q3_EXPERT_BLOCK;
            }
            for (uint32_t rank = 0; rank < 4; ++rank)
                expert_addresses[table_base + index.header.experts + rank] = 0;
        }
        // The speculative descriptor uses the compact address table, while the
        // cache-managed BDA shaders use the larger control record.  Keep this
        // path disabled until those interfaces are separated; binding one as
        // the other permits out-of-range reads and can TDR the driver.
        speculative_addresses_ready = true;
        flush_buffer(runtime, aux.slot_maps);

        // Device-managed Q4 cache metadata.  Each layer owns one 4 KiB control
        // record shared by the resolver, the compute-copy path, and BDA experts.
        constexpr uint32_t C_EXPERT_TO_SLOT = 0;
        constexpr uint32_t C_SLOT_EXPERT = 128;
        constexpr uint32_t C_SLOT_AGE = 148;
        constexpr uint32_t C_SELECTED_SLOT = 168;
        constexpr uint32_t C_NEEDS_COPY = 172;
        constexpr uint32_t C_MISSING_COUNT = 176;
        constexpr uint32_t C_MISSING_ID = 177;
        constexpr uint32_t C_MISSING_SLOT = 181;
        constexpr uint32_t C_CLOCK = 185;
        constexpr uint32_t C_REQUEST_STATE = 186;
        constexpr uint32_t C_HOST_ADDRESS = 256;
        constexpr uint32_t C_SELECTED_ADDRESS = 512;
        constexpr uint32_t C_CACHE_ADDRESS = 520;
        constexpr uint32_t C_COLD_ADDRESS = 528;
        constexpr uint32_t C_ZERO_ADDRESS = 536;
        const auto layer_control = [&](uint32_t layer) {
            return reinterpret_cast<uint32_t*>(
                static_cast<uint8_t*>(aux.cache_control.mapped) +
                static_cast<uint64_t>(layer) * GOSS_GPU_CONTROL_STRIDE);
        };
        const auto store_control_address = [](uint32_t* control, uint32_t word,
                                              uint64_t address) {
            control[word] = static_cast<uint32_t>(address);
            control[word + 1] = static_cast<uint32_t>(address >> 32u);
        };
        for (uint32_t number = 0; number < index.header.layers; ++number) {
            uint32_t* control = layer_control(number);
            std::fill(control + C_EXPERT_TO_SLOT,
                      control + C_EXPERT_TO_SLOT + index.header.experts, 0xffffffffu);
            std::fill(control + C_SLOT_EXPERT,
                      control + C_SLOT_EXPERT + GOSS_CACHE_SLOTS, 0xffffffffu);
            uint32_t age = 0;
            for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                const int32_t expert = caches[number].slots[slot].expert;
                if (expert >= 0) {
                    control[C_EXPERT_TO_SLOT + expert] = slot;
                    control[C_SLOT_EXPERT + slot] = static_cast<uint32_t>(expert);
                }
                control[C_SLOT_AGE + slot] = ++age;
            }
            control[C_CLOCK] = age;
            for (uint32_t expert = 0; expert < index.header.experts; ++expert)
                store_control_address(control, C_HOST_ADDRESS + 2u * expert, 0);
            for (uint32_t hot = 0; hot < GOSS_HOT_IMPORTS; ++hot) {
                const GossHotImport& entry = hot_imports[number][hot];
                if (entry.expert >= 0)
                    store_control_address(control,
                        C_HOST_ADDRESS + 2u * static_cast<uint32_t>(entry.expert),
                        hot_addresses[number][hot]);
            }
            store_control_address(control, C_CACHE_ADDRESS, cache_addresses[number]);
            store_control_address(control, C_ZERO_ADDRESS, zero_expert_address);
        }
        flush_buffer(runtime, aux.cache_control);

        double gpu_cache_acquisition_seconds = 0.0;
        std::vector<uint64_t> gpu_layer_cold(index.header.layers, 0);
        std::vector<uint64_t> gpu_layer_hot_copy(index.header.layers, 0);
        std::vector<uint32_t> dynamic_hot_victim(index.header.layers, 0);
        std::vector<uint32_t> gpu_route_trace;
        std::vector<std::array<uint32_t, 4>> previous_gpu_routes(index.header.layers);
        std::vector<std::array<uint32_t, 16>> previous_router_top16(index.header.layers);
        bool have_previous_gpu_routes = false;
        bool have_previous_router_top16 = false;
        uint64_t previous_route_set_matches = 0;
        uint64_t previous_route_rank_matches = 0;
        uint64_t previous_route_comparisons = 0;
        uint64_t previous_top8_matches = 0;
        uint64_t previous_top16_matches = 0;
        uint64_t previous_top16_comparisons = 0;
        uint64_t previous_cold_top8_matches = 0;
        uint64_t previous_cold_top16_matches = 0;
        uint64_t previous_cold_comparisons = 0;
        uint64_t early_rank_matches = 0;
        uint64_t early_top4_matches = 0;
        uint64_t early_top8_matches = 0;
        uint64_t early_top16_matches = 0;
        uint64_t early_comparisons = 0;
        uint64_t early_cold_top4_matches = 0;
        uint64_t early_cold_top8_matches = 0;
        uint64_t early_cold_top16_matches = 0;
        uint64_t early_cold_comparisons = 0;
        const auto record_gpu_expert = [&](VkCommandBuffer cb, uint32_t& cursor,
                                           uint32_t number) {
            const GossFusedExpertPush no_push{{0, 0, 0, 0}};
            const GossQuantizePush norm_quantize{
                index.header.dimension, 1, 0, 0};
            indirect_dispatch(cb, cursor, pipelines.quantize_expert,
                sets.quantize_norm, &norm_quantize, 1, 1);
            compute_buffer_barrier(cb, aux.quantized_expert_input);
            indirect_dispatch(cb, cursor, pipelines.expert_gate_speculative,
                sets.layers[number].gpu_experts.gate, &no_push,
                (index.header.hidden_dimension + 3u) / 4u, 4);
            compute_buffer_barrier(cb, aux.feed_forward);
            const GossQuantizePush down_quantize{
                index.header.hidden_dimension, index.header.top_k, 0, 0};
            indirect_dispatch(cb, cursor, pipelines.quantize_expert,
                sets.quantize_feed_forward, &down_quantize, 1, 1);
            compute_buffer_barrier(cb, aux.quantized_expert_input);
            indirect_dispatch(cb, cursor, pipelines.expert_down_speculative,
                sets.layers[number].gpu_experts.down, &no_push,
                (index.header.dimension + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.hidden_a);
        };
        const auto record_gpu_layer = [&](VkCommandBuffer cb, uint32_t& cursor,
                                          uint32_t number, uint32_t position) {
            record_shared_layer_indirect(cb, cursor, number, position);
            const uint32_t resolver_command = cursor;
            const GossResolvePush resolve_push{resolver_command + 1u,
                resolver_command + 2u, GOSS_MAX_INDIRECT_COMMANDS, 0};
            indirect_dispatch(cb, cursor, pipelines.cache_resolve,
                sets.layers[number].cache_resolve, &resolve_push, 1, 1);
            VkMemoryBarrier resolve_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            resolve_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            resolve_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                             VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &resolve_barrier,
                0, nullptr, 0, nullptr);
            const GossFusedExpertPush no_push{{0, 0, 0, 0}};
            indirect_dispatch(cb, cursor, pipelines.cache_copy,
                sets.layers[number].cache_copy, &no_push, 8, 4);
            compute_buffer_barrier(cb, caches[number].arena);
            indirect_dispatch(cb, cursor, pipelines.cache_copy_done,
                sets.layers[number].cache_resolve, &no_push, 1, 1);
            record_gpu_expert(cb, cursor, number);
        };
        const auto record_gpu_final = [&](VkCommandBuffer cb, uint32_t& cursor) {
            indirect_dispatch(cb, cursor, pipelines.rmsnorm, sets.final_norm,
                              &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            const GossLinearPush logits_push{0, shared_index.unembedding.rows,
                shared_index.unembedding.columns,
                shared_index.unembedding.packed_stride};
            indirect_dispatch(cb, cursor, pipelines.qgemv, sets.logits,
                &logits_push, (shared_index.unembedding.rows + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.logits);
            constexpr uint32_t greedy_groups = 256;
            const GossGreedyPush greedy_push{
                index.header.vocabulary, greedy_groups, 0, 0};
            indirect_dispatch(cb, cursor, pipelines.greedy, sets.greedy,
                              &greedy_push, greedy_groups, 1);
            compute_buffer_barrier(cb, aux.feed_forward);
            const GossGreedyPush greedy_finish{
                index.header.vocabulary, greedy_groups, 1, 0};
            indirect_dispatch(cb, cursor, pipelines.greedy, sets.greedy,
                              &greedy_finish, 1, 1);
        };

        std::array<uint64_t, 4> staging_addresses{};
        for (uint32_t rank = 0; rank < 4; ++rank)
            staging_addresses[rank] = goss_buffer_address(runtime, staging[rank]);

        const auto record_gpu_expert_direct = [&](VkCommandBuffer cb, uint32_t number) {
            const GossFusedExpertPush no_push{{0, 0, 0, 0}};
            const GossQuantizePush norm_quantize{
                index.header.dimension, 1, 0, 0};
            dispatch(cb, resources, pipelines.quantize_expert,
                sets.quantize_norm, &norm_quantize, 1, 1);
            compute_buffer_barrier(cb, aux.quantized_expert_input);
            dispatch(cb, resources, pipelines.expert_gate_speculative,
                sets.layers[number].gpu_experts.gate, &no_push,
                (index.header.hidden_dimension + 3u) / 4u, 4);
            compute_buffer_barrier(cb, aux.feed_forward);
            const GossQuantizePush down_quantize{
                index.header.hidden_dimension, index.header.top_k, 0, 0};
            dispatch(cb, resources, pipelines.quantize_expert,
                sets.quantize_feed_forward, &down_quantize, 1, 1);
            compute_buffer_barrier(cb, aux.quantized_expert_input);
            dispatch(cb, resources, pipelines.expert_down_speculative,
                sets.layers[number].gpu_experts.down, &no_push,
                (index.header.dimension + 3u) / 4u, 1);
            compute_buffer_barrier(cb, aux.hidden_a);
        };
        const auto record_gpu_final_direct = [&](VkCommandBuffer cb) {
            dispatch(cb, resources, pipelines.rmsnorm, sets.final_norm,
                     &norm_push, 1, 1);
            compute_buffer_barrier(cb, aux.norm);
            qgemv(cb, sets.logits, shared_index.unembedding, false);
            compute_buffer_barrier(cb, aux.logits);
            constexpr uint32_t greedy_groups = 256;
            const GossGreedyPush greedy_push{
                index.header.vocabulary, greedy_groups, 0, 0};
            dispatch(cb, resources, pipelines.greedy, sets.greedy,
                     &greedy_push, greedy_groups, 1);
            compute_buffer_barrier(cb, aux.feed_forward);
            const GossGreedyPush greedy_finish{
                index.header.vocabulary, greedy_groups, 1, 0};
            dispatch(cb, resources, pipelines.greedy, sets.greedy,
                     &greedy_finish, 1, 1);
        };

        const auto run_token_timeline = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                uint32_t* control = layer_control(layer);
                control[C_MISSING_COUNT] = 0;
                control[C_REQUEST_STATE] = 0;
                for (uint32_t rank = 0; rank < 4; ++rank)
                    control[C_NEEDS_COPY + rank] = 0;
            }
            flush_buffer(runtime, aux.cache_control);

            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                VkCommandBuffer shared_cb = layer_commands[layer * 2u];
                VkCommandBuffer expert_cb = layer_commands[layer * 2u + 1u];
                VK_CHECK(vkfn::ResetCommandBuffer(shared_cb, 0));
                VK_CHECK(vkfn::ResetCommandBuffer(expert_cb, 0));
                VkCommandBufferBeginInfo begin{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkfn::BeginCommandBuffer(shared_cb, &begin));
                if (layer == 0) {
                    VkMemoryBarrier host_upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    host_upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    host_upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(shared_cb, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_upload,
                        0, nullptr, 0, nullptr);
                    const GossEmbeddingPush embedding_push{
                        shared_index.header.vocabulary,
                        shared_index.header.dimension,
                        shared_index.embedding.packed_stride, 0};
                    dispatch(shared_cb, resources, pipelines.embedding, sets.embedding,
                        &embedding_push, (index.header.dimension + 63u) / 64u, 1);
                    compute_buffer_barrier(shared_cb, aux.hidden_a);
                }
                dispatch(shared_cb, resources, pipelines.rmsnorm,
                    sets.layers[layer].prediction_norm, &norm_push, 1, 1);
                compute_buffer_barrier(shared_cb, aux.speculative_norm);
                const GossRouterPush prediction_router_push{
                    index.header.dimension, index.header.experts,
                    index.header.top_k, 0};
                dispatch(shared_cb, resources, pipelines.router,
                    sets.layers[layer].prediction_router,
                    &prediction_router_push, index.header.experts, 1);
                compute_buffer_barrier(shared_cb, aux.prediction_routing);
                dispatch(shared_cb, resources, pipelines.router_select,
                    sets.layers[layer].prediction_router,
                    &prediction_router_push, 1, 1);
                compute_buffer_barrier(shared_cb, aux.prediction_routing);
                record_shared_layer(shared_cb, layer, position);
                compute_buffer_barrier(shared_cb, aux.routing);
                const GossResolvePush resolve_push{0, 0, 0, 0};
                dispatch(shared_cb, resources, pipelines.cache_resolve,
                    sets.layers[layer].cache_resolve, &resolve_push, 1, 1);
                VkMemoryBarrier host_result{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host_result.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                host_result.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkfn::CmdPipelineBarrier(shared_cb,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_result,
                    0, nullptr, 0, nullptr);
                VK_CHECK(vkfn::EndCommandBuffer(shared_cb));

                VK_CHECK(vkfn::BeginCommandBuffer(expert_cb, &begin));
                VkMemoryBarrier host_ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host_ready.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                host_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkfn::CmdPipelineBarrier(expert_cb, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_ready,
                    0, nullptr, 0, nullptr);
                const GossFusedExpertPush no_push{{0, 0, 0, 0}};
                dispatch(expert_cb, resources, pipelines.cache_copy,
                    sets.layers[layer].cache_copy, &no_push, 8, 4);
                compute_buffer_barrier(expert_cb, caches[layer].arena);
                record_gpu_expert_direct(expert_cb, layer);
                if (layer + 1u == index.header.layers)
                    record_gpu_final_direct(expert_cb);
                VK_CHECK(vkfn::EndCommandBuffer(expert_cb));
            }

            const uint64_t base = layer_timeline_base;
            std::vector<VkSubmitInfo> submits(index.header.layers * 2u);
            std::vector<VkTimelineSemaphoreSubmitInfo> timeline_submits(
                index.header.layers * 2u);
            std::vector<uint64_t> wait_values(index.header.layers * 2u, 0);
            std::vector<uint64_t> signal_values(index.header.layers * 2u, 0);
            std::vector<VkPipelineStageFlags> wait_stages(index.header.layers * 2u,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                const uint32_t shared_index_submit = layer * 2u;
                const uint32_t expert_index_submit = shared_index_submit + 1u;
                const uint64_t layer_value = base + layer + 1u;
                signal_values[shared_index_submit] = layer_value;
                timeline_submits[shared_index_submit] = {
                    VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timeline_submits[shared_index_submit].signalSemaphoreValueCount = 1;
                timeline_submits[shared_index_submit].pSignalSemaphoreValues =
                    &signal_values[shared_index_submit];
                submits[shared_index_submit] = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submits[shared_index_submit].pNext =
                    &timeline_submits[shared_index_submit];
                submits[shared_index_submit].commandBufferCount = 1;
                submits[shared_index_submit].pCommandBuffers =
                    &layer_commands[shared_index_submit];
                submits[shared_index_submit].signalSemaphoreCount = 1;
                submits[shared_index_submit].pSignalSemaphores = &gpu_layer_timeline;

                wait_values[expert_index_submit] = layer_value;
                timeline_submits[expert_index_submit] = {
                    VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timeline_submits[expert_index_submit].waitSemaphoreValueCount = 1;
                timeline_submits[expert_index_submit].pWaitSemaphoreValues =
                    &wait_values[expert_index_submit];
                submits[expert_index_submit] = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submits[expert_index_submit].pNext =
                    &timeline_submits[expert_index_submit];
                submits[expert_index_submit].waitSemaphoreCount = 1;
                submits[expert_index_submit].pWaitSemaphores = &host_layer_timeline;
                submits[expert_index_submit].pWaitDstStageMask =
                    &wait_stages[expert_index_submit];
                submits[expert_index_submit].commandBufferCount = 1;
                submits[expert_index_submit].pCommandBuffers =
                    &layer_commands[expert_index_submit];
            }
            const uint64_t completion_value = base + index.header.layers + 1u;
            const uint32_t final_submit = index.header.layers * 2u - 1u;
            signal_values[final_submit] = completion_value;
            timeline_submits[final_submit].signalSemaphoreValueCount = 1;
            timeline_submits[final_submit].pSignalSemaphoreValues =
                &signal_values[final_submit];
            submits[final_submit].signalSemaphoreCount = 1;
            submits[final_submit].pSignalSemaphores = &gpu_layer_timeline;

            flush_buffer(runtime, aux.cache_control);
            const auto gpu_start = std::chrono::steady_clock::now();
            VK_CHECK(vkfn::QueueSubmit(runtime.queue,
                static_cast<uint32_t>(submits.size()), submits.data(), VK_NULL_HANDLE));
            double acquisition_seconds = 0.0;
            uint64_t token_cold_misses = 0;
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                const uint64_t layer_value = base + layer + 1u;
                VkSemaphoreWaitInfo wait_info{
                    VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                wait_info.semaphoreCount = 1;
                wait_info.pSemaphores = &gpu_layer_timeline;
                wait_info.pValues = &layer_value;
                VK_CHECK(vkfn::WaitSemaphores(runtime.device, &wait_info, UINT64_MAX));
                invalidate_buffer(runtime, aux.cache_control);
                uint32_t* control = layer_control(layer);
                const uint32_t missing_count = control[C_MISSING_COUNT];
                if (missing_count > 4)
                    throw std::runtime_error("Invalid timeline cache miss count");
                if (missing_count != 0) {
                    const auto acquisition_start = std::chrono::steady_clock::now();
                    constexpr uint32_t stripes = 8;
                    const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        const uint32_t expert = control[C_MISSING_ID + item];
                        const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                            (static_cast<uint64_t>(layer) * index.header.experts + expert) *
                            GOSS_Q3_EXPERT_BLOCK;
                        for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                            const uint64_t offset = stripe * stripe_bytes;
                            acquisition_pool.schedule(layer,
                                expert_mapped.data() + file_offset + offset,
                                static_cast<uint8_t*>(staging[item].mapped) + offset,
                                static_cast<size_t>(stripe_bytes));
                        }
                    }
                    acquisition_pool.wait(layer);
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        flush_buffer(runtime, staging[item]);
                        const uint32_t missing_slot = control[C_MISSING_SLOT + item];
                        for (uint32_t rank = 0; rank < 4; ++rank) {
                            if (control[C_SELECTED_SLOT + rank] == missing_slot) {
                                store_control_address(control,
                                    C_COLD_ADDRESS + rank * 2u,
                                    staging_addresses[item]);
                                break;
                            }
                        }
                    }
                    control[C_REQUEST_STATE] = 3;
                    flush_buffer(runtime, aux.cache_control);
                    acquisition_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - acquisition_start).count();
                    token_cold_misses += missing_count;
                    gpu_layer_cold[layer] += missing_count;
                }
                VkSemaphoreSignalInfo signal_info{
                    VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
                signal_info.semaphore = host_layer_timeline;
                signal_info.value = layer_value;
                VK_CHECK(vkfn::SignalSemaphore(runtime.device, &signal_info));
            }
            VkSemaphoreWaitInfo completion_wait{
                VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            completion_wait.semaphoreCount = 1;
            completion_wait.pSemaphores = &gpu_layer_timeline;
            completion_wait.pValues = &completion_value;
            VK_CHECK(vkfn::WaitSemaphores(runtime.device, &completion_wait, UINT64_MAX));
            gpu_submission_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - gpu_start).count();
            expert_read_seconds += acquisition_seconds;
            gpu_cache_acquisition_seconds += acquisition_seconds;
            layer_timeline_base = completion_value;
            invalidate_buffer(runtime, aux.token_parameter);
            invalidate_buffer(runtime, aux.cache_control);
            invalidate_buffer(runtime, aux.routing);
            invalidate_buffer(runtime, aux.prediction_routing);

            uint64_t token_hot_copies = 0;
            const float* traced_routes = static_cast<const float*>(aux.routing.mapped);
            const float* predicted_routes =
                static_cast<const float*>(aux.prediction_routing.mapped);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                std::array<uint32_t, 16> current_top16{};
                for (uint32_t rank = 0; rank < 4; ++rank)
                    current_top16[rank] = static_cast<uint32_t>(std::lround(
                        traced_routes[static_cast<uint64_t>(layer) *
                                      GOSS_ROUTE_FLOATS + rank]));
                for (uint32_t rank = 4; rank < 16; ++rank)
                    current_top16[rank] = static_cast<uint32_t>(std::lround(
                        traced_routes[static_cast<uint64_t>(layer) *
                                      GOSS_ROUTE_FLOATS + 8u + rank - 4u]));
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    const uint32_t mode = layer_control(layer)[C_NEEDS_COPY + rank];
                    token_hot_copies += mode == 1u;
                    gpu_layer_hot_copy[layer] += mode == 1u;
                    gpu_route_trace.push_back(current_top16[rank]);
                    const uint32_t predicted_rank = static_cast<uint32_t>(std::lround(
                        predicted_routes[static_cast<uint64_t>(layer) *
                                         GOSS_ROUTE_FLOATS + rank]));
                    const auto prediction_contains = [&](uint32_t count) {
                        for (uint32_t candidate = 0; candidate < count; ++candidate) {
                            const uint32_t offset = candidate < 4 ? candidate :
                                8u + candidate - 4u;
                            if (static_cast<uint32_t>(std::lround(predicted_routes[
                                    static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS +
                                    offset])) == current_top16[rank]) return true;
                        }
                        return false;
                    };
                    const bool in_early4 = prediction_contains(4);
                    const bool in_early8 = prediction_contains(8);
                    const bool in_early16 = prediction_contains(16);
                    early_rank_matches += predicted_rank == current_top16[rank];
                    early_top4_matches += in_early4;
                    early_top8_matches += in_early8;
                    early_top16_matches += in_early16;
                    ++early_comparisons;
                    if (mode == 2u) {
                        early_cold_top4_matches += in_early4;
                        early_cold_top8_matches += in_early8;
                        early_cold_top16_matches += in_early16;
                        ++early_cold_comparisons;
                    }
                    if (have_previous_router_top16) {
                        const bool in_top8 = std::find(
                            previous_router_top16[layer].begin(),
                            previous_router_top16[layer].begin() + 8,
                            current_top16[rank]) !=
                            previous_router_top16[layer].begin() + 8;
                        const bool in_top16 = std::find(
                            previous_router_top16[layer].begin(),
                            previous_router_top16[layer].end(),
                            current_top16[rank]) !=
                            previous_router_top16[layer].end();
                        previous_top8_matches += in_top8;
                        previous_top16_matches += in_top16;
                        ++previous_top16_comparisons;
                        if (mode == 2u) {
                            previous_cold_top8_matches += in_top8;
                            previous_cold_top16_matches += in_top16;
                            ++previous_cold_comparisons;
                        }
                    }
                }
                previous_router_top16[layer] = current_top16;
            }
            have_previous_router_top16 = true;
            const uint64_t token_misses = token_hot_copies + token_cold_misses;
            hits += static_cast<uint64_t>(index.header.layers) * index.header.top_k -
                    token_misses;
            misses += token_misses;
            hot_import_hits += token_hot_copies;
            hot_import_misses += token_cold_misses;
            transferred_bytes += token_misses * GOSS_Q3_EXPERT_BLOCK;
        };

        uint64_t parallel_prefetch_value = 0;
        uint64_t parallel_main_value = 0;
        const auto run_token_parallel = [&](uint32_t token, uint32_t position) {
            const auto token_start = std::chrono::steady_clock::now();
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            double acquisition_seconds = 0.0;
            uint64_t token_prefetch_copies = 0;
            uint64_t token_actual_hot = 0;
            uint64_t token_actual_cold = 0;

            const auto begin_command = [](VkCommandBuffer cb) {
                VK_CHECK(vkfn::ResetCommandBuffer(cb, 0));
                VkCommandBufferBeginInfo begin{
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                VK_CHECK(vkfn::BeginCommandBuffer(cb, &begin));
            };
            const auto wait_timeline = [&](VkSemaphore semaphore, uint64_t value) {
                VkSemaphoreWaitInfo wait_info{
                    VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
                wait_info.semaphoreCount = 1;
                wait_info.pSemaphores = &semaphore;
                wait_info.pValues = &value;
                VK_CHECK(vkfn::WaitSemaphores(runtime.device, &wait_info, UINT64_MAX));
            };
            const auto submit_signal = [&](VkQueue queue, VkCommandBuffer cb,
                                           VkSemaphore signal_semaphore,
                                           uint64_t signal_value,
                                           VkSemaphore wait_semaphore,
                                           uint64_t wait_value) {
                VkTimelineSemaphoreSubmitInfo timeline_info{
                    VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
                timeline_info.signalSemaphoreValueCount = 1;
                timeline_info.pSignalSemaphoreValues = &signal_value;
                if (wait_semaphore != VK_NULL_HANDLE) {
                    timeline_info.waitSemaphoreValueCount = 1;
                    timeline_info.pWaitSemaphoreValues = &wait_value;
                }
                VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                submit.pNext = &timeline_info;
                submit.commandBufferCount = 1;
                submit.pCommandBuffers = &cb;
                submit.signalSemaphoreCount = 1;
                submit.pSignalSemaphores = &signal_semaphore;
                if (wait_semaphore != VK_NULL_HANDLE) {
                    submit.waitSemaphoreCount = 1;
                    submit.pWaitSemaphores = &wait_semaphore;
                    submit.pWaitDstStageMask = &wait_stage;
                }
                VK_CHECK(vkfn::QueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
            };

            VkCommandBuffer prediction_cb = layer_commands[index.header.layers * 2u];
            VkCommandBuffer copy_cb = layer_commands[index.header.layers * 2u + 1u];
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                uint32_t* control = layer_control(layer);
                control[C_MISSING_COUNT] = 0;
                control[C_REQUEST_STATE] = 0;
                for (uint32_t rank = 0; rank < 4; ++rank)
                    control[C_NEEDS_COPY + rank] = 0;
                flush_buffer(runtime, aux.cache_control);

                VkCommandBuffer shared_cb = layer_commands[layer * 2u];
                VkCommandBuffer expert_cb = layer_commands[layer * 2u + 1u];
                begin_command(shared_cb);
                if (layer == 0) {
                    VkMemoryBarrier host_upload{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    host_upload.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    host_upload.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(shared_cb, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_upload,
                        0, nullptr, 0, nullptr);
                    const GossEmbeddingPush embedding_push{
                        shared_index.header.vocabulary,
                        shared_index.header.dimension,
                        shared_index.embedding.packed_stride, 0};
                    dispatch(shared_cb, resources, pipelines.embedding, sets.embedding,
                        &embedding_push, (index.header.dimension + 63u) / 64u, 1);
                    compute_buffer_barrier(shared_cb, aux.hidden_a);
                }
                record_shared_layer(shared_cb, layer, position);
                compute_buffer_barrier(shared_cb, aux.routing);
                VK_CHECK(vkfn::EndCommandBuffer(shared_cb));

                uint64_t copy_value = 0;
                if (layer != 0) {
                    begin_command(prediction_cb);
                    dispatch(prediction_cb, resources, pipelines.rmsnorm,
                        sets.layers[layer].prediction_norm, &norm_push, 1, 1);
                    compute_buffer_barrier(prediction_cb, aux.speculative_norm);
                    const GossRouterPush prediction_push{
                        index.header.dimension, index.header.experts,
                        index.header.top_k, 0};
                    dispatch(prediction_cb, resources, pipelines.router,
                        sets.layers[layer].prediction_router, &prediction_push,
                        index.header.experts, 1);
                    compute_buffer_barrier(prediction_cb, aux.prediction_routing);
                    dispatch(prediction_cb, resources, pipelines.router_select,
                        sets.layers[layer].prediction_router, &prediction_push, 1, 1);
                    VkMemoryBarrier prediction_host{
                        VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    prediction_host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    prediction_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                    vkfn::CmdPipelineBarrier(prediction_cb,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &prediction_host,
                        0, nullptr, 0, nullptr);
                    VK_CHECK(vkfn::EndCommandBuffer(prediction_cb));

                    const uint64_t prediction_value = ++parallel_prefetch_value;
                    submit_signal(runtime.secondary_queue, prediction_cb,
                        host_layer_timeline, prediction_value,
                        gpu_layer_timeline, parallel_main_value);

                    VkSubmitInfo shared_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    shared_submit.commandBufferCount = 1;
                    shared_submit.pCommandBuffers = &shared_cb;
                    VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &shared_submit,
                                               VK_NULL_HANDLE));

                    wait_timeline(host_layer_timeline, prediction_value);
                    invalidate_buffer(runtime, aux.prediction_routing);
                    const float* predicted =
                        static_cast<const float*>(aux.prediction_routing.mapped) +
                        static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS;

                    struct PlannedCopy {
                        VkBuffer source;
                        uint64_t source_offset;
                        uint64_t destination_offset;
                    };
                    std::vector<PlannedCopy> planned;
                    std::array<bool, GOSS_CACHE_SLOTS> occupied{};
                    std::array<bool, 4> staged{};
                    uint32_t clock = control[C_CLOCK];
                    const auto acquisition_start = std::chrono::steady_clock::now();
                    for (uint32_t rank = 0; rank < 4; ++rank) {
                        const uint32_t expert = static_cast<uint32_t>(
                            std::lround(predicted[rank]));
                        uint32_t slot = control[C_EXPERT_TO_SLOT + expert];
                        if (slot != 0xffffffffu) {
                            occupied[slot] = true;
                            control[C_SLOT_AGE + slot] = ++clock;
                            continue;
                        }
                        bool mapped_hot = false;
                        for (uint32_t hot = 0; hot < GOSS_HOT_IMPORTS; ++hot)
                            mapped_hot = mapped_hot ||
                                hot_imports[layer][hot].expert ==
                                    static_cast<int32_t>(expert);
                        if (mapped_hot) continue;
                        uint32_t victim = 0;
                        uint32_t oldest = 0xffffffffu;
                        for (uint32_t candidate = 0; candidate < GOSS_CACHE_SLOTS;
                             ++candidate) {
                            if (!occupied[candidate] &&
                                control[C_SLOT_AGE + candidate] < oldest) {
                                oldest = control[C_SLOT_AGE + candidate];
                                victim = candidate;
                            }
                        }
                        const uint32_t evicted = control[C_SLOT_EXPERT + victim];
                        if (evicted != 0xffffffffu)
                            control[C_EXPERT_TO_SLOT + evicted] = 0xffffffffu;
                        control[C_EXPERT_TO_SLOT + expert] = victim;
                        control[C_SLOT_EXPERT + victim] = expert;
                        control[C_SLOT_AGE + victim] = ++clock;
                        occupied[victim] = true;

                        const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                            (static_cast<uint64_t>(layer) * index.header.experts +
                             expert) * GOSS_Q3_EXPERT_BLOCK;
                        PlannedCopy copy{};
                        copy.destination_offset =
                            static_cast<uint64_t>(victim) * GOSS_Q3_EXPERT_BLOCK;
                        staged[rank] = true;
                        constexpr uint32_t stripes = 8;
                        const uint64_t stripe_bytes =
                            GOSS_Q3_EXPERT_BLOCK / stripes;
                        for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                            const uint64_t offset = stripe * stripe_bytes;
                            acquisition_pool.schedule(layer,
                                expert_mapped.data() + file_offset + offset,
                                static_cast<uint8_t*>(staging[rank].mapped) + offset,
                                static_cast<size_t>(stripe_bytes));
                        }
                        copy.source = staging[rank].handle;
                        copy.source_offset = 0;
                        ++hot_import_misses;
                        planned.push_back(copy);
                    }
                    acquisition_pool.wait(layer);
                    for (uint32_t rank = 0; rank < 4; ++rank)
                        if (staged[rank]) flush_buffer(runtime, staging[rank]);
                    control[C_CLOCK] = clock;
                    flush_buffer(runtime, aux.cache_control);
                    acquisition_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - acquisition_start).count();

                    begin_command(copy_cb);
                    VkMemoryBarrier host_ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    host_ready.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    host_ready.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                               VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(copy_cb, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT |
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_ready,
                        0, nullptr, 0, nullptr);
                    for (const PlannedCopy& planned_copy : planned) {
                        VkBufferCopy copy{};
                        copy.srcOffset = planned_copy.source_offset;
                        copy.dstOffset = planned_copy.destination_offset;
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        vkfn::CmdCopyBuffer(copy_cb, planned_copy.source,
                            caches[layer].arena.handle, 1, &copy);
                    }
                    if (!planned.empty()) {
                        VkMemoryBarrier copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkfn::CmdPipelineBarrier(copy_cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copied,
                            0, nullptr, 0, nullptr);
                    }
                    VK_CHECK(vkfn::EndCommandBuffer(copy_cb));
                    copy_value = ++parallel_prefetch_value;
                    submit_signal(runtime.secondary_queue, copy_cb,
                        host_layer_timeline, copy_value, VK_NULL_HANDLE, 0);
                    token_prefetch_copies += planned.size();
                    transferred_bytes += static_cast<uint64_t>(planned.size()) *
                                         GOSS_Q3_EXPERT_BLOCK;
                } else {
                    VkSubmitInfo shared_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    shared_submit.commandBufferCount = 1;
                    shared_submit.pCommandBuffers = &shared_cb;
                    VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &shared_submit,
                                               VK_NULL_HANDLE));
                }

                begin_command(expert_cb);
                VkMemoryBarrier host_ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host_ready.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                host_ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkfn::CmdPipelineBarrier(expert_cb, VK_PIPELINE_STAGE_HOST_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &host_ready,
                    0, nullptr, 0, nullptr);
                const GossResolvePush resolve_push{0, 0, 0, 0};
                dispatch(expert_cb, resources, pipelines.cache_resolve,
                    sets.layers[layer].cache_resolve, &resolve_push, 1, 1);
                VkMemoryBarrier resolve_result{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                resolve_result.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                resolve_result.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                               VK_ACCESS_HOST_READ_BIT;
                vkfn::CmdPipelineBarrier(expert_cb,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                    VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &resolve_result,
                    0, nullptr, 0, nullptr);
                record_gpu_expert_direct(expert_cb, layer);
                if (layer + 1u == index.header.layers)
                    record_gpu_final_direct(expert_cb);
                VK_CHECK(vkfn::EndCommandBuffer(expert_cb));

                const uint64_t main_value = ++parallel_main_value;
                submit_signal(runtime.queue, expert_cb, gpu_layer_timeline,
                    main_value, layer == 0 ? VK_NULL_HANDLE : host_layer_timeline,
                    copy_value);
                wait_timeline(gpu_layer_timeline, main_value);
                invalidate_buffer(runtime, aux.cache_control);
                if (control[C_MISSING_COUNT] != 0u) {
                    invalidate_buffer(runtime, aux.cache_control);
                    const auto demand_start = std::chrono::steady_clock::now();
                    const uint32_t missing_count = control[C_MISSING_COUNT];
                    if (missing_count == 0 || missing_count > 4)
                        throw std::runtime_error("Invalid parallel cache miss count");
                    constexpr uint32_t stripes = 8;
                    const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        const uint32_t expert = control[C_MISSING_ID + item];
                        const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                            (static_cast<uint64_t>(layer) * index.header.experts +
                             expert) * GOSS_Q3_EXPERT_BLOCK;
                        for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                            const uint64_t offset = stripe * stripe_bytes;
                            acquisition_pool.schedule(layer,
                                expert_mapped.data() + file_offset + offset,
                                static_cast<uint8_t*>(staging[item].mapped) + offset,
                                static_cast<size_t>(stripe_bytes));
                        }
                    }
                    acquisition_pool.wait(layer);
                    std::array<uint32_t, 4> demand_slots{};
                    std::array<bool, GOSS_CACHE_SLOTS> demand_occupied{};
                    for (uint32_t rank = 0; rank < 4; ++rank) {
                        const uint32_t slot = control[C_SELECTED_SLOT + rank];
                        if (slot != 0xffffffffu) demand_occupied[slot] = true;
                    }
                    uint32_t clock = control[C_CLOCK];
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        flush_buffer(runtime, staging[item]);
                        uint32_t victim = 0;
                        uint32_t oldest = 0xffffffffu;
                        for (uint32_t candidate = 0; candidate < GOSS_CACHE_SLOTS;
                             ++candidate) {
                            if (!demand_occupied[candidate] &&
                                control[C_SLOT_AGE + candidate] < oldest) {
                                oldest = control[C_SLOT_AGE + candidate];
                                victim = candidate;
                            }
                        }
                        const uint32_t expert = control[C_MISSING_ID + item];
                        const uint32_t rank = control[C_MISSING_SLOT + item];
                        const uint32_t evicted = control[C_SLOT_EXPERT + victim];
                        if (evicted != 0xffffffffu)
                            control[C_EXPERT_TO_SLOT + evicted] = 0xffffffffu;
                        control[C_EXPERT_TO_SLOT + expert] = victim;
                        control[C_SLOT_EXPERT + victim] = expert;
                        control[C_SLOT_AGE + victim] = ++clock;
                        control[C_SELECTED_SLOT + rank] = victim;
                        control[C_NEEDS_COPY + rank] = 0;
                        store_control_address(control, C_SELECTED_ADDRESS + rank * 2u,
                            cache_addresses[layer] +
                            static_cast<uint64_t>(victim) * GOSS_Q3_EXPERT_BLOCK);
                        demand_occupied[victim] = true;
                        demand_slots[item] = victim;
                    }
                    control[C_CLOCK] = clock;
                    control[C_MISSING_COUNT] = 0;
                    flush_buffer(runtime, aux.cache_control);

                    begin_command(copy_cb);
                    VkMemoryBarrier demand_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    demand_host.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    demand_host.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                    vkfn::CmdPipelineBarrier(copy_cb, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &demand_host,
                        0, nullptr, 0, nullptr);
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        VkBufferCopy copy{};
                        copy.dstOffset = static_cast<uint64_t>(demand_slots[item]) *
                                         GOSS_Q3_EXPERT_BLOCK;
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        vkfn::CmdCopyBuffer(copy_cb, staging[item].handle,
                            caches[layer].arena.handle, 1, &copy);
                    }
                    VkMemoryBarrier demand_copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    demand_copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    demand_copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(copy_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &demand_copied,
                        0, nullptr, 0, nullptr);
                    VK_CHECK(vkfn::EndCommandBuffer(copy_cb));
                    const uint64_t demand_copy_value = ++parallel_prefetch_value;
                    submit_signal(runtime.secondary_queue, copy_cb,
                        host_layer_timeline, demand_copy_value, VK_NULL_HANDLE, 0);

                    begin_command(expert_cb);
                    VkMemoryBarrier demand_control{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    demand_control.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                    demand_control.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(expert_cb, VK_PIPELINE_STAGE_HOST_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &demand_control,
                        0, nullptr, 0, nullptr);
                    record_gpu_expert_direct(expert_cb, layer);
                    if (layer + 1u == index.header.layers)
                        record_gpu_final_direct(expert_cb);
                    VK_CHECK(vkfn::EndCommandBuffer(expert_cb));
                    const uint64_t recompute_value = ++parallel_main_value;
                    submit_signal(runtime.queue, expert_cb, gpu_layer_timeline,
                        recompute_value, host_layer_timeline, demand_copy_value);
                    wait_timeline(gpu_layer_timeline, recompute_value);
                    acquisition_seconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - demand_start).count();
                    token_actual_cold += missing_count;
                    gpu_layer_cold[layer] += missing_count;
                    transferred_bytes += static_cast<uint64_t>(missing_count) *
                                         GOSS_Q3_EXPERT_BLOCK;
                }
                invalidate_buffer(runtime, aux.cache_control);
                invalidate_buffer(runtime, aux.routing);
                const float* actual = static_cast<const float*>(aux.routing.mapped) +
                    static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS;
                const float* predicted = static_cast<const float*>(
                    aux.prediction_routing.mapped) +
                    static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS;
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    const uint32_t actual_expert = static_cast<uint32_t>(
                        std::lround(actual[rank]));
                    gpu_route_trace.push_back(actual_expert);
                    const uint32_t mode = control[C_NEEDS_COPY + rank];
                    token_actual_hot += mode == 3u;
                    if (layer != 0) {
                        early_rank_matches += static_cast<uint32_t>(
                            std::lround(predicted[rank])) == actual_expert;
                        bool in_set = false;
                        for (uint32_t candidate = 0; candidate < 4; ++candidate)
                            in_set = in_set || static_cast<uint32_t>(std::lround(
                                predicted[candidate])) == actual_expert;
                        early_top4_matches += in_set;
                        ++early_comparisons;
                    }
                }
            }
            const uint64_t actual_misses = token_actual_hot + token_actual_cold;
            misses += actual_misses;
            hits += static_cast<uint64_t>(index.header.layers) * 4u - actual_misses;
            hot_import_hits += token_actual_hot;
            expert_read_seconds += acquisition_seconds;
            gpu_cache_acquisition_seconds += acquisition_seconds;
            const double token_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - token_start).count();
            gpu_submission_seconds += token_seconds;
            (void)token_prefetch_copies;
            invalidate_buffer(runtime, aux.token_parameter);
        };

        const auto run_token_whole_host = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            execute([&](VkCommandBuffer cb) {
                const GossEmbeddingPush embedding_push{
                    shared_index.header.vocabulary, shared_index.header.dimension,
                    shared_index.embedding.packed_stride, 0};
                dispatch(cb, resources, pipelines.embedding, sets.embedding,
                    &embedding_push, (index.header.dimension + 63u) / 64u, 1);
                compute_buffer_barrier(cb, aux.hidden_a);
                const GossResolvePush resolve_push{0, 0, 0, 0};
                for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                    record_shared_layer(cb, layer, position);
                    compute_buffer_barrier(cb, aux.routing);
                    dispatch(cb, resources, pipelines.cache_resolve,
                        sets.layers[layer].cache_resolve, &resolve_push, 1, 1);
                    VkMemoryBarrier resolved{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    resolved.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                    resolved.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &resolved,
                        0, nullptr, 0, nullptr);
                    record_gpu_expert_direct(cb, layer);
                }
                record_gpu_final_direct(cb);
            });
            invalidate_buffer(runtime, aux.token_parameter);
        };

        std::vector<std::array<uint32_t, 4>> abort_prefetch_routes(
            index.header.layers);
        bool have_abort_prefetch_routes = true;
        invalidate_buffer(runtime, aux.routing);
        {
            const float* routes = static_cast<const float*>(aux.routing.mapped);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer)
                for (uint32_t rank = 0; rank < 4; ++rank)
                    abort_prefetch_routes[layer][rank] =
                        static_cast<uint32_t>(std::lround(routes[
                            static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS + rank]));
        }

        struct AbortPrefetchTask { uint32_t layer, expert, slot; };
        const auto prefetch_abort_routes = [&]() {
            if (!have_abort_prefetch_routes) return;
            const auto prefetch_start = std::chrono::steady_clock::now();
            std::vector<AbortPrefetchTask> tasks;
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                uint32_t* control = layer_control(layer);
                std::array<bool, GOSS_CACHE_SLOTS> occupied{};
                uint32_t clock = control[C_CLOCK];
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    const uint32_t expert = abort_prefetch_routes[layer][rank];
                    const uint64_t host =
                        static_cast<uint64_t>(control[C_HOST_ADDRESS + expert * 2u]) |
                        (static_cast<uint64_t>(
                            control[C_HOST_ADDRESS + expert * 2u + 1u]) << 32u);
                    if (host != 0) continue;
                    uint32_t slot = control[C_EXPERT_TO_SLOT + expert];
                    if (slot != 0xffffffffu) {
                        occupied[slot] = true;
                        control[C_SLOT_AGE + slot] = ++clock;
                        continue;
                    }
                    uint32_t victim = 0;
                    uint32_t oldest = 0xffffffffu;
                    for (uint32_t candidate = 0; candidate < GOSS_CACHE_SLOTS;
                         ++candidate) {
                        if (!occupied[candidate] &&
                            control[C_SLOT_AGE + candidate] < oldest) {
                            oldest = control[C_SLOT_AGE + candidate];
                            victim = candidate;
                        }
                    }
                    const uint32_t evicted = control[C_SLOT_EXPERT + victim];
                    if (evicted != 0xffffffffu)
                        control[C_EXPERT_TO_SLOT + evicted] = 0xffffffffu;
                    control[C_EXPERT_TO_SLOT + expert] = victim;
                    control[C_SLOT_EXPERT + victim] = expert;
                    control[C_SLOT_AGE + victim] = ++clock;
                    occupied[victim] = true;
                    tasks.push_back({layer, expert, victim});
                }
                control[C_CLOCK] = clock;
            }
            flush_buffer(runtime, aux.cache_control);

            for (size_t begin = 0; begin < tasks.size(); begin += 4u) {
                const uint32_t count = static_cast<uint32_t>(
                    std::min<size_t>(4u, tasks.size() - begin));
                constexpr uint32_t stripes = 8;
                const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                for (uint32_t item = 0; item < count; ++item) {
                    const AbortPrefetchTask& task = tasks[begin + item];
                    const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                        (static_cast<uint64_t>(task.layer) * index.header.experts +
                         task.expert) * GOSS_Q3_EXPERT_BLOCK;
                    for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                        const uint64_t offset = stripe * stripe_bytes;
                        acquisition_pool.schedule(task.layer,
                            expert_mapped.data() + file_offset + offset,
                            static_cast<uint8_t*>(staging[item].mapped) + offset,
                            static_cast<size_t>(stripe_bytes));
                    }
                }
                for (uint32_t item = 0; item < count; ++item)
                    acquisition_pool.wait(tasks[begin + item].layer);
                for (uint32_t item = 0; item < count; ++item)
                    flush_buffer(runtime, staging[item]);
                execute([&](VkCommandBuffer cb) {
                    for (uint32_t item = 0; item < count; ++item) {
                        const AbortPrefetchTask& task = tasks[begin + item];
                        VkBufferCopy copy{};
                        copy.dstOffset = static_cast<uint64_t>(task.slot) *
                                         GOSS_Q3_EXPERT_BLOCK;
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        vkfn::CmdCopyBuffer(cb, staging[item].handle,
                            caches[task.layer].arena.handle, 1, &copy);
                    }
                    VkMemoryBarrier copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &copied,
                        0, nullptr, 0, nullptr);
                });
            }
            transferred_bytes += static_cast<uint64_t>(tasks.size()) *
                                 GOSS_Q3_EXPERT_BLOCK;
            expert_read_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - prefetch_start).count();
        };

        // Finite abort/resume scheduler.  A cache or mapped-hot hit continues
        // entirely on the GPU.  An actual cold miss zeros the remaining
        // indirect dispatches, allowing the submission to finish normally;
        // the host then fills the reserved slots and resumes at that expert.
        // No shader ever waits for the CPU.
        const auto run_token_abort = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                uint32_t* control = layer_control(layer);
                control[C_MISSING_COUNT] = 0;
                for (uint32_t rank = 0; rank < 4; ++rank)
                    control[C_NEEDS_COPY + rank] = 0;
            }
            flush_buffer(runtime, aux.cache_control);

            uint32_t start_layer = 0;
            bool resume_at_expert = false;
            bool include_embedding = true;
            uint64_t token_copies = 0;
            uint64_t token_cold = 0;
            uint32_t pending_copy_layer = 0;
            uint32_t pending_copy_count = 0;
            for (;;) {
                std::memset(indirect_commands, 0,
                    GOSS_MAX_INDIRECT_COMMANDS * sizeof(VkDispatchIndirectCommand));
                execute([&](VkCommandBuffer cb) {
                    uint32_t cursor = 0;
                    if (pending_copy_count != 0) {
                        const uint32_t* pending_control =
                            layer_control(pending_copy_layer);
                        for (uint32_t item = 0; item < pending_copy_count; ++item) {
                            VkBufferCopy copy{};
                            copy.dstOffset = static_cast<uint64_t>(
                                pending_control[C_MISSING_SLOT + item]) *
                                GOSS_Q3_EXPERT_BLOCK;
                            copy.size = GOSS_Q3_EXPERT_BLOCK;
                            vkfn::CmdCopyBuffer(cb, staging[item].handle,
                                caches[pending_copy_layer].arena.handle, 1, &copy);
                        }
                        VkMemoryBarrier copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                        copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                        vkfn::CmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_TRANSFER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            0, 1, &copied, 0, nullptr, 0, nullptr);
                    }
                    if (include_embedding) {
                        const GossEmbeddingPush embedding_push{
                            shared_index.header.vocabulary,
                            shared_index.header.dimension,
                            shared_index.embedding.packed_stride, 0};
                        indirect_dispatch(cb, cursor, pipelines.embedding,
                            sets.embedding, &embedding_push,
                            (index.header.dimension + 63u) / 64u, 1);
                        compute_buffer_barrier(cb, aux.hidden_a);
                    }
                    uint32_t layer = start_layer;
                    if (resume_at_expert) {
                        record_gpu_expert(cb, cursor, layer);
                        ++layer;
                    }
                    for (; layer < index.header.layers; ++layer) {
                        record_shared_layer_indirect(cb, cursor, layer, position);
                        const uint32_t resolver_command = cursor;
                        const GossResolvePush resolve_push{
                            resolver_command + 1u, resolver_command + 2u,
                            GOSS_MAX_INDIRECT_COMMANDS, 0};
                        indirect_dispatch(cb, cursor, pipelines.cache_resolve_stop,
                            sets.layers[layer].cache_resolve,
                            &resolve_push, 1, 1);
                        VkMemoryBarrier resolve_barrier{
                            VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                        resolve_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                        resolve_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
                        vkfn::CmdPipelineBarrier(cb,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                            0, 1, &resolve_barrier, 0, nullptr, 0, nullptr);
                        const GossFusedExpertPush no_push{{0, 0, 0, 0}};
                        indirect_dispatch(cb, cursor,
                            pipelines.cache_copy_finite,
                            sets.layers[layer].cache_copy,
                            &no_push, 8, 4);
                        compute_buffer_barrier(cb, caches[layer].arena);
                        record_gpu_expert(cb, cursor, layer);
                    }
                    record_gpu_final(cb, cursor);
                });
                pending_copy_count = 0;

                invalidate_buffer(runtime, aux.cache_control);
                int32_t missed_layer = -1;
                for (uint32_t layer = start_layer; layer < index.header.layers;
                     ++layer) {
                    if (layer_control(layer)[C_MISSING_COUNT] != 0u) {
                        missed_layer = static_cast<int32_t>(layer);
                        break;
                    }
                }
                if (missed_layer < 0) break;

                const uint32_t layer = static_cast<uint32_t>(missed_layer);
                uint32_t* control = layer_control(layer);
                const uint32_t missing_count = control[C_MISSING_COUNT];
                if (missing_count == 0 || missing_count > 4)
                    throw std::runtime_error("Invalid finite cache miss count");
                const auto acquisition_start = std::chrono::steady_clock::now();
                constexpr uint32_t stripes = 8;
                const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                for (uint32_t item = 0; item < missing_count; ++item) {
                    const uint32_t expert = control[C_MISSING_ID + item];
                    const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                        (static_cast<uint64_t>(layer) * index.header.experts +
                         expert) * GOSS_Q3_EXPERT_BLOCK;
                    for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                        const uint64_t offset = stripe * stripe_bytes;
                        acquisition_pool.schedule(layer,
                            expert_mapped.data() + file_offset + offset,
                            static_cast<uint8_t*>(staging[item].mapped) + offset,
                            static_cast<size_t>(stripe_bytes));
                    }
                }
                acquisition_pool.wait(layer);
                for (uint32_t item = 0; item < missing_count; ++item)
                    flush_buffer(runtime, staging[item]);
                token_cold += missing_count;
                token_copies += missing_count;
                control[C_MISSING_COUNT] = 0;
                for (uint32_t rank = 0; rank < 4; ++rank)
                    control[C_NEEDS_COPY + rank] = 0;
                flush_buffer(runtime, aux.cache_control);
                expert_read_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - acquisition_start).count();
                start_layer = layer;
                resume_at_expert = true;
                include_embedding = false;
                pending_copy_layer = layer;
                pending_copy_count = missing_count;
            }
            transferred_bytes += token_cold * GOSS_Q3_EXPERT_BLOCK;
            misses += token_cold;
            hits += static_cast<uint64_t>(index.header.layers) * 4u - token_cold;
            hot_import_misses += token_cold;
            invalidate_buffer(runtime, aux.token_parameter);
            invalidate_buffer(runtime, aux.routing);
            const float* routes = static_cast<const float*>(aux.routing.mapped);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                std::array<uint32_t, 4> actual{};
                for (uint32_t rank = 0; rank < 4; ++rank)
                    actual[rank] = static_cast<uint32_t>(std::lround(routes[
                        static_cast<uint64_t>(layer) * GOSS_ROUTE_FLOATS + rank]));
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    previous_route_rank_matches +=
                        actual[rank] == abort_prefetch_routes[layer][rank];
                    previous_route_set_matches += std::find(
                        abort_prefetch_routes[layer].begin(),
                        abort_prefetch_routes[layer].end(), actual[rank]) !=
                        abort_prefetch_routes[layer].end();
                    ++previous_route_comparisons;
                }
                abort_prefetch_routes[layer] = actual;
            }
            have_abort_prefetch_routes = true;
        };

        const auto run_token_gpu = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                uint32_t* control = layer_control(layer);
                control[C_MISSING_COUNT] = 0;
                control[C_REQUEST_STATE] = 0;
                for (uint32_t rank = 0; rank < 4; ++rank)
                    control[C_NEEDS_COPY + rank] = 0;
            }
            flush_buffer(runtime, aux.cache_control);

            uint32_t start_layer = 0;
            bool start_at_expert = false;
            bool include_embedding = true;
            uint64_t token_cold_misses = 0;
            for (;;) {
                std::memset(indirect_commands, 0,
                    GOSS_MAX_INDIRECT_COMMANDS * sizeof(VkDispatchIndirectCommand));
                double cooperative_acquisition = 0.0;
                std::thread cache_service([&] {
                    for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                        uint32_t* control = layer_control(layer);
                        auto* state = reinterpret_cast<volatile LONG*>(
                            control + C_REQUEST_STATE);
                        LONG observed = 0;
                        while ((observed = InterlockedCompareExchange(state, 0, 0)) == 0)
                            YieldProcessor();
                        if (observed != 2) continue;
                        const auto request_start = std::chrono::steady_clock::now();
                        const uint32_t missing_count = control[C_MISSING_COUNT];
                        if (missing_count == 0 || missing_count > 4) {
                            InterlockedExchange(state, -1);
                            return;
                        }
                        constexpr uint32_t stripes = 8;
                        const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                        for (uint32_t item = 0; item < missing_count; ++item) {
                            const uint32_t expert = control[C_MISSING_ID + item];
                            const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                                (static_cast<uint64_t>(layer) * index.header.experts +
                                 expert) * GOSS_Q3_EXPERT_BLOCK;
                            for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                                const uint64_t offset = stripe * stripe_bytes;
                                acquisition_pool.schedule(layer,
                                    expert_mapped.data() + file_offset + offset,
                                    static_cast<uint8_t*>(staging[item].mapped) + offset,
                                    static_cast<size_t>(stripe_bytes));
                            }
                        }
                        acquisition_pool.wait(layer);
                        for (uint32_t item = 0; item < missing_count; ++item) {
                            flush_buffer(runtime, staging[item]);
                            const uint32_t missing_slot = control[C_MISSING_SLOT + item];
                            bool found_rank = false;
                            for (uint32_t rank = 0; rank < 4; ++rank) {
                                if (control[C_SELECTED_SLOT + rank] == missing_slot) {
                                    store_control_address(control,
                                        C_COLD_ADDRESS + rank * 2u,
                                        staging_addresses[item]);
                                    found_rank = true;
                                    break;
                                }
                            }
                            if (!found_rank) {
                                InterlockedExchange(state, -1);
                                return;
                            }
                        }
                        MemoryBarrier();
                        InterlockedExchange(state, 3);
                        while ((observed = InterlockedCompareExchange(state, 0, 0)) != 4) {
                            if (observed < 0) return;
                            YieldProcessor();
                        }
                        cooperative_acquisition += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - request_start).count();
                    }
                });
                execute([&](VkCommandBuffer cb) {
                    uint32_t cursor = 0;
                    if (include_embedding) {
                        const GossEmbeddingPush embedding_push{
                            shared_index.header.vocabulary,
                            shared_index.header.dimension,
                            shared_index.embedding.packed_stride, 0};
                        indirect_dispatch(cb, cursor, pipelines.embedding, sets.embedding,
                            &embedding_push, (index.header.dimension + 63u) / 64u, 1);
                        compute_buffer_barrier(cb, aux.hidden_a);
                    }
                    uint32_t layer = start_layer;
                    if (start_at_expert) {
                        record_gpu_expert(cb, cursor, layer);
                        ++layer;
                    }
                    for (; layer < index.header.layers; ++layer)
                        record_gpu_layer(cb, cursor, layer, position);
                    record_gpu_final(cb, cursor);
                });
                cache_service.join();
                invalidate_buffer(runtime, aux.cache_control);

                bool service_failed = false;
                for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                    const uint32_t* control = layer_control(layer);
                    service_failed = service_failed ||
                        static_cast<int32_t>(control[C_REQUEST_STATE]) < 0;
                    const uint32_t count = control[C_MISSING_COUNT];
                    token_cold_misses += count;
                    gpu_layer_cold[layer] += count;
                }
                if (service_failed)
                    throw std::runtime_error("Cooperative GPU cache service failed");
                transferred_bytes += token_cold_misses * GOSS_Q3_EXPERT_BLOCK;
                gpu_cache_acquisition_seconds += cooperative_acquisition;
                expert_read_seconds += cooperative_acquisition;
                break;

                int32_t missed_layer = -1;
                for (uint32_t layer = start_at_expert ? start_layer + 1u : start_layer;
                     layer < index.header.layers; ++layer) {
                    if (layer_control(layer)[C_MISSING_COUNT] != 0u) {
                        missed_layer = static_cast<int32_t>(layer);
                        break;
                    }
                }
                if (missed_layer < 0) break;

                uint32_t* control = layer_control(static_cast<uint32_t>(missed_layer));
                const uint32_t missing_count = control[C_MISSING_COUNT];
                if (missing_count == 0 || missing_count > 4)
                    throw std::runtime_error("Invalid GPU expert-cache miss record");
                const auto acquisition_start = std::chrono::steady_clock::now();
                constexpr uint32_t stripes = 8;
                const uint64_t stripe_bytes = GOSS_Q3_EXPERT_BLOCK / stripes;
                for (uint32_t item = 0; item < missing_count; ++item) {
                    const uint32_t expert = control[C_MISSING_ID + item];
                    const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                        (static_cast<uint64_t>(missed_layer) * index.header.experts + expert) *
                        GOSS_Q3_EXPERT_BLOCK;
                    for (uint32_t stripe = 0; stripe < stripes; ++stripe) {
                        const uint64_t offset = stripe * stripe_bytes;
                        acquisition_pool.schedule(static_cast<uint32_t>(missed_layer),
                            expert_mapped.data() + file_offset + offset,
                            static_cast<uint8_t*>(staging[item].mapped) + offset,
                            static_cast<size_t>(stripe_bytes));
                    }
                }
                acquisition_pool.wait(static_cast<uint32_t>(missed_layer));
                for (uint32_t item = 0; item < missing_count; ++item)
                    flush_buffer(runtime, staging[item]);
                execute([&](VkCommandBuffer cb) {
                    for (uint32_t item = 0; item < missing_count; ++item) {
                        VkBufferCopy copy{};
                        copy.srcOffset = 0;
                        copy.dstOffset = static_cast<uint64_t>(
                            control[C_MISSING_SLOT + item]) * GOSS_Q3_EXPERT_BLOCK;
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        vkfn::CmdCopyBuffer(cb, staging[item].handle,
                            caches[missed_layer].arena.handle, 1, &copy);
                    }
                    VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    transfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &transfer,
                        0, nullptr, 0, nullptr);
                });
                token_cold_misses += missing_count;
                gpu_layer_cold[missed_layer] += missing_count;
                transferred_bytes += static_cast<uint64_t>(missing_count) *
                                     GOSS_Q3_EXPERT_BLOCK;
                control[C_MISSING_COUNT] = 0;
                flush_buffer(runtime, aux.cache_control);
                const double acquisition_elapsed = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - acquisition_start).count();
                gpu_cache_acquisition_seconds += acquisition_elapsed;
                expert_read_seconds += acquisition_elapsed;
                start_layer = static_cast<uint32_t>(missed_layer);
                start_at_expert = true;
                include_embedding = false;
            }
            invalidate_buffer(runtime, aux.token_parameter);
            invalidate_buffer(runtime, aux.cache_control);
            invalidate_buffer(runtime, aux.routing);
            const float* traced_routes = static_cast<const float*>(aux.routing.mapped);
            for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
                std::array<uint32_t, 4> current{};
                for (uint32_t rank = 0; rank < index.header.top_k; ++rank) {
                    current[rank] = static_cast<uint32_t>(std::lround(
                        traced_routes[static_cast<uint64_t>(layer) *
                                      GOSS_ROUTE_FLOATS + rank]));
                    gpu_route_trace.push_back(current[rank]);
                    if (have_previous_gpu_routes) {
                        previous_route_rank_matches +=
                            current[rank] == previous_gpu_routes[layer][rank];
                        previous_route_set_matches += std::find(
                            previous_gpu_routes[layer].begin(),
                            previous_gpu_routes[layer].end(), current[rank]) !=
                            previous_gpu_routes[layer].end();
                        ++previous_route_comparisons;
                    }
                }
                previous_gpu_routes[layer] = current;
            }
            have_previous_gpu_routes = true;
            uint64_t token_hot_copies = 0;
            for (uint32_t layer = 0; layer < index.header.layers; ++layer)
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    const bool copied = layer_control(layer)[C_NEEDS_COPY + rank] == 1u;
                    token_hot_copies += copied;
                    gpu_layer_hot_copy[layer] += copied;
                }
            const uint64_t token_misses = token_hot_copies + token_cold_misses;
            hits += static_cast<uint64_t>(index.header.layers) * index.header.top_k -
                    token_misses;
            misses += token_misses;
            hot_import_hits += token_hot_copies;
            hot_import_misses += token_cold_misses;
            transferred_bytes += token_hot_copies * GOSS_Q3_EXPERT_BLOCK;
        };

        auto* slot_maps = reinterpret_cast<uint32_t*>(aux.slot_maps.mapped);

        struct GossPromotion { uint32_t layer, expert, slot; };
        std::vector<GossPromotion> pending_promotions;

        const auto run_token_bulk = [&](uint32_t token, uint32_t position) {
            *static_cast<uint32_t*>(aux.token_parameter.mapped) = token;
            flush_buffer(runtime, aux.token_parameter);
            flush_buffer(runtime, aux.slot_maps);
            execute([&](VkCommandBuffer cb) {
                if (!pending_promotions.empty()) {
                    for (const GossPromotion& promotion : pending_promotions) {
                        const uint64_t file_offset = GOSS_Q3_EXPERT_HEADER +
                            (static_cast<uint64_t>(promotion.layer) * index.header.experts +
                             promotion.expert) * GOSS_Q3_EXPERT_BLOCK;
                        VkBufferCopy copy{};
                        copy.srcOffset = file_offset -
                            layer_imports[promotion.layer][promotion.expert >> 4u].file_offset;
                        copy.dstOffset = static_cast<uint64_t>(promotion.slot) *
                            GOSS_Q3_EXPERT_BLOCK;
                        copy.size = GOSS_Q3_EXPERT_BLOCK;
                        vkfn::CmdCopyBuffer(cb,
                            layer_imports[promotion.layer][promotion.expert >> 4u].buffer.handle,
                            caches[promotion.layer].arena.handle, 1, &copy);
                        transferred_bytes += GOSS_Q3_EXPERT_BLOCK;
                    }
                    VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                    transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                    transfer.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &transfer,
                        0, nullptr, 0, nullptr);
                }
                const GossEmbeddingPush embedding_push{shared_index.header.vocabulary,
                    shared_index.header.dimension, shared_index.embedding.packed_stride, 0};
                dispatch(cb, resources, pipelines.embedding, sets.embedding,
                         &embedding_push, (index.header.dimension + 63u) / 64u, 1);
                compute_buffer_barrier(cb, aux.hidden_a);
                const GossFusedExpertPush no_push{{0, 0, 0, 0}};
                for (uint32_t number = 0; number < index.header.layers; ++number) {
                    record_shared_layer(cb, number, position);
                    compute_buffer_barrier(cb, aux.routing);
                    dispatch(cb, resources, pipelines.expert_gate_dynamic,
                             dynamic_experts[number].gate, &no_push,
                             (index.header.hidden_dimension + 3u) / 4u, 4);
                    compute_buffer_barrier(cb, aux.feed_forward);
                    dispatch(cb, resources, pipelines.expert_down_dynamic,
                             dynamic_experts[number].down, &no_push,
                             (index.header.dimension + 3u) / 4u, 1);
                    compute_buffer_barrier(cb, aux.moe_sum);
                    dispatch(cb, resources, pipelines.add,
                             sets.layers[number].mlp_residual, &add_push,
                             (index.header.dimension + 63u) / 64u, 1);
                    compute_buffer_barrier(cb, aux.hidden_a);
                }
                dispatch(cb, resources, pipelines.rmsnorm, sets.final_norm,
                         &norm_push, 1, 1);
                compute_buffer_barrier(cb, aux.norm);
                qgemv(cb, sets.logits, shared_index.unembedding, false);
                compute_buffer_barrier(cb, aux.logits);
                constexpr uint32_t greedy_groups = 256;
                const GossGreedyPush greedy_push{
                    index.header.vocabulary, greedy_groups, 0, 0};
                dispatch(cb, resources, pipelines.greedy, sets.greedy,
                         &greedy_push, greedy_groups, 1);
                compute_buffer_barrier(cb, aux.feed_forward);
                const GossGreedyPush greedy_finish{
                    index.header.vocabulary, greedy_groups, 1, 0};
                dispatch(cb, resources, pipelines.greedy, sets.greedy,
                         &greedy_finish, 1, 1);
            });
            pending_promotions.clear();
            invalidate_buffer(runtime, aux.token_parameter);
            invalidate_buffer(runtime, aux.routing);
        };

        const auto plan_promotions = [&]() {
            const auto acquisition_start = std::chrono::steady_clock::now();
            const float* all_routes = static_cast<const float*>(aux.routing.mapped);
            for (uint32_t number = 0; number < index.header.layers; ++number) {
                const float* values = all_routes +
                    static_cast<uint64_t>(number) * GOSS_ROUTE_FLOATS;
                std::array<uint32_t, 4> experts{};
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    experts[rank] = static_cast<uint32_t>(std::lround(values[rank]));
                    if (experts[rank] >= index.header.experts)
                        throw std::runtime_error("Invalid bulk GPT-OSS route");
                }
                GossLayerCache& cache = caches[number];
                std::array<bool, GOSS_CACHE_SLOTS> occupied{};
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    bool found = false;
                    for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                        if (cache.slots[slot].expert == static_cast<int32_t>(experts[rank])) {
                            cache.slots[slot].last_used = ++cache_clock;
                            occupied[slot] = true;
                            ++hits;
                            found = true;
                            break;
                        }
                    }
                    if (found) continue;
                    uint32_t victim = 0;
                    uint64_t oldest = std::numeric_limits<uint64_t>::max();
                    for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                        if (!occupied[slot] && cache.slots[slot].last_used < oldest) {
                            oldest = cache.slots[slot].last_used;
                            victim = slot;
                        }
                    }
                    GossSlot& target = cache.slots[victim];
                    if (target.expert >= 0)
                        slot_maps[static_cast<uint64_t>(number) * index.header.experts +
                            target.expert] = 0xffffffffu;
                    target.expert = static_cast<int32_t>(experts[rank]);
                    target.last_used = ++cache_clock;
                    slot_maps[static_cast<uint64_t>(number) * index.header.experts +
                        experts[rank]] = victim;
                    occupied[victim] = true;
                    pending_promotions.push_back({number, experts[rank], victim});
                    ++misses;
                }
            }
            expert_read_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - acquisition_start).count();
        };

        const uint64_t initial_hits = hits, initial_misses = misses;
        const uint64_t initial_transferred = transferred_bytes;
        const uint64_t initial_hot_hits = hot_import_hits;
        const uint64_t initial_hot_misses = hot_import_misses;
        const double initial_gpu_seconds = gpu_submission_seconds;
        const double initial_read_seconds = expert_read_seconds;
        char profile_flag[2]{};
        dispatch_profile_enabled =
            GetEnvironmentVariableA("GOSS_PROFILE", profile_flag, 2) != 0;
        const auto start = std::chrono::steady_clock::now();
        std::vector<uint32_t> generated;
        for (uint32_t count = 0; count < generation_limit; ++count) {
            const uint32_t next = *static_cast<const uint32_t*>(aux.token_parameter.mapped);
            if (next >= index.header.vocabulary)
                throw std::runtime_error("GPU argmax returned an invalid token");
            generated.push_back(next);
            tokens.push_back(next);
            if (next == tokenizer.eos()) break;
            run_token_abort(next, static_cast<uint32_t>(tokens.size() - 1));
        }
        dispatch_profile_enabled = false;
        if (profile_submissions != 0) {
            const std::array<const char*, 12> names{{"embedding", "norm", "shared_qgemv",
                "rope", "attention", "router", "quantize", "expert_gate",
                "expert_down", "add", "greedy", "other"}};
            std::cerr << "GPU PROFILE (ms/token):";
            for (size_t item = 0; item < names.size(); ++item)
                std::cerr << " " << names[item] << "="
                          << profile_ms[item] / std::max<size_t>(1, generated.size());
            std::cerr << " dispatch_span="
                      << profile_span_ms / std::max<size_t>(1, generated.size())
                      << " submissions=" << profile_submissions << "\n";
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        const uint64_t measured_hits = hits - initial_hits;
        const uint64_t measured_misses = misses - initial_misses;
        const uint64_t measured_transfer = transferred_bytes - initial_transferred;
        const uint64_t measured_hot_hits = hot_import_hits - initial_hot_hits;
        const uint64_t measured_hot_misses = hot_import_misses - initial_hot_misses;
        const double measured_gpu_seconds = gpu_submission_seconds - initial_gpu_seconds;
        const double measured_read_seconds = expert_read_seconds - initial_read_seconds;

        const uint32_t traced_tokens = static_cast<uint32_t>(
            gpu_route_trace.size() /
            (static_cast<uint64_t>(index.header.layers) * index.header.top_k));
        uint64_t simulated_lru_misses = 0, simulated_lfu_misses = 0;
        uint64_t simulated_optimal_misses = 0;
        constexpr std::array<uint32_t, 36> proposed_slots{{
            30,24,20,18,20,24,18,16,12,16,8,10,12,12,22,18,16,16,
            18,22,30,16,28,20,22,16,16,16,24,24,20,26,26,22,28,28}};
        uint64_t simulated_variable_misses = 0;
        for (uint32_t layer = 0; layer < index.header.layers; ++layer) {
            std::array<int32_t, GOSS_CACHE_SLOTS> lru_cache{}, lfu_cache{}, opt_cache{};
            std::array<uint64_t, GOSS_CACHE_SLOTS> lru_age{}, lfu_age{};
            std::array<uint64_t, 128> frequency{};
            for (uint32_t expert = 0; expert < index.header.experts; ++expert)
                frequency[expert] = expert_frequency[layer][expert];
            for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                lru_cache[slot] = lfu_cache[slot] = opt_cache[slot] =
                    caches[layer].slots[slot].expert;
                lru_age[slot] = lfu_age[slot] = slot;
            }
            uint64_t clock = GOSS_CACHE_SLOTS;
            std::vector<int32_t> variable_cache(proposed_slots[layer], -1);
            std::vector<uint64_t> variable_age(proposed_slots[layer], 0);
            for (uint32_t slot = 0;
                 slot < std::min<uint32_t>(GOSS_CACHE_SLOTS, proposed_slots[layer]); ++slot) {
                variable_cache[slot] = caches[layer].slots[slot].expert;
                variable_age[slot] = slot;
            }
            for (uint32_t token_index = 0; token_index < traced_tokens; ++token_index) {
                for (uint32_t rank = 0; rank < 4; ++rank) {
                    const uint64_t trace_index =
                        (static_cast<uint64_t>(token_index) * index.header.layers + layer) *
                        index.header.top_k + rank;
                    const uint32_t expert = gpu_route_trace[trace_index];
                    auto update_lru = [&](auto& cache, auto& ages, uint64_t& miss_count,
                                          bool lfu) {
                        uint32_t found = GOSS_CACHE_SLOTS;
                        for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot)
                            if (cache[slot] == static_cast<int32_t>(expert)) found = slot;
                        if (found == GOSS_CACHE_SLOTS) {
                            ++miss_count;
                            found = 0;
                            for (uint32_t slot = 1; slot < GOSS_CACHE_SLOTS; ++slot) {
                                const uint64_t left_score = lfu && cache[slot] >= 0 ?
                                    frequency[cache[slot]] : ages[slot];
                                const uint64_t right_score = lfu && cache[found] >= 0 ?
                                    frequency[cache[found]] : ages[found];
                                if (left_score < right_score ||
                                    (left_score == right_score && ages[slot] < ages[found]))
                                    found = slot;
                            }
                            cache[found] = static_cast<int32_t>(expert);
                        }
                        ages[found] = ++clock;
                    };
                    update_lru(lru_cache, lru_age, simulated_lru_misses, false);
                    update_lru(lfu_cache, lfu_age, simulated_lfu_misses, true);
                    uint32_t variable_slot = proposed_slots[layer];
                    for (uint32_t slot = 0; slot < proposed_slots[layer]; ++slot)
                        if (variable_cache[slot] == static_cast<int32_t>(expert))
                            variable_slot = slot;
                    if (variable_slot == proposed_slots[layer]) {
                        ++simulated_variable_misses;
                        variable_slot = 0;
                        for (uint32_t slot = 1; slot < proposed_slots[layer]; ++slot)
                            if (variable_age[slot] < variable_age[variable_slot])
                                variable_slot = slot;
                        variable_cache[variable_slot] = static_cast<int32_t>(expert);
                    }
                    variable_age[variable_slot] = ++clock;
                    ++frequency[expert];

                    uint32_t opt_slot = GOSS_CACHE_SLOTS;
                    for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot)
                        if (opt_cache[slot] == static_cast<int32_t>(expert)) opt_slot = slot;
                    if (opt_slot == GOSS_CACHE_SLOTS) {
                        ++simulated_optimal_misses;
                        uint64_t farthest = 0;
                        opt_slot = 0;
                        for (uint32_t slot = 0; slot < GOSS_CACHE_SLOTS; ++slot) {
                            uint64_t next = std::numeric_limits<uint64_t>::max();
                            for (uint32_t future_token = token_index;
                                 future_token < traced_tokens; ++future_token) {
                                for (uint32_t future_rank =
                                         future_token == token_index ? rank + 1u : 0u;
                                     future_rank < 4; ++future_rank) {
                                    const uint64_t future_index =
                                        (static_cast<uint64_t>(future_token) *
                                         index.header.layers + layer) * 4u + future_rank;
                                    if (gpu_route_trace[future_index] ==
                                        static_cast<uint32_t>(opt_cache[slot])) {
                                        next = future_index;
                                        future_token = traced_tokens;
                                        break;
                                    }
                                }
                            }
                            if (next > farthest) { farthest = next; opt_slot = slot; }
                        }
                        opt_cache[opt_slot] = static_cast<int32_t>(expert);
                    }
                }
            }
        }

        Buffer final_input = create_buffer(runtime, index.header.dimension * sizeof(float));
        execute([&](VkCommandBuffer cb) {
            VkBufferCopy copy{};
            copy.size = final_input.size;
            vkfn::CmdCopyBuffer(cb, aux.norm.handle, final_input.handle, 1, &copy);
            VkMemoryBarrier transfer{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            transfer.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            transfer.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkfn::CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &transfer, 0, nullptr, 0, nullptr);
        });
        invalidate_buffer(runtime, final_input);
        const float* input_values = static_cast<const float*>(final_input.mapped);
        const float* scales = reinterpret_cast<const float*>(
            shared_mapped.data() + shared_index.unembedding.scale_offset);
        const uint32_t* weights = reinterpret_cast<const uint32_t*>(
            shared_mapped.data() + shared_index.unembedding.weight_offset);
        float lm_error = 0.0f;
        const auto check_lm_row = [&](uint32_t row) {
            double sum = 0.0;
            const uint32_t* row_weights = weights +
                static_cast<uint64_t>(row) * shared_index.unembedding.packed_stride;
            for (uint32_t column = 0; column < index.header.dimension; ++column) {
                const uint32_t packed = row_weights[column >> 3u];
                int32_t quantized = static_cast<int32_t>((packed >>
                    ((column & 7u) * 4u)) & 15u);
                if (quantized >= 8) quantized -= 16;
                sum += static_cast<double>(input_values[column]) * quantized;
            }
            const float reference = static_cast<float>(sum * scales[row]);
            lm_error = std::max(lm_error,
                std::abs(reference - static_cast<const float*>(aux.logits.mapped)[row]));
        };
        for (uint32_t row = 0; row < index.header.vocabulary; row += 997u) check_lm_row(row);
        check_lm_row(static_cast<uint32_t>(std::max_element(
            static_cast<const float*>(aux.logits.mapped),
            static_cast<const float*>(aux.logits.mapped) + index.header.vocabulary) -
            static_cast<const float*>(aux.logits.mapped)));
        if (!std::isfinite(lm_error) || lm_error > 0.05f || generated.empty()) {
            throw std::runtime_error("GPT-OSS Vulkan numerical check failed");
        }

        const double hit_rate = measured_hits + measured_misses == 0 ? 0.0 :
            100.0 * measured_hits / static_cast<double>(measured_hits + measured_misses);
        const double hot_hit_rate = measured_hot_hits + measured_hot_misses == 0 ? 0.0 :
            100.0 * measured_hot_hits /
            static_cast<double>(measured_hot_hits + measured_hot_misses);
        const double effective_hit_rate = measured_hits + measured_hot_hits +
            measured_hot_misses == 0 ? 0.0 :
            100.0 * (measured_hits + measured_hot_hits) /
            static_cast<double>(measured_hits + measured_hot_hits + measured_hot_misses);
        const uint64_t learned_residency = shared_index.shared_bytes +
            peak_populated_slots * GOSS_Q3_EXPERT_BLOCK;
        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Model parameters: " << index.header.total_parameters
                  << " (active/token " << index.header.active_parameters << ")\n"
                  << std::fixed << std::setprecision(3)
                  << "Quantized model: "
                  << (shared_index.file_bytes + expert_mapped.size()) /
                     (1024.0 * 1024.0 * 1024.0)
                  << " GiB\nPeak device learned-weight residency: "
                  << learned_residency / (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "Expert cache capacity: "
                  << (static_cast<uint64_t>(index.header.layers) * GOSS_CACHE_SLOTS *
                      GOSS_Q3_EXPERT_BLOCK / (1024.0 * 1024.0 * 1024.0))
                  << " GiB\nImported hot-expert backing: "
                  << (static_cast<uint64_t>(index.header.layers) * GOSS_HOT_IMPORTS *
                      GOSS_Q3_EXPERT_BLOCK / (1024.0 * 1024.0 * 1024.0))
                  << " GiB\nExpert transfer/token: "
                  << (measured_transfer / static_cast<double>(generated.size()) /
                      (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
                  << "CPU-staged expert read/token: "
                  << (measured_hot_misses * GOSS_Q3_EXPERT_BLOCK /
                      static_cast<double>(generated.size()) /
                      (1024.0 * 1024.0 * 1024.0)) << " GiB\n"
                  << "Cache hits/misses: " << measured_hits << "/" << measured_misses
                  << " (" << hit_rate << "% hit)\n"
                  << "Imported-hot hits/misses: " << measured_hot_hits << "/"
                  << measured_hot_misses << " (" << hot_hit_rate
                  << "% hot hit; " << effective_hit_rate << "% effective hit)\nGeneration: "
                  << generated.size() / seconds << " tok/s\n"
                  << "Measured GPU submit/wait: " << measured_gpu_seconds / generated.size()
                  << " s/token; expert read wait: "
                  << measured_read_seconds / generated.size() << " s/token\n"
                  << "Sampled LM-head CPU/GPU max abs delta: " << lm_error
                  << "\nGenerated token IDs:";
        for (uint32_t token : generated) std::cout << " " << token;
        std::cout << "\n--- generated text ---\n" << tokenizer.decode(generated)
                  << "\n--- end ---\nRESULT: PASS - GPT-OSS-120B native Vulkan MXFP4 expert streaming\n";
        std::cout << "GPU cache cold/hot-copy by layer:";
        for (uint32_t layer = 0; layer < index.header.layers; ++layer)
            std::cout << " " << layer << ":" << gpu_layer_cold[layer]
                      << "/" << gpu_layer_hot_copy[layer];
        std::cout << "\n";
        std::cout << "Previous-route predictor rank/set recall: "
                  << (100.0 * previous_route_rank_matches /
                      std::max<uint64_t>(1, previous_route_comparisons)) << "/"
                   << (100.0 * previous_route_set_matches /
                       std::max<uint64_t>(1, previous_route_comparisons)) << "%\n";
        std::cout << "Previous-router top8/top16 recall: "
                  << (100.0 * previous_top8_matches /
                      std::max<uint64_t>(1, previous_top16_comparisons)) << "/"
                  << (100.0 * previous_top16_matches /
                      std::max<uint64_t>(1, previous_top16_comparisons))
                  << "%; cold-only "
                  << (100.0 * previous_cold_top8_matches /
                      std::max<uint64_t>(1, previous_cold_comparisons)) << "/"
                  << (100.0 * previous_cold_top16_matches /
                      std::max<uint64_t>(1, previous_cold_comparisons)) << "%\n";
        std::cout << "Early-router rank/top4/top8/top16 recall: "
                  << (100.0 * early_rank_matches /
                      std::max<uint64_t>(1, early_comparisons)) << "/"
                  << (100.0 * early_top4_matches /
                      std::max<uint64_t>(1, early_comparisons)) << "/"
                  << (100.0 * early_top8_matches /
                      std::max<uint64_t>(1, early_comparisons)) << "/"
                  << (100.0 * early_top16_matches /
                      std::max<uint64_t>(1, early_comparisons))
                  << "%; cold-only "
                  << (100.0 * early_cold_top4_matches /
                      std::max<uint64_t>(1, early_cold_comparisons)) << "/"
                  << (100.0 * early_cold_top8_matches /
                      std::max<uint64_t>(1, early_cold_comparisons)) << "/"
                  << (100.0 * early_cold_top16_matches /
                      std::max<uint64_t>(1, early_cold_comparisons)) << "%\n";
        std::cout << "Simulated LRU/LFU/optimal cache misses: "
                  << simulated_lru_misses << "/" << simulated_lfu_misses << "/"
                  << simulated_optimal_misses << "; variable-slot LRU "
                  << simulated_variable_misses << "\n";
        std::cout << "GPU route trace:";
        for (uint32_t expert : gpu_route_trace) std::cout << " " << expert;
        std::cout << "\n";

        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));
        destroy_buffer(runtime, final_input);
        vkfn::DestroyFence(runtime.device, fence, nullptr);
        vkfn::DestroySemaphore(runtime.device, timeline, nullptr);
        vkfn::DestroySemaphore(runtime.device, gpu_layer_timeline, nullptr);
        vkfn::DestroySemaphore(runtime.device, host_layer_timeline, nullptr);
        vkfn::DestroyQueryPool(runtime.device, dispatch_profile_pool, nullptr);
        dispatch_profile_pool = VK_NULL_HANDLE;
        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines)
            vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        for (VkShaderModule module : resources.shader_modules)
            vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        for (auto& layer : layer_imports)
            for (GossImportedRange& imported : layer)
                destroy_buffer(runtime, imported.buffer);
        for (auto& layer : hot_imports)
            for (GossHotImport& hot : layer)
                destroy_buffer(runtime, hot.imported.buffer);
        for (Buffer& buffer : staging) destroy_buffer(runtime, buffer);
        destroy_buffer(runtime, zero_expert);
        for (GossLayerCache& cache : caches) destroy_buffer(runtime, cache.arena);
        for (Buffer& state : aux.attention_states) destroy_buffer(runtime, state);
        std::array<Buffer*, 20> buffers{&aux.rope_sin, &aux.rope_cos, &aux.logits,
            &aux.routing, &aux.prediction_routing, &aux.cache_control, &aux.indirect_commands,
            &aux.moe_sum, &aux.feed_forward, &aux.projection,
            &aux.context, &aux.qkv, &aux.norm, &aux.hidden_b, &aux.hidden_a,
            &aux.speculative_norm, &aux.speculative_hidden_b,
            &aux.quantized_expert_input, &aux.token_parameter, &aux.dummy};
        for (Buffer* buffer : buffers) destroy_buffer(runtime, *buffer);
        destroy_buffer(runtime, shared);
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
