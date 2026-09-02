#define OVLLM_GPTOSS_RUNTIME_ONLY
#include "m8_moe.cpp"
#include "dsv4_finite_queue_ring.hpp"

#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <psapi.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include "expert_acquisition_trace.hpp"
#include "xtllm_chat.hpp"

// DeepSeek-V4-Flash (0731) main-model executor.  This file deliberately starts
// from the low-level Vulkan substrate rather than the shape-specialized GPT-OSS
// executor.  Every submitted command buffer is finite and is completed before
// it is reset; shaders never poll memory that the host is expected to update.

namespace dsv4 {

constexpr uint32_t kDimension = 4096;
constexpr uint32_t kMoeDimension = 2048;
constexpr uint32_t kLayers = 43;
constexpr uint32_t kHashLayers = 3;
constexpr uint32_t kHeads = 64;
constexpr uint32_t kHeadDimension = 512;
constexpr uint32_t kRopeDimension = 64;
constexpr uint32_t kExperts = 256;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kHcMultiplicity = 4;
constexpr uint32_t kVocabulary = 129280;
constexpr uint32_t kWindow = 128;
// The ratio-128 HCA path has no completed compressed item at positions 0..127.
// Keeping the first correctness path strictly below 128 therefore lets it omit
// that path without changing the model's attention set.
// Stop at position 126.  At position 127 the ratio-128 compressor emits its
// first item, which belongs to the long-context path rather than this bounded
// bring-up executor.
constexpr uint32_t kShortContext = 127;
// Q4 shared weights free enough VRAM for several historical routed experts per
// layer. The row-Q8 regression remains on six through a format-aware default.
constexpr uint32_t kPersistentExpertSlotsPerLayer = 13;
constexpr uint32_t kHostExpertSlotsPerLayer = 32;
constexpr uint32_t kLegacyExpertFillWorkers = 3;
constexpr uint32_t kMaximumExpertFillWorkers = 6;
constexpr uint32_t kHostCacheRecordsPerBlock = 32;
constexpr uint32_t kHybridHostL1Slots = 1760;
constexpr uint32_t kSharedBuffers = 2;
constexpr uint32_t kWeightGroup = 32;
constexpr uint64_t kFileHeaderBytes = 4096;
constexpr uint64_t kExpertWeightBytes = 4ull * 1024 * 1024;
constexpr uint64_t kExpertScaleBytes = 256ull * 1024;
constexpr uint64_t kExpertRecordBytes = 13369344ull;
constexpr uint64_t kExpertW1Offset = 0;
constexpr uint64_t kExpertW1ScaleOffset = kExpertW1Offset + kExpertWeightBytes;
constexpr uint64_t kExpertW3Offset = kExpertW1ScaleOffset + kExpertScaleBytes;
constexpr uint64_t kExpertW3ScaleOffset = kExpertW3Offset + kExpertWeightBytes;
constexpr uint64_t kExpertW2Offset = kExpertW3ScaleOffset + kExpertScaleBytes;
constexpr uint64_t kExpertW2ScaleOffset = kExpertW2Offset + kExpertWeightBytes;
static_assert(kExpertW2ScaleOffset + kExpertScaleBytes == kExpertRecordBytes,
              "DeepSeek FP4 expert record layout drifted");

// A budgeted run reserves two GiB for every non-expert host allocation in this
// process (upload/staging buffers, tokenizer and runtime scratch).  The expert
// tier is rounded down to complete 408-MiB blocks, so its committed backing plus
// this reserve can never exceed DSV4_RAM_GIB.  Direct I/O is mandatory in that
// mode; mapped expert payload pages are never touched and cannot become a hidden
// Windows filesystem-cache tier.
constexpr uint64_t kHostBudgetReserveBytes = 2ull << 30;
static uint64_t dsv4_host_buffer_allocated_bytes = 0;
static uint64_t dsv4_plain_host_allocated_bytes = 0;

struct HostMemorySnapshot {
    uint64_t private_bytes = 0;
    uint64_t working_set_bytes = 0;
};

static HostMemorySnapshot host_memory_snapshot() {
    using Query = BOOL (WINAPI*)(HANDLE, PPROCESS_MEMORY_COUNTERS, DWORD);
    const HMODULE kernel = GetModuleHandleA("kernel32.dll");
    const auto query = kernel ? reinterpret_cast<Query>(
        GetProcAddress(kernel, "K32GetProcessMemoryInfo")) : nullptr;
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!query || !query(GetCurrentProcess(),
                         reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&counters),
                         sizeof(counters))) return {};
    return {static_cast<uint64_t>(counters.PrivateUsage),
            static_cast<uint64_t>(counters.WorkingSetSize)};
}

enum class TensorFormat : uint32_t {
    f32 = 1,
    bf16 = 2,
    f16 = 3,
    e4m3 = 4,
    e8m0 = 5,
    i8 = 6,
    u8 = 7,
    i64 = 8,
    u32 = 9,
    q8_row = 100,
    fp8_block128 = 101,
    q4g64t = 102,
    nvfp4_bf16 = 103,
};

enum class GroupKind : int32_t { global = 0, layer = 1, mtp = 2 };

#pragma pack(push, 1)
struct SharedHeader {
    char magic[8];                    // "OVD4SHR\0"
    uint32_t version;
    uint32_t header_bytes;
    uint32_t shared_format;
    uint32_t tensor_entry_bytes;
    uint32_t dimension;
    uint32_t moe_dimension;
    uint32_t layers;
    uint32_t heads;
    uint32_t kv_heads;
    uint32_t head_dimension;
    uint32_t q_lora_rank;
    uint32_t o_lora_rank;
    uint32_t o_groups;
    uint32_t qk_rope_dimension;
    uint32_t vocabulary;
    uint32_t experts;
    uint32_t top_k;
    uint32_t shared_experts;
    uint32_t hc_multiplicity;
    uint32_t hc_iterations;
    uint32_t hash_layers;
    uint32_t window;
    uint32_t index_heads;
    uint32_t index_head_dimension;
    uint32_t index_top_k;
    uint32_t maximum_position;
    uint32_t mtp_layers;
    uint32_t compress_ratio_count;
    uint32_t bos_token;
    uint32_t eos_token;
    uint32_t pad_token;
    uint32_t user_token;
    uint32_t assistant_token;
    uint32_t think_token;
    uint32_t end_think_token;
    uint32_t dsml_token;
    float rms_epsilon;
    float hc_epsilon;
    float route_scale;
    float swiglu_limit;
    float rope_theta;
    float compress_rope_theta;
    float yarn_factor;
    float beta_fast;
    float beta_slow;
    float original_max_position;
    float reserved_float0;
    float reserved_float1;
    uint64_t group_table_offset;
    uint64_t group_count;
    uint64_t tensor_table_offset;
    uint64_t tensor_count;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint64_t expert_record_bytes;
    uint64_t expert_core_records;
    uint64_t reserved_u64_0;
    uint64_t reserved_u64_1;
    uint8_t padding[40];
};

struct GroupEntry {
    int32_t kind;
    int32_t index;
    uint32_t first_tensor;
    uint32_t tensor_count;
    uint64_t data_begin;
    uint64_t data_end;
    uint64_t record_begin;
    uint64_t record_bytes;
    uint8_t padding[16];
};

struct ExpertHeader {
    char magic[8];                    // "OVD4EXP\0"
    uint32_t version;
    uint32_t header_bytes;
    uint32_t dimension;
    uint32_t moe_dimension;
    uint32_t layers;
    uint32_t experts;
    uint32_t mtp_layers;
    uint32_t record_bytes;
    uint64_t core_offset;
    uint64_t mtp_offset;
    uint64_t core_records;
    uint64_t total_records;
    uint64_t file_bytes;
    uint64_t w1_weight_offset;
    uint64_t w1_scale_offset;
    uint64_t w3_weight_offset;
    uint64_t w3_scale_offset;
    uint64_t w2_weight_offset;
    uint64_t w2_scale_offset;
    uint64_t reserved;
};

struct TokenizerHeader {
    char magic[8];                    // "OVBPE2\0\0"
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocabulary;
    uint32_t base_vocabulary;
    uint32_t merge_count;
    uint32_t bos;
    uint32_t eos;
    uint32_t configured_pad;
    uint32_t pad_piece;
    uint32_t user;
    uint32_t assistant;
    uint32_t think;
    uint32_t end_think;
    uint32_t dsml;
    uint32_t latest_reminder;
    uint32_t unknown;
    uint32_t token_entry_bytes;
    uint32_t merge_entry_bytes;
    uint32_t added_count;
    uint32_t reserved_u32;
    uint64_t token_table_offset;
    uint64_t merge_table_offset;
    uint64_t pieces_offset;
    uint64_t pieces_bytes;
    uint64_t pretokenizer_json_offset;
    uint64_t pretokenizer_json_bytes;
    uint64_t file_bytes;
    uint64_t reserved_u64;
    uint8_t padding[104];
};

struct TokenEntry {
    uint64_t piece_offset;
    uint32_t byte_length;
    uint32_t flags;
};

struct MergeEntry {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    uint32_t rank;
};

struct TensorEntry {
    char name[96];
    uint32_t dtype;
    uint32_t rank;
    uint64_t shape[8];
    uint64_t data_offset;
    uint64_t data_bytes;
    uint64_t auxiliary_offset;
    uint64_t auxiliary_bytes;
    uint64_t row_stride;
    uint64_t auxiliary_stride;
    uint64_t flags;
    uint64_t reserved;
    uint8_t padding[24];
};
#pragma pack(pop)
static_assert(sizeof(SharedHeader) == 320, "Unexpected DeepSeek shared header");
static_assert(sizeof(GroupEntry) == 64, "Unexpected DeepSeek group entry");
static_assert(sizeof(TensorEntry) == 256, "Unexpected DeepSeek tensor entry");
static_assert(sizeof(ExpertHeader) == 136, "Unexpected DeepSeek expert header");
static_assert(sizeof(TokenizerHeader) == 256, "Unexpected DeepSeek tokenizer header");
static_assert(sizeof(TokenEntry) == 16, "Unexpected DeepSeek token entry");
static_assert(sizeof(MergeEntry) == 16, "Unexpected DeepSeek merge entry");

struct TensorView {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t auxiliary_offset = 0;
    uint64_t auxiliary_bytes = 0;
    TensorFormat format = TensorFormat::f32;
    uint64_t flags = 0;
    uint32_t rank = 0;
    std::array<uint64_t, 8> shape{};
    uint64_t row_stride = 0;
    uint64_t auxiliary_stride = 0;
    uint32_t layer = UINT32_MAX;
};

class ReadOnlyMapping {
public:
    explicit ReadOnlyMapping(const std::string& path, bool writable_view = false)
        : path_(path), writable_view_(writable_view) {
        const DWORD access = writable_view ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
        file_ = CreateFileA(path.c_str(), access, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Could not open DeepSeek runtime file: " + path);
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file_, &length) || length.QuadPart <= 0)
            throw std::runtime_error("Could not size DeepSeek runtime file: " + path);
        size_ = static_cast<uint64_t>(length.QuadPart);
        mapping_ = CreateFileMappingA(file_, nullptr,
            writable_view ? PAGE_READWRITE : PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_)
            throw std::runtime_error("Could not create DeepSeek file mapping: " + path);
        const DWORD map_access = writable_view ?
            (FILE_MAP_READ | FILE_MAP_WRITE) : FILE_MAP_READ;
        data_ = static_cast<const uint8_t*>(MapViewOfFile(
            mapping_, map_access, 0, 0, 0));
        if (!data_) throw std::runtime_error("Could not map DeepSeek runtime file: " + path);
    }

    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
    ~ReadOnlyMapping() {
        if (data_) UnmapViewOfFile(data_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }
    const uint8_t* data() const { return data_; }
    uint64_t size() const { return size_; }
    const std::string& path() const { return path_; }
    bool writable_view() const { return writable_view_; }

private:
    std::string path_;
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    const uint8_t* data_ = nullptr;
    uint64_t size_ = 0;
    bool writable_view_ = false;
};

static std::string bounded_name(const char* name, size_t capacity) {
    size_t length = 0;
    while (length != capacity && name[length] != '\0') ++length;
    if (length == capacity) throw std::runtime_error("Unterminated DeepSeek tensor name");
    return std::string(name, length);
}

class SharedIndex {
public:
    explicit SharedIndex(const ReadOnlyMapping& file) : file_(file) {
        if (file.size() < kFileHeaderBytes) throw std::runtime_error("Truncated model.ovs");
        std::memcpy(&header_, file.data(), sizeof(header_));
        if (std::memcmp(header_.magic, "OVD4SHR\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kFileHeaderBytes ||
            header_.tensor_entry_bytes != sizeof(TensorEntry) ||
            (header_.shared_format != 1 && header_.shared_format != 2 &&
             header_.shared_format != 3)) {
            throw std::runtime_error("Unsupported DeepSeek shared container");
        }
        if (header_.dimension != kDimension || header_.moe_dimension != kMoeDimension ||
            header_.layers != kLayers || header_.hash_layers != kHashLayers ||
            header_.heads != kHeads || header_.kv_heads != 1 ||
            header_.head_dimension != kHeadDimension || header_.q_lora_rank != 1024 ||
            header_.o_lora_rank != 1024 || header_.o_groups != 8 ||
            header_.qk_rope_dimension != kRopeDimension || header_.experts != kExperts ||
            header_.top_k != kTopK || header_.vocabulary != kVocabulary ||
            header_.shared_experts != 1 || header_.window != kWindow ||
            header_.hc_multiplicity != kHcMultiplicity || header_.hc_iterations != 20 ||
            header_.index_heads != 64 || header_.index_head_dimension != 128 ||
            header_.index_top_k != 512 || header_.compress_ratio_count != kLayers ||
            header_.expert_record_bytes != kExpertRecordBytes ||
            header_.expert_core_records != static_cast<uint64_t>(kLayers) * kExperts) {
            throw std::runtime_error("model.ovs is not DeepSeek-V4-Flash-0731");
        }
        if (header_.file_bytes != file.size() || header_.group_table_offset != sizeof(header_) ||
            header_.tensor_table_offset < kFileHeaderBytes ||
            header_.data_offset < header_.tensor_table_offset) {
            throw std::runtime_error("Invalid DeepSeek shared-container bounds");
        }
        const uint64_t group_bytes = header_.group_count * sizeof(GroupEntry);
        const uint64_t tensor_bytes =
            static_cast<uint64_t>(header_.tensor_count) * sizeof(TensorEntry);
        if (header_.group_count == 0 || header_.group_count > 64 ||
            header_.tensor_count == 0 || header_.tensor_count > 10000 ||
            group_bytes > kFileHeaderBytes - header_.group_table_offset ||
            header_.tensor_table_offset > file.size() ||
            tensor_bytes > file.size() - header_.tensor_table_offset ||
            header_.tensor_table_offset + tensor_bytes > header_.data_offset) {
            throw std::runtime_error("DeepSeek metadata tables exceed model.ovs");
        }
        layer_begin_.fill(UINT64_MAX);
        layer_end_.fill(0);
        const uint64_t ratio_offset = header_.group_table_offset + group_bytes;
        const uint64_t ratio_bytes = static_cast<uint64_t>(header_.compress_ratio_count) * 4u;
        if (ratio_offset + ratio_bytes > kFileHeaderBytes)
            throw std::runtime_error("DeepSeek compression ratios exceed header");
        std::memcpy(compress_ratios_.data(), file.data() + ratio_offset,
                    static_cast<size_t>(ratio_bytes));
        if (compress_ratios_[0] != 0 || compress_ratios_[1] != 0 ||
            compress_ratios_[2] != 4)
            throw std::runtime_error("Unexpected DeepSeek compression schedule");
        const auto* groups = reinterpret_cast<const GroupEntry*>(
            file.data() + header_.group_table_offset);
        const auto* entries = reinterpret_cast<const TensorEntry*>(
            file.data() + header_.tensor_table_offset);
        tensors_.reserve(static_cast<size_t>(header_.tensor_count));
        std::vector<bool> seen(static_cast<size_t>(header_.tensor_count));
        for (uint64_t group_index = 0; group_index < header_.group_count; ++group_index) {
            const GroupEntry& group = groups[group_index];
            if (static_cast<uint64_t>(group.first_tensor) + group.tensor_count >
                    header_.tensor_count || group.data_begin < header_.data_offset ||
                group.data_end <= group.data_begin || group.data_end > file.size() ||
                (group.data_begin & 4095u) != 0 || (group.data_end & 4095u) != 0) {
                throw std::runtime_error("Invalid DeepSeek tensor group");
            }
            uint32_t layer = UINT32_MAX;
            if (group.kind == static_cast<int32_t>(GroupKind::layer)) {
                if (group.index < 0 || static_cast<uint32_t>(group.index) >= kLayers)
                    throw std::runtime_error("Invalid DeepSeek layer group");
                layer = static_cast<uint32_t>(group.index);
                if (layer_begin_[layer] != UINT64_MAX)
                    throw std::runtime_error("Duplicate DeepSeek layer group");
                layer_begin_[layer] = group.data_begin;
                layer_end_[layer] = group.data_end;
            } else if (group.kind == static_cast<int32_t>(GroupKind::global)) {
                if (group.index != -1 || global_begin_ != UINT64_MAX)
                    throw std::runtime_error("Invalid or duplicate DeepSeek global group");
                global_begin_ = group.data_begin;
                global_end_ = group.data_end;
            } else if (group.kind != static_cast<int32_t>(GroupKind::mtp)) {
                throw std::runtime_error("Unknown DeepSeek tensor group kind");
            }
            for (uint32_t local = 0; local < group.tensor_count; ++local) {
                const uint32_t tensor = group.first_tensor + local;
                if (seen[tensor]) throw std::runtime_error("Tensor appears in two groups");
                seen[tensor] = true;
                add(entries[tensor], layer, group.data_begin, group.data_end);
            }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::runtime_error("Ungrouped tensor in model.ovs");
        if (global_begin_ == UINT64_MAX || global_end_ <= global_begin_)
            throw std::runtime_error("model.ovs has no global tensor group");
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            if (layer_begin_[layer] == UINT64_MAX || layer_end_[layer] <= layer_begin_[layer])
                throw std::runtime_error("model.ovs has an empty main layer");
        }
    }

    const SharedHeader& header() const { return header_; }
    const TensorView& require(const std::string& name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end())
            throw std::runtime_error("Missing DeepSeek tensor: " + name);
        return found->second;
    }
    const TensorView* find(const std::string& name) const {
        const auto found = tensors_.find(name);
        return found == tensors_.end() ? nullptr : &found->second;
    }
    std::pair<uint64_t, uint64_t> layer_range(uint32_t layer) const {
        if (layer >= kLayers) throw std::runtime_error("Invalid DeepSeek layer");
        return {layer_begin_[layer], layer_end_[layer]};
    }
    uint32_t compression_ratio(uint32_t layer) const { return compress_ratios_.at(layer); }
    std::pair<uint64_t, uint64_t> global_range() const {
        return {global_begin_, global_end_};
    }
    uint64_t maximum_layer_bytes() const {
        uint64_t maximum = 0;
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            const auto range = layer_range(layer);
            maximum = std::max(maximum, range.second - range.first);
        }
        return maximum;
    }

private:
    void add(const TensorEntry& entry, uint32_t layer,
             uint64_t group_begin, uint64_t group_end) {
        if (entry.rank > 8 || entry.data_offset < group_begin ||
            entry.data_offset > group_end || entry.data_bytes > group_end - entry.data_offset ||
            (entry.auxiliary_bytes != 0 &&
             (entry.auxiliary_offset < group_begin || entry.auxiliary_offset > group_end ||
              entry.auxiliary_bytes > group_end - entry.auxiliary_offset))) {
            throw std::runtime_error("Invalid DeepSeek tensor entry");
        }
        TensorView view;
        view.offset = entry.data_offset;
        view.bytes = entry.data_bytes;
        view.auxiliary_offset = entry.auxiliary_offset;
        view.auxiliary_bytes = entry.auxiliary_bytes;
        view.format = static_cast<TensorFormat>(entry.dtype);
        view.flags = entry.flags;
        view.rank = entry.rank;
        std::copy(std::begin(entry.shape), std::end(entry.shape), view.shape.begin());
        view.row_stride = entry.row_stride;
        view.auxiliary_stride = entry.auxiliary_stride;
        view.layer = layer;
        const std::string name = bounded_name(entry.name, sizeof(entry.name));
        if (!tensors_.emplace(name, view).second)
            throw std::runtime_error("Duplicate DeepSeek tensor: " + name);
    }

    const ReadOnlyMapping& file_;
    SharedHeader header_{};
    std::unordered_map<std::string, TensorView> tensors_;
    std::array<uint64_t, kLayers> layer_begin_{};
    std::array<uint64_t, kLayers> layer_end_{};
    std::array<uint32_t, kLayers> compress_ratios_{};
    uint64_t global_begin_ = UINT64_MAX;
    uint64_t global_end_ = 0;
};

class ExpertIndex {
public:
    explicit ExpertIndex(const ReadOnlyMapping& file) : file_(file) {
        if (file.size() < kFileHeaderBytes) throw std::runtime_error("Truncated experts.ovx");
        std::memcpy(&header_, file.data(), sizeof(header_));
        const bool native_fp4 = std::memcmp(header_.magic, "OVD4EXP\0", 8) == 0 &&
            header_.reserved == 0;
        const bool q4g64t = std::memcmp(header_.magic, "OVD4Q4T\0", 8) == 0 &&
            header_.reserved == 102;
        if ((!native_fp4 && !q4g64t) ||
            header_.version != 1 || header_.header_bytes != kFileHeaderBytes ||
            header_.layers != kLayers || header_.experts != kExperts ||
            header_.dimension != kDimension || header_.moe_dimension != kMoeDimension ||
            header_.record_bytes != kExpertRecordBytes ||
            header_.core_records != static_cast<uint64_t>(kLayers) * kExperts ||
            header_.core_offset != kFileHeaderBytes ||
            header_.w1_weight_offset != kExpertW1Offset ||
            header_.w1_scale_offset != kExpertW1ScaleOffset ||
            header_.w3_weight_offset != kExpertW3Offset ||
            header_.w3_scale_offset != kExpertW3ScaleOffset ||
            header_.w2_weight_offset != kExpertW2Offset ||
            header_.w2_scale_offset != kExpertW2ScaleOffset ||
            header_.file_bytes != file.size()) {
            throw std::runtime_error("Unsupported DeepSeek expert container");
        }
        q4g64t_ = q4g64t;
        if (header_.total_records < header_.core_records ||
            header_.total_records > (file.size() - kFileHeaderBytes) / kExpertRecordBytes ||
            kFileHeaderBytes + header_.total_records * kExpertRecordBytes != file.size() ||
            header_.mtp_offset != kFileHeaderBytes + header_.core_records * kExpertRecordBytes) {
            throw std::runtime_error("experts.ovx size does not match its header");
        }
    }

    const uint8_t* core_record(uint32_t layer, uint32_t expert) const {
        if (layer >= kLayers || expert >= kExperts)
            throw std::runtime_error("Invalid DeepSeek expert key");
        return file_.data() + header_.core_offset +
            (static_cast<uint64_t>(layer) * kExperts + expert) * kExpertRecordBytes;
    }
    uint64_t core_record_offset(uint32_t layer, uint32_t expert) const {
        if (layer >= kLayers || expert >= kExperts)
            throw std::runtime_error("Invalid DeepSeek expert key");
        return header_.core_offset +
            (static_cast<uint64_t>(layer) * kExperts + expert) * kExpertRecordBytes;
    }
    const std::string& path() const { return file_.path(); }
    uint64_t size() const { return file_.size(); }
    bool q4g64t() const { return q4g64t_; }

private:
    const ReadOnlyMapping& file_;
    ExpertHeader header_{};
    bool q4g64t_ = false;
};

#pragma pack(push, 1)
struct ExpertBundleHeader {
    char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layers;
    uint32_t experts;
    uint32_t top_k;
    uint32_t record_format_tag;
    uint32_t bundle_records;
    uint32_t flags;
    uint64_t record_bytes;
    uint64_t source_header_bytes;
    uint64_t source_data_bytes;
    uint64_t source_file_bytes;
    uint64_t data_file_bytes;
    uint64_t bundle_count;
    uint64_t logical_count;
    uint64_t copy_count;
    uint64_t bundle_table_offset;
    uint64_t logical_table_offset;
    uint64_t copy_table_offset;
    uint64_t file_bytes;
    uint8_t source_header_sha256[32];
};
struct ExpertBundleEntry {
    uint64_t data_offset;
    uint64_t data_bytes;
    uint32_t bundle_id;
    uint16_t record_count;
    uint16_t flags;
    uint64_t trace_requests;
    uint64_t same_set_weight;
    uint64_t adjacent_weight;
    uint64_t original_reads;
    uint64_t bundled_reads;
};
struct ExpertBundleLogicalEntry {
    uint16_t layer;
    uint16_t expert;
    uint32_t first_copy;
    uint32_t copy_count;
    uint64_t source_offset;
    uint64_t request_count;
    uint32_t flags;
};
struct ExpertBundleCopyEntry {
    uint16_t layer;
    uint16_t expert;
    uint32_t bundle_id;
    uint16_t ordinal;
    uint16_t flags;
    uint64_t data_offset;
    uint64_t read_cost_bytes;
    uint32_t live_state;
    uint32_t location;
    uint64_t timeline;
    uint64_t submitted_ns;
    uint64_t eta_ns;
    double predicted_probability;
    uint64_t request_count;
    uint32_t reserved;
};
#pragma pack(pop)
static_assert(sizeof(ExpertBundleHeader) == 168,
              "Unexpected DeepSeek bundle header");
static_assert(sizeof(ExpertBundleEntry) == 64,
              "Unexpected DeepSeek bundle entry");
static_assert(sizeof(ExpertBundleLogicalEntry) == 32,
              "Unexpected DeepSeek bundle logical entry");
static_assert(sizeof(ExpertBundleCopyEntry) == 80,
              "Unexpected DeepSeek bundle copy entry");

// Immutable physical-copy metadata.  Residency and transfer state continue to
// live in ExpertBlueprint, so the mapped index never becomes an unsafe
// CPU/GPU rendezvous surface.
class ExpertBundleIndex {
public:
    struct Copy {
        uint32_t copy_index = 0;
        uint32_t bundle_id = 0;
        uint32_t ordinal = 0;
        uint32_t flags = 0;
        uint64_t data_offset = 0;
        uint64_t read_cost_bytes = 0;
        float predicted_probability = 0.0f;
    };

    ExpertBundleIndex(const ReadOnlyMapping& file, const ExpertIndex& experts)
        : file_(file) {
        if (!experts.q4g64t() || file.size() < kFileHeaderBytes)
            throw std::runtime_error(
                "DeepSeek affinity bundles require canonical Q4G64T experts");
        std::memcpy(&header_, file.data(), sizeof(header_));
        constexpr uint64_t logical_count =
            static_cast<uint64_t>(kLayers) * kExperts;
        if (std::memcmp(header_.magic, "OVD4BND\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kFileHeaderBytes ||
            header_.layers != kLayers || header_.experts != kExperts ||
            header_.top_k != kTopK || header_.record_format_tag != 102u ||
            header_.bundle_records != kTopK || header_.flags != 0u ||
            header_.record_bytes != kExpertRecordBytes ||
            header_.source_header_bytes != kFileHeaderBytes ||
            header_.source_data_bytes != experts.size() - kFileHeaderBytes ||
            header_.source_file_bytes != experts.size() ||
            header_.logical_count != logical_count ||
            header_.bundle_count == 0 || header_.copy_count < logical_count ||
            header_.file_bytes != file.size()) {
            throw std::runtime_error("Unsupported DeepSeek expert-bundle index");
        }
        const auto table_valid = [&](uint64_t offset, uint64_t count,
                                     uint64_t entry_bytes) {
            return offset >= kFileHeaderBytes && (offset & 4095u) == 0u &&
                offset <= file.size() &&
                count <= (file.size() - offset) / entry_bytes;
        };
        if (!table_valid(header_.bundle_table_offset, header_.bundle_count,
                         sizeof(ExpertBundleEntry)) ||
            !table_valid(header_.logical_table_offset, header_.logical_count,
                         sizeof(ExpertBundleLogicalEntry)) ||
            !table_valid(header_.copy_table_offset, header_.copy_count,
                         sizeof(ExpertBundleCopyEntry))) {
            throw std::runtime_error("DeepSeek expert-bundle tables exceed index");
        }
        const auto* bundles = reinterpret_cast<const ExpertBundleEntry*>(
            file.data() + header_.bundle_table_offset);
        bundles_.assign(bundles, bundles + header_.bundle_count);
        uint64_t physical_records = 0;
        uint64_t expected_offset = 0;
        for (uint64_t index = 0; index < header_.bundle_count; ++index) {
            const ExpertBundleEntry& bundle = bundles_[index];
            if (bundle.bundle_id != index || bundle.record_count == 0 ||
                bundle.record_count > kTopK || bundle.data_offset != expected_offset ||
                bundle.data_bytes !=
                    static_cast<uint64_t>(bundle.record_count) * kExpertRecordBytes ||
                (bundle.data_offset & 65535u) != 0u) {
                throw std::runtime_error("Invalid DeepSeek physical expert bundle");
            }
            physical_records += bundle.record_count;
            expected_offset += bundle.data_bytes;
        }
        if (expected_offset != header_.data_file_bytes ||
            physical_records != header_.copy_count)
            throw std::runtime_error("DeepSeek bundle-store size drift");

        const auto* logical = reinterpret_cast<const ExpertBundleLogicalEntry*>(
            file.data() + header_.logical_table_offset);
        const auto* copies = reinterpret_cast<const ExpertBundleCopyEntry*>(
            file.data() + header_.copy_table_offset);
        copies_.resize(static_cast<size_t>(logical_count));
        std::vector<bool> physical_seen(static_cast<size_t>(physical_records));
        for (uint32_t key = 0; key < logical_count; ++key) {
            const uint32_t layer = key / kExperts, expert = key % kExperts;
            const ExpertBundleLogicalEntry& item = logical[key];
            if (item.layer != layer || item.expert != expert ||
                item.copy_count == 0 || item.first_copy > header_.copy_count ||
                item.copy_count > header_.copy_count - item.first_copy ||
                item.source_offset != kFileHeaderBytes +
                    static_cast<uint64_t>(key) * kExpertRecordBytes) {
                throw std::runtime_error("Invalid DeepSeek logical bundle entry");
            }
            std::vector<Copy>& output = copies_[key];
            output.reserve(item.copy_count);
            for (uint32_t local = 0; local < item.copy_count; ++local) {
                const uint32_t copy_index = item.first_copy + local;
                const ExpertBundleCopyEntry& source = copies[copy_index];
                if (source.layer != layer || source.expert != expert ||
                    source.bundle_id >= bundles_.size())
                    throw std::runtime_error("Invalid DeepSeek expert-bundle copy key");
                const ExpertBundleEntry& bundle = bundles_[source.bundle_id];
                if (source.ordinal >= bundle.record_count ||
                    source.data_offset != bundle.data_offset +
                        static_cast<uint64_t>(source.ordinal) * kExpertRecordBytes ||
                    source.read_cost_bytes != kExpertRecordBytes ||
                    source.data_offset > header_.data_file_bytes - kExpertRecordBytes)
                    throw std::runtime_error("Invalid DeepSeek expert-bundle copy range");
                const uint64_t physical = source.data_offset / kExpertRecordBytes;
                if (physical >= physical_seen.size() || physical_seen[physical])
                    throw std::runtime_error("Duplicate DeepSeek physical copy index");
                physical_seen[physical] = true;
                output.push_back({copy_index, source.bundle_id, source.ordinal,
                                  source.flags, source.data_offset,
                                  source.read_cost_bytes,
                                  static_cast<float>(source.predicted_probability)});
            }
        }
        if (std::find(physical_seen.begin(), physical_seen.end(), false) !=
            physical_seen.end())
            throw std::runtime_error("Unindexed DeepSeek physical expert copy");
    }

    const std::vector<Copy>& copies(uint32_t layer, uint32_t expert) const {
        if (layer >= kLayers || expert >= kExperts)
            throw std::runtime_error("Invalid DeepSeek bundle-copy key");
        return copies_[static_cast<size_t>(layer) * kExperts + expert];
    }
    uint64_t data_file_bytes() const { return header_.data_file_bytes; }

private:
    const ReadOnlyMapping& file_;
    ExpertBundleHeader header_{};
    std::vector<ExpertBundleEntry> bundles_;
    std::vector<std::vector<Copy>> copies_;
};

static DescriptorRange tensor_data_range(const Buffer& arena, uint64_t file_base,
                                         const TensorView& tensor) {
    if (tensor.offset < file_base || tensor.bytes == 0 ||
        tensor.offset - file_base > arena.size ||
        tensor.bytes > arena.size - (tensor.offset - file_base))
        throw std::runtime_error("DeepSeek tensor data is outside its GPU arena");
    return arena_range(arena, tensor.offset - file_base, tensor.bytes);
}

static DescriptorRange tensor_auxiliary_range(const Buffer& arena, uint64_t file_base,
                                              const TensorView& tensor) {
    if (tensor.auxiliary_offset < file_base || tensor.auxiliary_bytes == 0 ||
        tensor.auxiliary_offset - file_base > arena.size ||
        tensor.auxiliary_bytes > arena.size - (tensor.auxiliary_offset - file_base))
        throw std::runtime_error("DeepSeek tensor scales are outside their GPU arena");
    return arena_range(arena, tensor.auxiliary_offset - file_base,
                       tensor.auxiliary_bytes);
}

static void require_linear_matrix(const TensorView& tensor, uint32_t rows,
                                  uint32_t inner, const char* purpose) {
    const bool shape = tensor.rank == 2 && tensor.shape[0] == rows &&
                       tensor.shape[1] == inner;
    const bool q8 = tensor.format == TensorFormat::q8_row &&
        tensor.bytes == static_cast<uint64_t>(rows) * inner &&
        tensor.auxiliary_bytes == static_cast<uint64_t>(rows) * sizeof(float);
    const bool q4 = tensor.format == TensorFormat::q4g64t && inner % 64u == 0u &&
        tensor.bytes == static_cast<uint64_t>(rows) * inner / 2u &&
        tensor.auxiliary_bytes == static_cast<uint64_t>(rows) * (inner / 64u) * 2u &&
        tensor.row_stride == inner / 2u &&
        tensor.auxiliary_stride == (inner / 64u) * 2u && tensor.flags == 64u;
    if (!shape || (!q8 && !q4))
        throw std::runtime_error(std::string("Invalid Q8/Q4 linear matrix for ") + purpose);
}

class Tokenizer {
public:
    explicit Tokenizer(const ReadOnlyMapping& file) : file_(file) {
        if (file.size() < sizeof(TokenizerHeader))
            throw std::runtime_error("Truncated tokenizer.ovb");
        std::memcpy(&header_, file.data(), sizeof(header_));
        const bool deepseek = header_.vocabulary == kVocabulary &&
            header_.added_count == 1283 && header_.configured_pad == 1 &&
            header_.user == 128803 && header_.assistant == 128804 &&
            header_.think == 128821 && header_.end_think == 128822 &&
            header_.dsml == 128825 && header_.latest_reminder == 128828;
        const bool step37 = header_.vocabulary == 128896 &&
            header_.added_count == 818 && header_.configured_pad == 2 &&
            header_.user == 128006 && header_.assistant == 128007 &&
            header_.think == 128798 && header_.end_think == 128799;
        if (std::memcmp(header_.magic, "OVBPE2\0\0", 8) != 0 ||
            header_.version != 2 || header_.header_bytes != sizeof(TokenizerHeader) ||
            (!deepseek && !step37) || header_.base_vocabulary != 128000 ||
            header_.merge_count != 127741 || header_.token_entry_bytes != sizeof(TokenEntry) ||
            header_.merge_entry_bytes != sizeof(MergeEntry) ||
            header_.file_bytes != file.size() || header_.bos != 0 || header_.eos != 1 ||
            header_.pad_piece != 2 || header_.unknown != UINT32_MAX) {
            throw std::runtime_error("Unsupported DeepSeek tokenizer container");
        }
        validate_table(header_.token_table_offset,
            static_cast<uint64_t>(header_.vocabulary) * sizeof(TokenEntry),
            "token table");
        validate_table(header_.merge_table_offset,
            static_cast<uint64_t>(header_.merge_count) * sizeof(MergeEntry),
            "merge table");
        validate_table(header_.pieces_offset, header_.pieces_bytes, "token pieces");
        validate_table(header_.pretokenizer_json_offset, header_.pretokenizer_json_bytes,
                       "pre-tokenizer JSON");
        entries_ = reinterpret_cast<const TokenEntry*>(
            file.data() + header_.token_table_offset);
        const auto* disk_merges = reinterpret_cast<const MergeEntry*>(
            file.data() + header_.merge_table_offset);
        pieces_.resize(header_.vocabulary);
        special_.resize(header_.vocabulary);
        byte_token_.fill(UINT32_MAX);
        for (uint32_t token = 0; token < header_.vocabulary; ++token) {
            const TokenEntry& entry = entries_[token];
            if (entry.piece_offset < header_.pieces_offset ||
                entry.piece_offset > header_.pieces_offset + header_.pieces_bytes ||
                entry.byte_length > header_.pieces_offset + header_.pieces_bytes -
                                    entry.piece_offset) {
                throw std::runtime_error("Tokenizer piece exceeds tokenizer.ovb");
            }
            pieces_[token] = std::string(reinterpret_cast<const char*>(
                file.data() + entry.piece_offset), entry.byte_length);
            const bool configured_control = token == header_.bos || token == header_.eos ||
                token == header_.pad_piece || token == header_.user ||
                token == header_.assistant || token == header_.think ||
                token == header_.end_think || token == header_.dsml ||
                token == header_.latest_reminder;
            // The official tokenizer marks several chat control pieces as
            // added/indivisible but special:false.  Header identity, not the HF
            // display flag, determines whether generation should hide them.
            if (configured_control && (entry.flags & 3u) == 0)
                throw std::runtime_error("Configured tokenizer control is not added");
            special_[token] = configured_control || (entry.flags & 2u) != 0;
            if (token < header_.base_vocabulary && pieces_[token].size() == 1) {
                const uint8_t byte = static_cast<uint8_t>(pieces_[token][0]);
                if (byte_token_[byte] == UINT32_MAX) byte_token_[byte] = token;
            }
        }
        merges_.reserve(header_.merge_count * 2u);
        for (uint32_t index = 0; index < header_.merge_count; ++index) {
            const MergeEntry& merge = disk_merges[index];
            if (merge.left >= header_.base_vocabulary ||
                merge.right >= header_.base_vocabulary ||
                merge.result >= header_.base_vocabulary || merge.rank >= header_.merge_count) {
                throw std::runtime_error("Invalid DeepSeek BPE merge");
            }
            const uint64_t key = pair_key(merge.left, merge.right);
            if (!merges_.emplace(key, Merge{merge.result, merge.rank}).second)
                throw std::runtime_error("Duplicate DeepSeek BPE merge pair");
        }
        if (std::find(byte_token_.begin(), byte_token_.end(), UINT32_MAX) !=
                byte_token_.end())
            throw std::runtime_error("DeepSeek tokenizer lacks a raw byte token");
        require_special(header_.bos);
        require_special(header_.eos);
        require_special(header_.user);
        require_special(header_.assistant);
        require_special(header_.think);
        require_special(header_.end_think);
    }

    std::vector<uint32_t> encode_text(const std::string& text) const {
        std::vector<uint32_t> output;
        for (const std::string& segment : split_ascii(text)) {
            std::vector<uint32_t> symbols;
            symbols.reserve(segment.size());
            for (const uint8_t byte : std::vector<uint8_t>(segment.begin(), segment.end()))
                symbols.push_back(byte_token_[byte]);
            while (symbols.size() > 1) {
                uint32_t best_rank = UINT32_MAX;
                size_t best_index = symbols.size();
                uint32_t best_result = 0;
                for (size_t index = 0; index + 1 < symbols.size(); ++index) {
                    const auto found = merges_.find(pair_key(symbols[index], symbols[index + 1]));
                    if (found != merges_.end() && found->second.rank < best_rank) {
                        best_rank = found->second.rank;
                        best_result = found->second.result;
                        best_index = index;
                    }
                }
                if (best_index == symbols.size()) break;
                symbols[best_index] = best_result;
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
            }
            output.insert(output.end(), symbols.begin(), symbols.end());
        }
        return output;
    }

    std::vector<uint32_t> chat_prompt(const std::string& user_text,
                                      bool thinking = false,
                                      const std::string& system_text = {}) const {
        std::filesystem::path transcript_path;
        if (xtllm_chat::referenced_path(user_text, transcript_path))
            return xtllm_chat::render_deepseek(
                xtllm_chat::read(transcript_path),
                [this](const std::string& value) { return encode_text(value); },
                header_.bos, header_.eos, header_.user, header_.assistant,
                header_.think, header_.end_think, thinking,
                header_.vocabulary == 128896, system_text);
        std::vector<uint32_t> result;
        result.push_back(header_.bos);
        if (header_.vocabulary == 128896) {
            const auto append_text = [&](const std::string& text) {
                const std::vector<uint32_t> encoded = encode_text(text);
                result.insert(result.end(), encoded.begin(), encoded.end());
            };
            if (!system_text.empty()) {
                result.push_back(header_.user);
                append_text("system\n" + system_text);
                result.push_back(header_.assistant);
                append_text("\n");
            }
            result.push_back(header_.user);
            append_text("user\n" + user_text);
            result.push_back(header_.assistant);
            append_text("\n");
            result.push_back(header_.user);
            append_text("assistant\n");
            result.push_back(thinking ? header_.think : header_.end_think);
            append_text("\n");
            return result;
        }
        if (!system_text.empty()) {
            std::vector<uint32_t> system = encode_text(system_text);
            result.insert(result.end(), system.begin(), system.end());
        }
        result.push_back(header_.user);
        std::vector<uint32_t> user = encode_text(user_text);
        result.insert(result.end(), user.begin(), user.end());
        result.push_back(header_.assistant);
        result.push_back(thinking ? header_.think : header_.end_think);
        return result;
    }

    std::string decode_piece(uint32_t token) const {
        if (token >= pieces_.size() || special_[token]) return {};
        return pieces_[token];
    }
    std::string decode(const std::vector<uint32_t>& tokens) const {
        std::string result;
        for (uint32_t token : tokens) result += decode_piece(token);
        return result;
    }
    uint32_t eos() const { return header_.eos; }

private:
    struct Merge { uint32_t result; uint32_t rank; };
    static uint64_t pair_key(uint32_t left, uint32_t right) {
        return static_cast<uint64_t>(left) << 32u | right;
    }
    void validate_table(uint64_t offset, uint64_t bytes, const char* label) const {
        if (offset < header_.header_bytes || offset > file_.size() ||
            bytes > file_.size() - offset)
            throw std::runtime_error(std::string("DeepSeek ") + label + " exceeds tokenizer.ovb");
    }
    void require_special(uint32_t token) const {
        if (token >= special_.size() || !special_[token])
            throw std::runtime_error("DeepSeek tokenizer special-token flags are inconsistent");
    }
    static bool ascii_letter(uint8_t value) {
        return (value >= 'A' && value <= 'Z') || (value >= 'a' && value <= 'z');
    }
    static bool ascii_digit(uint8_t value) { return value >= '0' && value <= '9'; }
    static bool ascii_space(uint8_t value) {
        return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
               value == '\v' || value == '\f';
    }
    static bool ascii_punctuation_or_symbol(uint8_t value) {
        return (value >= 0x21 && value <= 0x2f) ||
               (value >= 0x3a && value <= 0x40) ||
               (value >= 0x5b && value <= 0x60) ||
               (value >= 0x7b && value <= 0x7e);
    }

    // Common-ASCII implementation of the checkpoint's three ordered isolated
    // splits.  In particular this is not the usual GPT-2 contraction regex:
    // numbers are first isolated in chunks of at most three, and punctuation
    // immediately followed by ASCII letters is one segment.  The converter
    // retains the authoritative JSON for a future full-Unicode regex port.
    static std::vector<std::string> split_ascii(const std::string& text) {
        std::vector<std::string> segments;
        size_t position = 0;
        while (position < text.size()) {
            const size_t begin = position;
            const uint8_t first = static_cast<uint8_t>(text[position]);
            // Ordered split 1: \\p{N}{1,3}.
            if (ascii_digit(first)) {
                uint32_t count = 0;
                while (position < text.size() && count != 3 &&
                       ascii_digit(static_cast<uint8_t>(text[position]))) {
                    ++position;
                    ++count;
                }
            // Third-regex alternative 1: one ASCII punctuation/symbol followed
            // immediately by one or more ASCII letters.
            } else if (ascii_punctuation_or_symbol(first) &&
                       position + 1 < text.size() &&
                       ascii_letter(static_cast<uint8_t>(text[position + 1]))) {
                ++position;
                while (position < text.size() &&
                       ascii_letter(static_cast<uint8_t>(text[position]))) ++position;
            // Third-regex alternative 2: an optional single non-letter,
            // non-punctuation prefix (normally a space), then letters.
            } else if (ascii_letter(first) ||
                       ((!ascii_punctuation_or_symbol(first) && first != '\r' &&
                         first != '\n') && position + 1 < text.size() &&
                        ascii_letter(static_cast<uint8_t>(text[position + 1])))) {
                if (!ascii_letter(first)) ++position;
                while (position < text.size() &&
                       ascii_letter(static_cast<uint8_t>(text[position]))) ++position;
            // Third-regex alternative 3: optional one plain space, a symbol
            // run, then any immediately following CR/LF sequence.
            } else if ((first == ' ' && position + 1 < text.size() &&
                        ascii_punctuation_or_symbol(
                            static_cast<uint8_t>(text[position + 1]))) ||
                       ascii_punctuation_or_symbol(first)) {
                if (first == ' ') ++position;
                while (position < text.size() && ascii_punctuation_or_symbol(
                           static_cast<uint8_t>(text[position]))) ++position;
                while (position < text.size() &&
                       (text[position] == '\r' || text[position] == '\n')) ++position;
            // Third-regex alternatives 4--6.  Whitespace immediately before a
            // newline belongs to the newline segment; other whitespace is a run.
            } else if (ascii_space(first)) {
                size_t newline = position;
                while (newline < text.size() && ascii_space(
                           static_cast<uint8_t>(text[newline])) &&
                       text[newline] != '\r' && text[newline] != '\n') ++newline;
                if (newline < text.size() &&
                    (text[newline] == '\r' || text[newline] == '\n')) {
                    position = newline;
                    while (position < text.size() &&
                           (text[position] == '\r' || text[position] == '\n')) ++position;
                } else {
                    while (position < text.size() && ascii_space(
                               static_cast<uint8_t>(text[position])) &&
                           text[position] != '\r' && text[position] != '\n') ++position;
                }
            } else if (first >= 0x80) {
                // Ordered split 2 is implemented exactly for UTF-8 CJK only in
                // the future Unicode port.  Keeping a non-ASCII byte run intact
                // is lossless and deterministic for the initial English check.
                while (position < text.size() &&
                       static_cast<uint8_t>(text[position]) >= 0x80) ++position;
            } else {
                ++position;
            }
            if (position == begin) ++position;
            segments.push_back(text.substr(begin, position - begin));
        }
        return segments;
    }

    const ReadOnlyMapping& file_;
    TokenizerHeader header_{};
    const TokenEntry* entries_ = nullptr;
    std::vector<std::string> pieces_;
    std::vector<bool> special_;
    std::array<uint32_t, 256> byte_token_{};
    std::unordered_map<uint64_t, Merge> merges_;
};

static Buffer create_host_buffer_uninitialized(const Runtime& runtime, VkDeviceSize size) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    VkMemoryAllocateFlagsInfo address{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    address.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &address;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_memory_type(
        runtime, requirements.memoryTypeBits, buffer.coherent);
    const VkResult allocated = vkfn::AllocateMemory(
        runtime.device, &allocation, nullptr, &buffer.memory);
    if (allocated != VK_SUCCESS)
        throw std::runtime_error("DeepSeek host-visible Vulkan allocation failed (" +
            std::to_string(static_cast<uint64_t>(size)) + " bytes, VkResult " +
            std::to_string(static_cast<int32_t>(allocated)) + ")");
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes,
                                        active_vulkan_buffer_bytes);
    dsv4_host_buffer_allocated_bytes += buffer.allocation_size;
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle, buffer.memory, 0));
    VK_CHECK(vkfn::MapMemory(runtime.device, buffer.memory, 0, VK_WHOLE_SIZE, 0,
                            &buffer.mapped));
    return buffer;
}

static void flush_buffer_range(const Runtime& runtime, const Buffer& buffer,
                               VkDeviceSize offset, VkDeviceSize size) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.offset = offset;
    range.size = size;
    VK_CHECK(vkfn::FlushMappedMemoryRanges(runtime.device, 1, &range));
}

struct Dsv4ImportedRange {
    Buffer buffer{};
    VkDeviceSize data_offset = 0;
};

static Dsv4ImportedRange import_dsv4_host_range(const Runtime& runtime,
                                                 const uint8_t* data,
                                                 VkDeviceSize bytes,
                                                 VkExternalMemoryHandleTypeFlagBits handle_type =
                                                     VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
                                                 bool transfer_only = false) {
    if (!external_host_memory_enabled || !vkfn::GetMemoryHostPointerPropertiesEXT)
        throw std::runtime_error("AMD Vulkan driver lacks external host memory");
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &host_properties;
    vkfn::GetPhysicalDeviceProperties2(runtime.physical, &properties);
    const VkDeviceSize alignment = host_properties.minImportedHostPointerAlignment;
    if (alignment == 0 || (alignment & (alignment - 1u)) != 0)
        throw std::runtime_error("Invalid external-host pointer alignment");
    const uintptr_t pointer_value = reinterpret_cast<uintptr_t>(data);
    if ((pointer_value & (alignment - 1u)) != 0 || (bytes & (alignment - 1u)) != 0)
        throw std::runtime_error("Expert record is not exactly external-host aligned (alignment " +
            std::to_string(static_cast<uint64_t>(alignment)) + ")");
    void* const imported_pointer = const_cast<uint8_t*>(data);

    VkMemoryHostPointerPropertiesEXT pointer_properties{
        VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    const VkResult query = vkfn::GetMemoryHostPointerPropertiesEXT(
        runtime.device, handle_type, imported_pointer, &pointer_properties);
    VK_CHECK(query);

    Dsv4ImportedRange result;
    result.data_offset = 0;
    Buffer& buffer = result.buffer;
    buffer.size = bytes;
    VkExternalMemoryBufferCreateInfo external{
        VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    external.handleTypes = handle_type;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.pNext = &external;
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (!transfer_only) info.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    if (requirements.size > bytes) {
        vkfn::DestroyBuffer(runtime.device, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
        throw std::runtime_error("Imported expert backing range is too small");
    }
    const uint32_t allowed = requirements.memoryTypeBits &
                             pointer_properties.memoryTypeBits;
    if (allowed == 0)
        throw std::runtime_error("No Vulkan memory type for imported expert record");
    VkImportMemoryHostPointerInfoEXT import{
        VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    import.handleType = handle_type;
    import.pHostPointer = imported_pointer;
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.pNext = &import;
    allocation.allocationSize = bytes;
    bool imported_coherent = false;
    allocation.memoryTypeIndex = choose_memory_type(
        runtime, allowed, imported_coherent);
    if (!imported_coherent) {
        vkfn::DestroyBuffer(runtime.device, buffer.handle, nullptr);
        buffer.handle = VK_NULL_HANDLE;
        throw std::runtime_error(
            "External DeepSeek host memory is not host-coherent");
    }
    buffer.coherent = true;
    VK_CHECK(vkfn::AllocateMemory(runtime.device, &allocation, nullptr,
                                  &buffer.memory));
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes,
                                        active_vulkan_buffer_bytes);
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle,
                                    buffer.memory, 0));
    return result;
}

struct Dsv4Pipelines {
    VkPipeline embedding = VK_NULL_HANDLE;
    VkPipeline embedding_q4g64t = VK_NULL_HANDLE;
    VkPipeline quantize_q8 = VK_NULL_HANDLE;
    VkPipeline quantize_q8_strided_batch = VK_NULL_HANDLE;
    VkPipeline q8_gemv = VK_NULL_HANDLE;
    VkPipeline q8_gemv_residual = VK_NULL_HANDLE;
    VkPipeline q8_grouped_gemv = VK_NULL_HANDLE;
    VkPipeline q4g64t_gemv = VK_NULL_HANDLE;
    VkPipeline q4g64t_gemv_residual = VK_NULL_HANDLE;
    VkPipeline q4g64t_grouped_gemv = VK_NULL_HANDLE;
    VkPipeline q4g64t_gemv_r1x4 = VK_NULL_HANDLE;
    VkPipeline q4g64t_gemv_residual_r1x4 = VK_NULL_HANDLE;
    VkPipeline q4g64t_grouped_gemv_r1x4 = VK_NULL_HANDLE;
    VkPipeline router_top6 = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_fp4 = VK_NULL_HANDLE;
    VkPipeline expert_down_fp4 = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_fp4_r4 = VK_NULL_HANDLE;
    VkPipeline expert_down_fp4_r4 = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_fp4_r1x4 = VK_NULL_HANDLE;
    VkPipeline expert_down_fp4_r1x4 = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_q4g64t = VK_NULL_HANDLE;
    VkPipeline expert_down_q4g64t = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_q4g64t_swar = VK_NULL_HANDLE;
    VkPipeline expert_down_q4g64t_swar = VK_NULL_HANDLE;
    VkPipeline expert_gate_up_q4g64t_bda_batch = VK_NULL_HANDLE;
    VkPipeline expert_down_q4g64t_bda_batch = VK_NULL_HANDLE;
    VkPipeline reduce_experts = VK_NULL_HANDLE;
    VkPipeline rmsnorm = VK_NULL_HANDLE;
    VkPipeline swiglu = VK_NULL_HANDLE;
    VkPipeline hc_mix = VK_NULL_HANDLE;
    VkPipeline hc_sinkhorn = VK_NULL_HANDLE;
    VkPipeline hc_pre = VK_NULL_HANDLE;
    VkPipeline hc_mix_sinkhorn_fused = VK_NULL_HANDLE;
    VkPipeline hc_pre_rmsnorm_fused = VK_NULL_HANDLE;
    VkPipeline rmsnorm_rope_fused = VK_NULL_HANDLE;
    VkPipeline hc_post = VK_NULL_HANDLE;
    VkPipeline hc_head = VK_NULL_HANDLE;
    VkPipeline partial_rope = VK_NULL_HANDLE;
    VkPipeline attention_short = VK_NULL_HANDLE;
    VkPipeline compress_ratio4 = VK_NULL_HANDLE;
    VkPipeline greedy_argmax = VK_NULL_HANDLE;
};

struct QuantizePush {
    uint32_t count;
    uint32_t group_size;
    uint32_t packed_words;
    uint32_t scale_u32;
};
struct GemvPush {
    uint32_t rows;
    uint32_t inner;
    uint32_t activation_scale_u32;
    uint32_t output_offset;
};
struct GroupedGemvPush {
    uint32_t groups;
    uint32_t rows_per_group;
    uint32_t inner;
    uint32_t activation_scale_u32;
};
struct RouterPush {
    uint32_t experts;
    uint32_t top_k;
    uint32_t route_scale_bits;
    uint32_t token_flags;
};
struct ExpertPush {
    uint32_t rank;
    uint32_t activation_scale_u32;
    uint32_t swiglu_limit_bits;
    uint32_t unused;
};
struct HcMixPush {
    uint32_t dimension;
    uint32_t hc;
    uint32_t mix_count;
    uint32_t epsilon_bits;
};
struct HcSplitPush {
    uint32_t tokens;
    uint32_t hc;
    uint32_t iterations;
    uint32_t epsilon_bits;
};
struct HcApplyPush {
    uint32_t dimension;
    uint32_t hc;
    uint32_t split_stride;
    uint32_t unused;
};
struct HcFusedPush {
    uint32_t rms_epsilon_bits;
    uint32_t hc_epsilon_bits;
    uint32_t unused0;
    uint32_t unused1;
};
struct RmsPush {
    uint32_t rows;
    uint32_t columns;
    uint32_t epsilon_bits;
    uint32_t output_offset;
};
struct RopePush {
    uint32_t vectors;
    uint32_t head_dimension;
    uint32_t rope_dimension;
    uint32_t position_flags;
};
struct AttentionPush {
    uint32_t heads;
    uint32_t head_dimension;
    uint32_t source_count;
    uint32_t index_offset;
};
struct CompressPush {
    uint32_t group_count;
    uint32_t token_count;
    uint32_t head_dimension;
    uint32_t ratio;
};
struct EmbeddingPush {
    uint32_t vocabulary;
    uint32_t dimension;
    uint32_t row_stride_u32;
    uint32_t output_offset;
};
struct SwigluPush {
    uint32_t count;
    uint32_t limit_bits;
    uint32_t gate_offset;
    uint32_t up_offset;
};
struct ArgmaxPush {
    uint32_t count;
    uint32_t groups;
    uint32_t phase;
    uint32_t unused;
};
static_assert(sizeof(QuantizePush) == 16 && sizeof(GemvPush) == 16 &&
              sizeof(GroupedGemvPush) == 16 && sizeof(RouterPush) == 16 &&
              sizeof(ExpertPush) == 16 && sizeof(HcMixPush) == 16 &&
              sizeof(HcSplitPush) == 16 && sizeof(HcApplyPush) == 16 &&
              sizeof(RmsPush) == 16 && sizeof(RopePush) == 16 &&
              sizeof(AttentionPush) == 16 && sizeof(CompressPush) == 16 &&
              sizeof(EmbeddingPush) == 16 && sizeof(SwigluPush) == 16 &&
              sizeof(ArgmaxPush) == 16);

static VkPipeline create_dsv4_pipeline(const Runtime& runtime,
                                       ComputeResources& resources,
                                       const std::filesystem::path& shader_path,
                                       uint32_t required_subgroup_size) {
    const std::vector<uint32_t> code = read_spirv(shader_path.string());
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * sizeof(uint32_t);
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateShaderModule(runtime.device, &module_info, nullptr, &module));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    subgroup.requiredSubgroupSize = required_subgroup_size;
    VkPipelineShaderStageCreateInfo stage{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pNext = &subgroup;
    stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage;
    pipeline_info.layout = resources.pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateComputePipelines(runtime.device, VK_NULL_HANDLE, 1,
                                           &pipeline_info, nullptr, &pipeline));
    resources.shader_modules.push_back(module);
    resources.pipelines.push_back(pipeline);
    return pipeline;
}

static VkDescriptorSet create_dsv4_set(
    const Runtime& runtime, const ComputeResources& resources,
    const std::array<DescriptorRange, 6>& ranges) {
    VkDescriptorSetAllocateInfo allocation{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = resources.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &resources.descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkfn::AllocateDescriptorSets(runtime.device, &allocation, &set));
    std::array<VkDescriptorBufferInfo, 6> infos{};
    std::array<VkWriteDescriptorSet, 6> writes{};
    for (uint32_t binding = 0; binding < ranges.size(); ++binding) {
        if (!ranges[binding].buffer || ranges[binding].range == 0)
            throw std::runtime_error("DeepSeek descriptor has an empty range");
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
    vkfn::UpdateDescriptorSets(runtime.device, static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    return set;
}

static void dispatch_dsv4(VkCommandBuffer command, const ComputeResources& resources,
                          VkPipeline pipeline, VkDescriptorSet set, const void* push,
                          uint32_t groups_x, uint32_t groups_y = 1) {
    if (!pipeline || !set || groups_x == 0 || groups_y == 0)
        throw std::runtime_error("Invalid bounded DeepSeek dispatch");
    vkfn::CmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkfn::CmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                                resources.pipeline_layout, 0, 1, &set, 0, nullptr);
    vkfn::CmdPushConstants(command, resources.pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, push);
    vkfn::CmdDispatch(command, groups_x, groups_y, 1);
}

class Dsv4KernelLibrary {
public:
    Dsv4KernelLibrary(const Runtime& runtime,
                      const std::filesystem::path& shader_directory)
        : runtime_(runtime), resources_(create_compute_resources(runtime, 8192)),
          // Router alternatives can legally be compiled as predicated loads.
          // Keep the fallback descriptor large enough for the largest optional
          // table so an untaken hash/bias path can never become descriptor OOB.
          dummy_(create_device_buffer(runtime,
              static_cast<VkDeviceSize>(kVocabulary) * kTopK * sizeof(uint32_t))) {
        VkPhysicalDeviceSubgroupSizeControlProperties subgroup{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &subgroup;
        vkfn::GetPhysicalDeviceProperties2(runtime.physical, &properties);
        if (subgroup.minSubgroupSize > 64 || subgroup.maxSubgroupSize < 64 ||
            (subgroup.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0)
            throw std::runtime_error("AMD device cannot provide required wave64 compute");
        const auto load = [&](const char* name) {
            return create_dsv4_pipeline(runtime_, resources_,
                shader_directory / (std::string(name) + ".comp.spv"), 64u);
        };
        pipelines_.embedding = load("dsv4_embedding");
        pipelines_.embedding_q4g64t = load("dsv4_embedding_q4g64t");
        pipelines_.quantize_q8 = load("dsv4_quantize_q8");
        pipelines_.quantize_q8_strided_batch = load("dsv4_quantize_q8_strided_batch");
        pipelines_.q8_gemv = load("dsv4_q8_gemv");
        pipelines_.q8_gemv_residual = load("dsv4_q8_gemv_residual");
        pipelines_.q8_grouped_gemv = load("dsv4_q8_grouped_gemv");
        pipelines_.q4g64t_gemv = load("dsv4_q4g64t_gemv");
        pipelines_.q4g64t_gemv_residual = load("dsv4_q4g64t_gemv_residual");
        pipelines_.q4g64t_grouped_gemv = load("dsv4_q4g64t_grouped_gemv");
        pipelines_.q4g64t_gemv_r1x4 = load("dsv4_q4g64t_gemv_r1x4");
        pipelines_.q4g64t_gemv_residual_r1x4 = load("dsv4_q4g64t_gemv_residual_r1x4");
        pipelines_.q4g64t_grouped_gemv_r1x4 = load("dsv4_q4g64t_grouped_gemv_r1x4");
        pipelines_.router_top6 = load("dsv4_router_top6");
        pipelines_.expert_gate_up_fp4 = load("dsv4_expert_gate_up_fp4");
        pipelines_.expert_down_fp4 = load("dsv4_expert_down_fp4");
        pipelines_.expert_gate_up_fp4_r4 = load("dsv4_expert_gate_up_fp4_r4");
        pipelines_.expert_down_fp4_r4 = load("dsv4_expert_down_fp4_r4");
        pipelines_.expert_gate_up_fp4_r1x4 = load("dsv4_expert_gate_up_fp4_r1x4");
        pipelines_.expert_down_fp4_r1x4 = load("dsv4_expert_down_fp4_r1x4");
        pipelines_.expert_gate_up_q4g64t = load("dsv4_expert_gate_up_q4g64t");
        pipelines_.expert_down_q4g64t = load("dsv4_expert_down_q4g64t");
        pipelines_.expert_gate_up_q4g64t_swar = load("dsv4_expert_gate_up_q4g64t_swar");
        pipelines_.expert_down_q4g64t_swar = load("dsv4_expert_down_q4g64t_swar");
        pipelines_.expert_gate_up_q4g64t_bda_batch =
            load("dsv4_expert_gate_up_q4g64t_bda_batch");
        pipelines_.expert_down_q4g64t_bda_batch =
            load("dsv4_expert_down_q4g64t_bda_batch");
        pipelines_.reduce_experts = load("dsv4_reduce_experts");
        pipelines_.rmsnorm = load("dsv4_rmsnorm");
        pipelines_.swiglu = load("dsv4_swiglu");
        pipelines_.hc_mix = load("dsv4_hc_mix");
        pipelines_.hc_sinkhorn = load("dsv4_hc_sinkhorn");
        pipelines_.hc_pre = load("dsv4_hc_pre");
        pipelines_.hc_mix_sinkhorn_fused = load("dsv4_hc_mix_sinkhorn_fused");
        pipelines_.hc_pre_rmsnorm_fused = load("dsv4_hc_pre_rmsnorm_fused");
        pipelines_.rmsnorm_rope_fused = load("dsv4_rmsnorm_rope_fused");
        pipelines_.hc_post = load("dsv4_hc_post");
        pipelines_.hc_head = load("dsv4_hc_head");
        pipelines_.partial_rope = load("dsv4_partial_rope");
        pipelines_.attention_short = load("dsv4_attention_short");
        pipelines_.compress_ratio4 = load("dsv4_compress_ratio4");
        pipelines_.greedy_argmax = load("dsv4_greedy_argmax");
    }

    Dsv4KernelLibrary(const Dsv4KernelLibrary&) = delete;
    Dsv4KernelLibrary& operator=(const Dsv4KernelLibrary&) = delete;
    ~Dsv4KernelLibrary() {
        for (VkPipeline pipeline : resources_.pipelines)
            vkfn::DestroyPipeline(runtime_.device, pipeline, nullptr);
        for (VkShaderModule module : resources_.shader_modules)
            vkfn::DestroyShaderModule(runtime_.device, module, nullptr);
        if (resources_.descriptor_pool)
            vkfn::DestroyDescriptorPool(runtime_.device,
                                        resources_.descriptor_pool, nullptr);
        if (resources_.pipeline_layout)
            vkfn::DestroyPipelineLayout(runtime_.device,
                                        resources_.pipeline_layout, nullptr);
        if (resources_.descriptor_layout)
            vkfn::DestroyDescriptorSetLayout(runtime_.device,
                                             resources_.descriptor_layout, nullptr);
        destroy_buffer(runtime_, dummy_);
    }

    VkDescriptorSet set(const std::array<DescriptorRange, 6>& ranges) const {
        return create_dsv4_set(runtime_, resources_, ranges);
    }
    void update_binding(VkDescriptorSet set, uint32_t binding,
                        const DescriptorRange& range) const {
        if (!set || binding >= 6 || !range.buffer || range.range == 0)
            throw std::runtime_error("Invalid DeepSeek descriptor update");
        VkDescriptorBufferInfo info{};
        info.buffer = range.buffer;
        info.offset = range.offset;
        info.range = range.range;
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkfn::UpdateDescriptorSets(runtime_.device, 1, &write, 0, nullptr);
    }
    DescriptorRange dummy() const { return whole(dummy_); }
    const ComputeResources& resources() const { return resources_; }
    const Dsv4Pipelines& pipelines() const { return pipelines_; }

private:
    const Runtime& runtime_;
    ComputeResources resources_{};
    Buffer dummy_{};
    Dsv4Pipelines pipelines_{};
};

class FiniteQueue {
public:
    FiniteQueue(const Runtime& runtime, VkQueue queue)
        : runtime_(runtime), queue_(queue) {
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = runtime.queue_family;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pool_info, nullptr, &pool_));
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = pool_;
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &allocation, &command_));
        VkSemaphoreTypeCreateInfo timeline_type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_type;
        VK_CHECK(vkfn::CreateSemaphore(runtime.device, &semaphore_info, nullptr, &timeline_));
    }

    FiniteQueue(const FiniteQueue&) = delete;
    FiniteQueue& operator=(const FiniteQueue&) = delete;
    ~FiniteQueue() {
        if (timeline_) vkfn::DestroySemaphore(runtime_.device, timeline_, nullptr);
        if (pool_) vkfn::DestroyCommandPool(runtime_.device, pool_, nullptr);
    }

    uint64_t submit(const std::function<void(VkCommandBuffer)>& record,
                    VkSemaphore wait_semaphore = VK_NULL_HANDLE,
                    uint64_t wait_value = 0,
                    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        if (submitted_ != completed_) wait(submitted_);
        VK_CHECK(vkfn::ResetCommandBuffer(command_, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkfn::BeginCommandBuffer(command_, &begin));
        record(command_);
        VK_CHECK(vkfn::EndCommandBuffer(command_));
        const uint64_t signal_value = ++submitted_;
        VkTimelineSemaphoreSubmitInfo timeline_info{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        if (wait_semaphore) {
            timeline_info.waitSemaphoreValueCount = 1;
            timeline_info.pWaitSemaphoreValues = &wait_value;
        }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.pNext = &timeline_info;
        if (wait_semaphore) {
            submit.waitSemaphoreCount = 1;
            submit.pWaitSemaphores = &wait_semaphore;
            submit.pWaitDstStageMask = &wait_stage;
        }
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &timeline_;
        VK_CHECK(vkfn::QueueSubmit(queue_, 1, &submit, VK_NULL_HANDLE));
        return signal_value;
    }

    void wait(uint64_t value) {
        if (value <= completed_) return;
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &timeline_;
        info.pValues = &value;
        // A bounded wait makes a lost/hung workload an explicit failure.  It is
        // not a TDR workaround: normal decode layer segments complete in far
        // less than this and contain no unbounded device-side loops.
        const VkResult result = vkfn::WaitSemaphores(runtime_.device, &info,
                                                      10ull * 1000 * 1000 * 1000);
        if (result == VK_TIMEOUT)
            throw std::runtime_error("Vulkan layer segment exceeded 10 seconds");
        if (result != VK_SUCCESS)
            throw std::runtime_error("Vulkan timeline wait failed");
        completed_ = std::max(completed_, value);
    }

    VkSemaphore semaphore() const { return timeline_; }

private:
    const Runtime& runtime_;
    VkQueue queue_ = VK_NULL_HANDLE;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkCommandBuffer command_ = VK_NULL_HANDLE;
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;
};

static experiment::QueueRingApi finite_queue_ring_api() {
    experiment::QueueRingApi api{};
    api.create_command_pool = vkfn::CreateCommandPool;
    api.destroy_command_pool = vkfn::DestroyCommandPool;
    api.allocate_command_buffers = vkfn::AllocateCommandBuffers;
    api.reset_command_buffer = vkfn::ResetCommandBuffer;
    api.begin_command_buffer = vkfn::BeginCommandBuffer;
    api.end_command_buffer = vkfn::EndCommandBuffer;
    api.create_semaphore = vkfn::CreateSemaphore;
    api.destroy_semaphore = vkfn::DestroySemaphore;
    api.queue_submit = vkfn::QueueSubmit;
    api.wait_semaphores = vkfn::WaitSemaphores;
    return api;
}

struct CacheSlot {
    int32_t expert = -1;
    uint64_t age = 0;
};

// Opt-in scheduling metadata for every routed expert.  It mirrors the real
// cache namespaces and finite transfer timelines; it never chooses a route or
// victim, so enabling it cannot change model results or cache policy.
class ExpertBlueprint {
public:
    enum Tier : uint32_t {
        kNvme = 1u << 0u,
        kRam = 1u << 1u,
        kVram = 1u << 2u,
        kStaging = 1u << 3u,
    };
    enum class TransferState : uint8_t { idle, nvme_read, host_to_vram };
    struct Entry {
        uint32_t tiers = kNvme;
        uint32_t vram_slot = UINT32_MAX;
        uint32_t ram_location = UINT32_MAX;
        uint32_t staging_slot = UINT32_MAX;
        uint64_t request_count = 0;
        uint64_t last_request_tick = 0;
        uint64_t last_state_tick = 0;
        uint64_t last_request_ns = 0;
        uint64_t last_state_ns = 0;
        uint64_t transfer_timeline = 0;
        uint64_t estimated_ready_ns = 0;
        uint64_t nvme_data_offset = 0;
        uint64_t nvme_read_cost_bytes = kExpertRecordBytes;
        double last_transfer_seconds = 0.0;
        float predicted_probability = 0.0f;
        uint32_t nvme_copy = UINT32_MAX;
        uint32_t nvme_bundle = UINT32_MAX;
        TransferState transfer_state = TransferState::idle;
    };
    struct Metrics {
        uint64_t route_layers = 0;
        uint64_t top6_vram_ready_layers = 0;
        uint64_t top6_vram_or_ram_ready_layers = 0;
        uint64_t layers_with_ssd_miss = 0;
        uint64_t expert_read_count = 0;
        uint64_t expert_read_bytes = 0;
        uint64_t h2d_count = 0;
        uint64_t h2d_bytes = 0;
        uint64_t idle_entries = 0;
        uint64_t nvme_read_entries = 0;
        uint64_t h2d_entries = 0;
        uint64_t pending_transfers = 0;
        uint64_t maximum_timeline = 0;
        double estimated_outstanding_seconds = 0.0;
    };

    ExpertBlueprint()
        : enabled_(std::getenv("DSV4_EXPERT_BLUEPRINT") != nullptr) {
        if (enabled_) entries_ = std::make_unique<Table>();
    }
    bool enabled() const { return enabled_; }

    static uint32_t local_ram_location(uint32_t layer, uint32_t slot) {
        return 0x80000000u | (layer << 16u) | slot;
    }

    void request(uint32_t layer, uint32_t expert) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        ++item.request_count;
        item.last_request_tick = ++tick_;
        item.last_request_ns = now_ns();
    }

    bool vram_ready(uint32_t layer, uint32_t expert, uint32_t slot) const {
        if (!enabled_) return false;
        const Entry& item = at(layer, expert);
        return (item.tiers & kVram) != 0u && item.vram_slot == slot &&
               item.transfer_state == TransferState::idle;
    }
    bool ram_ready(uint32_t layer, uint32_t expert, uint32_t location) const {
        if (!enabled_) return false;
        const Entry& item = at(layer, expert);
        return (item.tiers & kRam) != 0u && item.ram_location == location;
    }
    void touch_vram(uint32_t layer, uint32_t expert, uint32_t slot) {
        if (!enabled_) return;
        if (!vram_ready(layer, expert, slot))
            throw std::runtime_error("ExpertBlueprint VRAM cache drift");
        touch(at(layer, expert));
    }
    void touch_ram(uint32_t layer, uint32_t expert, uint32_t location) {
        if (!enabled_) return;
        if (!ram_ready(layer, expert, location))
            throw std::runtime_error("ExpertBlueprint RAM cache drift");
        touch(at(layer, expert));
    }

    void reserve_vram(uint32_t layer, uint32_t expert, uint32_t slot) {
        if (!enabled_) return;
        const uint32_t key = make_key(layer, expert);
        const auto prior = vram_owners_.find(slot);
        if (prior != vram_owners_.end() && prior->second != key) {
            Entry& old = at_key(prior->second);
            old.tiers &= ~kVram;
            old.vram_slot = UINT32_MAX;
            old.transfer_state = TransferState::idle;
            touch(old);
        }
        Entry& item = at(layer, expert);
        if (item.vram_slot != UINT32_MAX && item.vram_slot != slot)
            vram_owners_.erase(item.vram_slot);
        item.tiers &= ~kVram;
        item.vram_slot = slot;
        vram_owners_[slot] = key;
        touch(item);
    }
    void evict_vram(uint32_t layer, uint32_t expert, uint32_t slot) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        if (item.vram_slot != slot)
            throw std::runtime_error("ExpertBlueprint VRAM eviction drift");
        item.tiers &= ~kVram;
        item.vram_slot = UINT32_MAX;
        item.transfer_state = TransferState::idle;
        vram_owners_.erase(slot);
        touch(item);
    }

    void reserve_ram(uint32_t layer, uint32_t expert, uint32_t location,
                     bool ready) {
        if (!enabled_) return;
        const uint32_t key = make_key(layer, expert);
        const auto prior = ram_owners_.find(location);
        if (prior != ram_owners_.end() && prior->second != key) {
            Entry& old = at_key(prior->second);
            old.tiers &= ~kRam;
            old.ram_location = UINT32_MAX;
            touch(old);
        }
        Entry& item = at(layer, expert);
        if (item.ram_location != UINT32_MAX && item.ram_location != location)
            ram_owners_.erase(item.ram_location);
        item.ram_location = location;
        if (ready) item.tiers |= kRam;
        else item.tiers &= ~kRam;
        ram_owners_[location] = key;
        touch(item);
    }
    void evict_ram(uint32_t layer, uint32_t expert, uint32_t location) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        if (item.ram_location != location)
            throw std::runtime_error("ExpertBlueprint RAM eviction drift");
        item.tiers &= ~kRam;
        item.ram_location = UINT32_MAX;
        ram_owners_.erase(location);
        touch(item);
    }
    void swap_ram(uint32_t first_layer, uint32_t first_expert,
                  uint32_t first_location, int32_t second_layer,
                  int32_t second_expert, uint32_t second_location) {
        if (!enabled_) return;
        Entry& first = at(first_layer, first_expert);
        if (!ram_ready(first_layer, first_expert, first_location))
            throw std::runtime_error("ExpertBlueprint RAM promotion drift");
        ram_owners_.erase(first_location);
        if (second_layer >= 0) {
            Entry& second = at(static_cast<uint32_t>(second_layer),
                               static_cast<uint32_t>(second_expert));
            if (!ram_ready(static_cast<uint32_t>(second_layer),
                           static_cast<uint32_t>(second_expert), second_location))
                throw std::runtime_error("ExpertBlueprint RAM demotion drift");
            second.ram_location = first_location;
            ram_owners_[first_location] = make_key(
                static_cast<uint32_t>(second_layer),
                static_cast<uint32_t>(second_expert));
            touch(second);
        }
        first.ram_location = second_location;
        ram_owners_[second_location] = make_key(first_layer, first_expert);
        touch(first);
    }

    void begin_disk_read(uint32_t layer, uint32_t expert,
                         uint32_t staging_slot) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        item.tiers |= kStaging;
        item.staging_slot = staging_slot;
        item.transfer_state = TransferState::nvme_read;
        item.last_transfer_seconds = 0.0;
        item.last_state_ns = now_ns();
        item.estimated_ready_ns = item.last_state_ns + seconds_ns(read_ema_seconds_);
        item.last_state_tick = ++tick_;
        ++expert_read_count_;
        expert_read_bytes_ += kExpertRecordBytes;
    }
    void select_nvme_copy(uint32_t layer, uint32_t expert, uint32_t copy,
                          uint32_t bundle, uint64_t offset, uint64_t read_cost,
                          float predicted_probability) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        item.nvme_copy = copy;
        item.nvme_bundle = bundle;
        item.nvme_data_offset = offset;
        item.nvme_read_cost_bytes = read_cost;
        item.predicted_probability = predicted_probability;
        touch(item);
    }
    void complete_disk_read(uint32_t layer, uint32_t expert,
                            uint32_t ram_location, bool persistent_ram) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        const uint64_t completed = now_ns();
        const double elapsed = seconds_between(item.last_state_ns, completed);
        update_ema(read_ema_seconds_, elapsed);
        item.last_transfer_seconds = elapsed;
        item.transfer_state = TransferState::idle;
        item.estimated_ready_ns = completed;
        if (persistent_ram) reserve_ram(layer, expert, ram_location, true);
        else item.tiers |= kStaging;
        touch(item);
    }
    void mark_staging(uint32_t layer, uint32_t expert, uint32_t staging_slot) {
        if (!enabled_) return;
        Entry& item = at(layer, expert);
        item.tiers |= kStaging;
        item.staging_slot = staging_slot;
        touch(item);
    }
    void begin_h2d(uint32_t layer, uint32_t expert, uint32_t vram_slot,
                   uint32_t staging_slot, uint64_t timeline) {
        if (!enabled_) return;
        reserve_vram(layer, expert, vram_slot);
        Entry& item = at(layer, expert);
        if (staging_slot != UINT32_MAX) {
            item.tiers |= kStaging;
            item.staging_slot = staging_slot;
        }
        item.transfer_state = TransferState::host_to_vram;
        item.transfer_timeline = timeline;
        item.last_state_ns = now_ns();
        item.estimated_ready_ns = item.last_state_ns + seconds_ns(h2d_ema_seconds_);
        item.last_state_tick = ++tick_;
        pending_.push_back({make_key(layer, expert), timeline, item.last_state_ns});
        ++h2d_count_;
        h2d_bytes_ += kExpertRecordBytes;
        maximum_timeline_ = std::max(maximum_timeline_, timeline);
    }
    void complete_timeline(uint64_t timeline) {
        if (!enabled_ || timeline == 0) return;
        const uint64_t completed = now_ns();
        size_t output = 0;
        for (const Pending& pending : pending_) {
            if (pending.timeline <= timeline) {
                Entry& item = at_key(pending.key);
                const double elapsed = seconds_between(pending.started_ns, completed);
                update_ema(h2d_ema_seconds_, elapsed);
                item.last_transfer_seconds = elapsed;
                item.tiers |= kVram;
                item.tiers &= ~kStaging;
                item.staging_slot = UINT32_MAX;
                item.transfer_state = TransferState::idle;
                item.estimated_ready_ns = completed;
                touch(item);
            } else {
                pending_[output++] = pending;
            }
        }
        pending_.resize(output);
    }

    void record_route(bool all_vram, bool all_vram_or_ram, bool ssd_miss) {
        if (!enabled_) return;
        ++route_layers_;
        top6_vram_ready_layers_ += all_vram;
        top6_vram_or_ram_ready_layers_ += all_vram_or_ram;
        layers_with_ssd_miss_ += ssd_miss;
    }
    void reset_metrics() {
        if (!enabled_) return;
        route_layers_ = top6_vram_ready_layers_ = 0;
        top6_vram_or_ram_ready_layers_ = layers_with_ssd_miss_ = 0;
        expert_read_count_ = expert_read_bytes_ = 0;
        h2d_count_ = h2d_bytes_ = maximum_timeline_ = 0;
    }
    Metrics metrics() const {
        Metrics result;
        if (!enabled_) return result;
        result.route_layers = route_layers_;
        result.top6_vram_ready_layers = top6_vram_ready_layers_;
        result.top6_vram_or_ram_ready_layers = top6_vram_or_ram_ready_layers_;
        result.layers_with_ssd_miss = layers_with_ssd_miss_;
        result.expert_read_count = expert_read_count_;
        result.expert_read_bytes = expert_read_bytes_;
        result.h2d_count = h2d_count_;
        result.h2d_bytes = h2d_bytes_;
        result.pending_transfers = pending_.size();
        result.maximum_timeline = maximum_timeline_;
        const uint64_t now = now_ns();
        for (const auto& layer : *entries_)
            for (const Entry& item : layer) {
                if (item.transfer_state == TransferState::idle) ++result.idle_entries;
                else if (item.transfer_state == TransferState::nvme_read)
                    ++result.nvme_read_entries;
                else ++result.h2d_entries;
                if (item.transfer_state != TransferState::idle &&
                    item.estimated_ready_ns > now) {
                    result.estimated_outstanding_seconds = std::max(
                        result.estimated_outstanding_seconds,
                        seconds_between(now, item.estimated_ready_ns));
                }
            }
        return result;
    }

private:
    using Table = std::array<std::array<Entry, kExperts>, kLayers>;
    struct Pending { uint32_t key; uint64_t timeline; uint64_t started_ns; };
    static uint32_t make_key(uint32_t layer, uint32_t expert) {
        if (layer >= kLayers || expert >= kExperts)
            throw std::runtime_error("Invalid ExpertBlueprint key");
        return layer * kExperts + expert;
    }
    Entry& at(uint32_t layer, uint32_t expert) {
        make_key(layer, expert);
        return (*entries_)[layer][expert];
    }
    const Entry& at(uint32_t layer, uint32_t expert) const {
        make_key(layer, expert);
        return (*entries_)[layer][expert];
    }
    Entry& at_key(uint32_t key) {
        return at(key / kExperts, key % kExperts);
    }
    static uint64_t now_ns() {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    static uint64_t seconds_ns(double seconds) {
        return static_cast<uint64_t>(std::max(0.0, seconds) * 1.0e9);
    }
    static double seconds_between(uint64_t begin, uint64_t end) {
        return end >= begin ? static_cast<double>(end - begin) / 1.0e9 : 0.0;
    }
    static void update_ema(double& average, double sample) {
        if (sample <= 0.0) return;
        average = average <= 0.0 ? sample : average * 0.875 + sample * 0.125;
    }
    void touch(Entry& item) {
        item.last_state_tick = ++tick_;
        item.last_state_ns = now_ns();
    }

    bool enabled_ = false;
    std::unique_ptr<Table> entries_;
    std::unordered_map<uint32_t, uint32_t> vram_owners_;
    std::unordered_map<uint32_t, uint32_t> ram_owners_;
    std::vector<Pending> pending_;
    uint64_t tick_ = 0;
    double read_ema_seconds_ = 0.004;
    double h2d_ema_seconds_ = 0.002;
    uint64_t route_layers_ = 0;
    uint64_t top6_vram_ready_layers_ = 0;
    uint64_t top6_vram_or_ram_ready_layers_ = 0;
    uint64_t layers_with_ssd_miss_ = 0;
    uint64_t expert_read_count_ = 0;
    uint64_t expert_read_bytes_ = 0;
    uint64_t h2d_count_ = 0;
    uint64_t h2d_bytes_ = 0;
    uint64_t maximum_timeline_ = 0;
};

// Causal, opt-in rank-1 predictor for the next learned layer.  It learns only
// from routes which have already completed in this process.  The prediction is
// never substituted for the authoritative router result: it is only a bounded
// RAM admission/retention hint, and only above a trace-calibrated confidence.
class AdjacentRank1Predictor {
public:
    struct Prediction {
        uint32_t layer = UINT32_MAX;
        uint32_t expert = UINT32_MAX;
        float confidence = 0.0f;
        bool valid = false;
    };
    struct Metrics {
        uint64_t qualified = 0;
        uint64_t route_hits = 0;
        uint64_t host_hints = 0;
        uint64_t learned_edges = 0;
        float threshold = 0.0f;
    };

    AdjacentRank1Predictor()
        : enabled_(std::getenv("DSV4_ADJACENT_RANK1_HINT") != nullptr) {
        if (!enabled_) return;
        if (const char* configured =
                std::getenv("DSV4_ADJACENT_RANK1_CONFIDENCE"))
            threshold_ = std::stof(configured);
        if (!std::isfinite(threshold_) || threshold_ < 0.5f || threshold_ > 1.0f)
            throw std::runtime_error(
                "DSV4_ADJACENT_RANK1_CONFIDENCE must be in [0.5,1]");
        affinity_.resize(static_cast<size_t>(kLayers) * kExperts * kExperts);
    }

    bool enabled() const { return enabled_; }

    Prediction observe(uint32_t layer,
                       const std::array<uint32_t, kTopK>& route) {
        if (!enabled_) return {};
        if (pending_.valid && pending_.layer == layer) {
            for (uint32_t expert : route)
                if (expert == pending_.expert) {
                    ++route_hits_;
                    break;
                }
        }
        pending_ = {};

        // Learn the just-observed adjacent edge after evaluating it.  Thus a
        // prediction for token T never sees the target route from token T.
        if (previous_layer_ != UINT32_MAX && previous_layer_ + 1u == layer) {
            for (uint32_t source : previous_route_) {
                uint16_t& observations = seen_[layer][source];
                if (observations != UINT16_MAX) ++observations;
                for (uint32_t target : route) {
                    uint16_t& count = affinity_[index(layer, source, target)];
                    if (count != UINT16_MAX) ++count;
                    ++learned_edges_;
                }
            }
        }
        for (uint32_t expert : route) {
            uint16_t& count = marginal_[layer][expert];
            if (count != UINT16_MAX) ++count;
        }
        if (layer_observations_[layer] != UINT16_MAX)
            ++layer_observations_[layer];
        previous_route_ = route;
        previous_layer_ = layer;

        const uint32_t destination = layer + 1u;
        if (destination >= kLayers || destination < kHashLayers ||
            layer_observations_[destination] == 0u)
            return {};
        uint32_t best = UINT32_MAX;
        float best_probability = -1.0f;
        for (uint32_t candidate = 0; candidate < kExperts; ++candidate) {
            const float prior = static_cast<float>(marginal_[destination][candidate]) /
                                layer_observations_[destination];
            float probability = 0.0f;
            for (uint32_t source : route) {
                const float denominator =
                    static_cast<float>(seen_[destination][source]) + 2.0f;
                probability +=
                    (static_cast<float>(affinity_[index(
                         destination, source, candidate)]) + 2.0f * prior) /
                    denominator;
            }
            probability /= static_cast<float>(kTopK);
            if (probability > best_probability) {
                best_probability = probability;
                best = candidate;
            }
        }
        if (best == UINT32_MAX || best_probability < threshold_) return {};
        pending_ = {destination, best, best_probability, true};
        ++qualified_;
        return pending_;
    }

    void record_host_hint(bool installed) {
        if (enabled_ && pending_.valid && installed) ++host_hints_;
    }
    void reset_metrics() {
        qualified_ = route_hits_ = host_hints_ = learned_edges_ = 0;
    }
    Metrics metrics() const {
        return {qualified_, route_hits_, host_hints_, learned_edges_, threshold_};
    }

private:
    static size_t index(uint32_t layer, uint32_t source, uint32_t target) {
        return (static_cast<size_t>(layer) * kExperts + source) * kExperts + target;
    }
    bool enabled_ = false;
    float threshold_ = 0.85f;
    std::vector<uint16_t> affinity_;
    std::array<std::array<uint16_t, kExperts>, kLayers> marginal_{};
    std::array<std::array<uint16_t, kExperts>, kLayers> seen_{};
    std::array<uint16_t, kLayers> layer_observations_{};
    std::array<uint32_t, kTopK> previous_route_{};
    uint32_t previous_layer_ = UINT32_MAX;
    Prediction pending_{};
    uint64_t qualified_ = 0;
    uint64_t route_hits_ = 0;
    uint64_t host_hints_ = 0;
    uint64_t learned_edges_ = 0;
};

class GlobalExpertCache {
public:
    using BatchSlots = std::array<std::array<uint32_t, kTopK>, 5>;
    struct SetPolicyMetrics {
        uint64_t route_sets = 0;
        uint64_t complete_set_ready = 0;
        uint64_t one_missing_sets = 0;
        uint64_t initial_hits = 0;
        uint64_t requested_experts = 0;
        uint64_t total_misses = 0;
        uint64_t maximum_set_misses = 0;
        uint64_t victim_decisions = 0;
        uint64_t changed_victims = 0;
        uint64_t coactivation_saved = 0;
        uint64_t adjacent_affinity_saved = 0;
        uint64_t adjacent_prediction_sets = 0;
        uint64_t adjacent_prediction_hits = 0;
        uint64_t adjacent_prediction_targets = 0;
        uint64_t coactivation_updates = 0;
        uint64_t adjacent_updates = 0;
    };

    GlobalExpertCache(const Runtime& runtime, uint32_t requested,
                      ExpertBlueprint* blueprint = nullptr)
        : runtime_(runtime), blueprint_(blueprint) {
        if (requested == 0 || requested > kPersistentExpertSlotsPerLayer)
            throw std::runtime_error("Invalid requested device expert-cache capacity");
        capacities_.fill(requested);
        const bool fused_profile =
            std::getenv("DSV4_FUSED_PROFILED_DEVICE_CACHE") != nullptr;
        const bool compact_profile =
            std::getenv("DSV4_COMPACT_PROFILED_DEVICE_CACHE") != nullptr;
        const bool mid_profile =
            std::getenv("DSV4_MID_PROFILED_DEVICE_CACHE") != nullptr;
        const bool high_profile =
            std::getenv("DSV4_HIGH_PROFILED_DEVICE_CACHE") != nullptr;
        const bool nearfull_profile =
            std::getenv("DSV4_NEARFULL_PROFILED_DEVICE_CACHE") != nullptr;
        const bool maximum_profile =
            std::getenv("DSV4_MAX_PROFILED_DEVICE_CACHE") != nullptr;
        const bool suite_profile =
            std::getenv("DSV4_SUITE_PROFILED_DEVICE_CACHE") != nullptr;
        const bool lab_profile =
            std::getenv("DSV4_LAB_PROFILED_DEVICE_CACHE") != nullptr;
        set_policy_ = std::getenv("DSV4_TOP6_SET_POLICY") != nullptr;
        if (set_policy_ && !suite_profile)
            throw std::runtime_error(
                "DSV4_TOP6_SET_POLICY requires DSV4_SUITE_PROFILED_DEVICE_CACHE");
        if (set_policy_ && (!blueprint_ || !blueprint_->enabled()))
            throw std::runtime_error(
                "DSV4_TOP6_SET_POLICY requires DSV4_EXPERT_BLUEPRINT");
        profiled_partition_ =
            std::getenv("DSV4_PROFILED_DEVICE_CACHE") != nullptr ||
            fused_profile || compact_profile || mid_profile || high_profile ||
            nearfull_profile || maximum_profile || suite_profile || lab_profile;
        if (lab_profile) {
            // Five-prompt chat/reasoning/coding trace suite, 16-GiB host tier.
            // Exact same 460-record VRAM budget as the retained suite profile.
            constexpr std::array<uint32_t, kLayers> profiled{
                7,6,6,10,10,15,11,15,10,9,9,11,14,16,12,14,14,9,11,14,7,9,
                10,12,10,10,10,10,11,9,9,10,9,11,12,11,10,9,14,10,12,12,10};
            capacities_ = profiled;
        } else if (suite_profile) {
            // Joint chat/reasoning/code route profile: exactly 460 records,
            // with every layer retaining at least one complete Top-6 set.
            constexpr std::array<uint32_t, kLayers> profiled{
                6,6,6,11,11,16,9,13,11,10,11,14,14,13,16,16,12,14,14,9,7,9,
                9,10,9,10,10,9,12,8,9,10,13,11,11,10,10,11,8,9,11,10,12};
            capacities_ = profiled;
        } else if (maximum_profile) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,7,8,12,11,15,15,11,12,12,12,11,13,14,18,17,17,14,13,10,13,
                9,9,12,15,10,11,10,12,8,10,10,14,16,11,13,10,12,11,13,11,11,9};
            capacities_ = profiled;
        } else if (nearfull_profile) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,6,8,12,11,15,15,11,10,10,10,11,13,14,18,17,17,14,11,10,13,
                9,9,12,15,10,11,10,12,8,10,10,14,12,11,13,10,12,8,9,11,11,9};
            capacities_ = profiled;
        } else if (high_profile) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,6,8,9,9,13,15,11,10,10,10,11,13,14,16,12,16,9,9,9,11,9,9,
                12,15,10,11,10,12,8,9,10,14,12,11,11,10,12,8,8,10,11,9};
            capacities_ = profiled;
        } else if (mid_profile) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,6,6,9,9,7,11,10,10,9,10,11,12,11,11,12,13,8,9,9,7,7,8,9,
                9,10,10,10,12,8,8,8,8,11,9,10,9,12,8,8,10,10,8};
            capacities_ = profiled;
        } else if (compact_profile) {
            // Same 344 records as eight slots/layer, redistributed from the
            // fused-model route trace to preserve more useful experts.
            constexpr std::array<uint32_t, kLayers> profiled{
                8,6,6,7,7,7,8,8,9,9,8,9,10,10,11,9,11,7,9,9,7,7,7,8,9,9,
                8,8,8,8,8,8,8,7,9,6,8,7,7,7,7,7,8};
            capacities_ = profiled;
        } else if (fused_profile) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,7,8,12,11,15,15,11,12,12,12,11,13,14,18,17,17,14,13,10,13,9,
                9,12,15,10,11,14,12,8,10,10,14,16,11,16,10,12,11,12,11,11,9};
            capacities_ = profiled;
        } else if (profiled_partition_) {
            constexpr std::array<uint32_t, kLayers> profiled{
                8,7,8,12,9,13,14,11,11,12,10,11,14,14,12,19,14,15,11,9,15,9,
                9,13,12,9,11,15,13,15,9,10,13,12,17,16,10,12,13,13,15,12,9};
            capacities_ = profiled;
        }
        chunks_.reserve(kLayers);
        for (uint32_t chunk = 0; chunk < kLayers; ++chunk) {
            bases_[chunk] = total_slots_;
            chunks_.push_back(create_device_buffer(runtime_,
                static_cast<VkDeviceSize>(capacities_[chunk]) * kExpertRecordBytes));
            total_slots_ += capacities_[chunk];
        }
        if (total_slots_ <= kTopK)
            throw std::runtime_error("DeepSeek global expert cache is too small");
        if (set_policy_) {
            if (total_slots_ != 460u)
                throw std::runtime_error(
                    "Top-6 set policy requires the exact 460-slot suite profile");
            const size_t affinity_entries = static_cast<size_t>(kLayers) *
                kExperts * kExperts;
            same_layer_affinity_.resize(affinity_entries);
            adjacent_layer_affinity_.resize(affinity_entries);
        }
        global_policy_ = std::getenv("DSV4_GLOBAL_DEVICE_CACHE") != nullptr;
        if (profiled_partition_ && global_policy_)
            throw std::runtime_error("Profiled and global device caches are incompatible");
        persistent_slots_ = global_policy_ ? total_slots_ - kTopK : total_slots_;
        entries_.resize(persistent_slots_);
        for (auto& layer_locations : locations_) layer_locations.fill(UINT32_MAX);
        force_admission_ = std::getenv("DSV4_FORCE_CACHE_ADMISSION") != nullptr;
    }
    ~GlobalExpertCache() {
        for (Buffer& chunk : chunks_) destroy_buffer(runtime_, chunk);
    }

    std::array<uint32_t, kTopK> resolve(uint32_t layer,
                                       const std::array<uint32_t, kTopK>& experts,
                                       std::vector<uint32_t>& missing_ranks,
                                       std::vector<uint32_t>& promotion_ranks) {
        std::array<uint32_t, kTopK> result{};
        result.fill(UINT32_MAX);
        std::vector<bool> reserved(persistent_slots_, false);
        for (uint32_t expert : experts) {
            if (expert >= kExperts)
                throw std::runtime_error("Router selected an invalid expert");
            ++frequency_[layer][expert];
            if (blueprint_) blueprint_->request(layer, expert);
        }
        if (!global_policy_)
            return resolve_partitioned(layer, experts, missing_ranks, promotion_ranks);

        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint32_t slot = locations_[layer][experts[rank]];
            if (slot != UINT32_MAX) {
                if (blueprint_) blueprint_->touch_vram(layer, experts[rank], slot);
                result[rank] = slot;
                reserved[slot] = true;
                entries_[slot].age = ++clock_;
            }
        }
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (result[rank] != UINT32_MAX) continue;
            uint32_t slot = UINT32_MAX;
            uint32_t least_used = UINT32_MAX;
            uint64_t oldest = UINT64_MAX;
            for (uint32_t candidate = 0; candidate < persistent_slots_; ++candidate) {
                if (!reserved[candidate]) {
                    const Entry& entry = entries_[candidate];
                    const uint32_t uses = entry.layer < 0 ? 0u :
                        frequency_[static_cast<uint32_t>(entry.layer)]
                                  [static_cast<uint32_t>(entry.expert)];
                    if (uses < least_used ||
                        (uses == least_used && entry.age < oldest)) {
                        least_used = uses;
                        oldest = entry.age;
                        slot = candidate;
                    }
                }
            }
            if (slot == UINT32_MAX)
                throw std::runtime_error("Global DeepSeek cache has no finite victim");
            Entry& victim = entries_[slot];
            if (force_admission_ || victim.layer < 0 ||
                frequency_[layer][experts[rank]] > least_used) {
                if (victim.layer >= 0) {
                    if (blueprint_) blueprint_->evict_vram(
                        static_cast<uint32_t>(victim.layer),
                        static_cast<uint32_t>(victim.expert), slot);
                    locations_[static_cast<uint32_t>(victim.layer)]
                              [static_cast<uint32_t>(victim.expert)] = UINT32_MAX;
                }
                victim.layer = static_cast<int32_t>(layer);
                victim.expert = static_cast<int32_t>(experts[rank]);
                victim.age = ++clock_;
                locations_[layer][experts[rank]] = slot;
                reserved[slot] = true;
                result[rank] = slot;
                if (blueprint_)
                    blueprint_->reserve_vram(layer, experts[rank], slot);
            } else {
                // Six physical transient records are reserved outside the LFU
                // namespace. Cold routes are DMA-staged there for this layer,
                // preserving hot global residents without direct shader PCIe.
                result[rank] = persistent_slots_ + rank;
                if (blueprint_)
                    blueprint_->reserve_vram(layer, experts[rank], result[rank]);
            }
            missing_ranks.push_back(rank);
            promotion_ranks.push_back(rank);
        }
        return result;
    }

    BatchSlots resolve_batch(uint32_t layer, const uint32_t* route_words,
                             uint32_t batch, std::vector<uint32_t>& missing_flat) {
        if (!global_policy_ || !force_admission_)
            throw std::runtime_error(
                "Batched verification requires global forced-admission expert cache");
        BatchSlots result{};
        for (auto& token : result) token.fill(UINT32_MAX);
        std::vector<bool> reserved(persistent_slots_, false);
        std::unordered_map<uint32_t, uint32_t> selected;
        if (batch == 0u || batch > 5u)
            throw std::runtime_error("Invalid batched expert count");
        for (uint32_t token = 0; token < batch; ++token)
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const uint32_t expert = route_words[token * 64u + rank];
                if (expert >= kExperts)
                    throw std::runtime_error("Batched router selected an invalid expert");
                ++frequency_[layer][expert];
                if (blueprint_) blueprint_->request(layer, expert);
                const uint32_t resident = locations_[layer][expert];
                if (resident != UINT32_MAX) {
                    if (blueprint_) blueprint_->touch_vram(layer, expert, resident);
                    reserved[resident] = true;
                    entries_[resident].age = ++clock_;
                    selected.emplace(expert, resident);
                }
            }
        for (uint32_t token = 0; token < batch; ++token)
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const uint32_t expert = route_words[token * 64u + rank];
                const auto existing = selected.find(expert);
                if (existing != selected.end()) {
                    result[token][rank] = existing->second;
                    continue;
                }
                uint32_t slot = UINT32_MAX, least_used = UINT32_MAX;
                uint64_t oldest = UINT64_MAX;
                for (uint32_t candidate = 0; candidate < persistent_slots_; ++candidate) {
                    if (reserved[candidate]) continue;
                    const Entry& entry = entries_[candidate];
                    const uint32_t uses = entry.layer < 0 ? 0u :
                        frequency_[static_cast<uint32_t>(entry.layer)]
                                  [static_cast<uint32_t>(entry.expert)];
                    if (uses < least_used || (uses == least_used && entry.age < oldest)) {
                        slot = candidate; least_used = uses; oldest = entry.age;
                    }
                }
                if (slot == UINT32_MAX)
                    throw std::runtime_error("Batched expert cache has no finite victim");
                Entry& victim = entries_[slot];
                if (victim.layer >= 0) {
                    if (blueprint_) blueprint_->evict_vram(
                        static_cast<uint32_t>(victim.layer),
                        static_cast<uint32_t>(victim.expert), slot);
                    locations_[static_cast<uint32_t>(victim.layer)]
                              [static_cast<uint32_t>(victim.expert)] = UINT32_MAX;
                }
                victim.layer = static_cast<int32_t>(layer);
                victim.expert = static_cast<int32_t>(expert);
                victim.age = ++clock_;
                locations_[layer][expert] = slot;
                reserved[slot] = true;
                selected.emplace(expert, slot);
                result[token][rank] = slot;
                if (blueprint_) blueprint_->reserve_vram(layer, expert, slot);
                missing_flat.push_back(token * kTopK + rank);
            }
        return result;
    }

    DescriptorRange record(uint32_t slot) const {
        if (slot >= total_slots_) throw std::runtime_error("Invalid global expert slot");
        const uint32_t chunk = chunk_for_slot(slot);
        return arena_range(chunks_[chunk],
            static_cast<VkDeviceSize>(slot - bases_[chunk]) * kExpertRecordBytes,
            kExpertRecordBytes);
    }
    const Buffer& arena(uint32_t slot) const {
        if (slot >= total_slots_) throw std::runtime_error("Invalid global expert slot");
        return chunks_[chunk_for_slot(slot)];
    }
    uint32_t capacity() const { return total_slots_; }
    bool set_policy_enabled() const { return set_policy_; }
    SetPolicyMetrics set_policy_metrics() const { return set_policy_metrics_; }
    void reset_set_policy_metrics() { set_policy_metrics_ = {}; }
    bool contains(uint32_t layer, uint32_t expert) const {
        return layer < kLayers && expert < kExperts &&
               locations_[layer][expert] != UINT32_MAX;
    }

private:
    std::array<uint32_t, kTopK> resolve_partitioned(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        std::vector<uint32_t>& missing_ranks,
        std::vector<uint32_t>& promotion_ranks) {
        const uint32_t base = bases_[layer];
        const uint32_t layer_capacity = capacities_[layer];
        std::array<uint32_t, kTopK> result{};
        result.fill(UINT32_MAX);
        std::vector<bool> reserved(layer_capacity, false);
        uint32_t initial_hits = 0;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint32_t slot = locations_[layer][experts[rank]];
            if (slot != UINT32_MAX) {
                if (blueprint_) blueprint_->touch_vram(layer, experts[rank], slot);
                result[rank] = slot;
                reserved[slot - base] = true;
                entries_[slot].age = ++clock_;
                ++initial_hits;
            }
        }
        if (set_policy_)
            record_set_readiness_and_prediction(layer, experts, initial_hits);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (result[rank] != UINT32_MAX) continue;
            uint32_t selected = UINT32_MAX;
            uint32_t least_used = UINT32_MAX;
            uint64_t oldest = UINT64_MAX;
            uint32_t baseline = UINT32_MAX;
            uint32_t baseline_used = UINT32_MAX;
            uint64_t baseline_oldest = UINT64_MAX;
            uint64_t least_set_value = UINT64_MAX;
            uint64_t least_set_age = UINT64_MAX;
            bool found_empty = false;
            for (uint32_t local = 0; local < layer_capacity; ++local) {
                if (reserved[local]) continue;
                const uint32_t slot = base + local;
                const Entry& entry = entries_[slot];
                const uint32_t used = entry.expert < 0 ? 0u :
                    frequency_[layer][static_cast<uint32_t>(entry.expert)];
                if (used < baseline_used ||
                    (used == baseline_used && entry.age < baseline_oldest)) {
                    baseline = slot;
                    baseline_used = used;
                    baseline_oldest = entry.age;
                }
                if (!set_policy_) {
                    if (used < least_used ||
                        (used == least_used && entry.age < oldest)) {
                        selected = slot;
                        least_used = used;
                        oldest = entry.age;
                    }
                    continue;
                }
                // Empty records always win over eviction.  Once full, rank
                // residents by their utility to complete a future Top-6 set,
                // rather than by individual LFU alone.
                if (entry.expert < 0) {
                    if (!found_empty || entry.age < least_set_age) {
                        selected = slot;
                        least_set_age = entry.age;
                    }
                    found_empty = true;
                    continue;
                }
                if (found_empty) continue;
                const uint64_t set_value = resident_set_value(
                    layer, static_cast<uint32_t>(entry.expert), experts,
                    base, layer_capacity);
                if (set_value < least_set_value ||
                    (set_value == least_set_value && entry.age < least_set_age)) {
                    selected = slot;
                    least_set_value = set_value;
                    least_set_age = entry.age;
                }
            }
            if (selected == UINT32_MAX)
                throw std::runtime_error("Partitioned DeepSeek cache has no finite victim");
            if (set_policy_ && entries_[selected].expert >= 0) {
                ++set_policy_metrics_.victim_decisions;
                if (baseline != UINT32_MAX && baseline != selected) {
                    ++set_policy_metrics_.changed_victims;
                    const SetAffinity baseline_affinity = set_affinity(
                        layer, static_cast<uint32_t>(entries_[baseline].expert),
                        experts, base, layer_capacity);
                    const SetAffinity selected_affinity = set_affinity(
                        layer, static_cast<uint32_t>(entries_[selected].expert),
                        experts, base, layer_capacity);
                    if (baseline_affinity.coactivation >
                        selected_affinity.coactivation)
                        ++set_policy_metrics_.coactivation_saved;
                    if (baseline_affinity.adjacent > selected_affinity.adjacent)
                        ++set_policy_metrics_.adjacent_affinity_saved;
                }
            }
            Entry& victim = entries_[selected];
            if (victim.expert >= 0) {
                if (blueprint_) blueprint_->evict_vram(
                    layer, static_cast<uint32_t>(victim.expert), selected);
                locations_[layer][static_cast<uint32_t>(victim.expert)] = UINT32_MAX;
            }
            victim.layer = static_cast<int32_t>(layer);
            victim.expert = static_cast<int32_t>(experts[rank]);
            victim.age = ++clock_;
            locations_[layer][experts[rank]] = selected;
            reserved[selected - base] = true;
            result[rank] = selected;
            if (blueprint_)
                blueprint_->reserve_vram(layer, experts[rank], selected);
            missing_ranks.push_back(rank);
            promotion_ranks.push_back(rank);
        }
        if (set_policy_) learn_set_route(layer, experts);
        return result;
    }

    struct SetAffinity {
        uint64_t coactivation = 0;
        uint64_t adjacent = 0;
    };
    static size_t affinity_index(uint32_t layer, uint32_t first,
                                 uint32_t second) {
        return (static_cast<size_t>(layer) * kExperts + first) * kExperts + second;
    }
    static void saturating_increment(uint16_t& value) {
        if (value != UINT16_MAX) ++value;
    }
    SetAffinity set_affinity(uint32_t layer, uint32_t candidate,
                             const std::array<uint32_t, kTopK>& current,
                             uint32_t base, uint32_t capacity) const {
        SetAffinity value{};
        for (uint32_t expert : current)
            if (expert != candidate)
                value.coactivation += same_layer_affinity_[
                    affinity_index(layer, candidate, expert)];
        // Preserve historically useful complete sets already resident in this
        // layer, including alternatives to the presently selected route.
        for (uint32_t local = 0; local < capacity; ++local) {
            const Entry& peer = entries_[base + local];
            if (peer.expert >= 0 &&
                static_cast<uint32_t>(peer.expert) != candidate)
                value.coactivation += same_layer_affinity_[affinity_index(
                    layer, candidate, static_cast<uint32_t>(peer.expert))] / 2u;
        }
        if (previous_route_layer_ != UINT32_MAX &&
            previous_route_layer_ + 1u == layer) {
            for (uint32_t source : previous_route_)
                value.adjacent += adjacent_layer_affinity_[
                    affinity_index(layer, source, candidate)];
        }
        return value;
    }
    uint64_t resident_set_value(
        uint32_t layer, uint32_t candidate,
        const std::array<uint32_t, kTopK>& current,
        uint32_t base, uint32_t capacity) const {
        const SetAffinity affinity = set_affinity(
            layer, candidate, current, base, capacity);
        const uint64_t uses = frequency_[layer][candidate];
        // LFU remains a weak stability prior; learned set utility is large
        // enough to change placement when it improves whole-route readiness.
        return uses * 32ull + affinity.coactivation * 2ull + affinity.adjacent;
    }
    void record_set_readiness_and_prediction(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        uint32_t initial_hits) {
        ++set_policy_metrics_.route_sets;
        set_policy_metrics_.requested_experts += kTopK;
        set_policy_metrics_.initial_hits += initial_hits;
        const uint32_t misses = kTopK - initial_hits;
        set_policy_metrics_.total_misses += misses;
        set_policy_metrics_.maximum_set_misses = std::max<uint64_t>(
            set_policy_metrics_.maximum_set_misses, misses);
        if (initial_hits == kTopK) ++set_policy_metrics_.complete_set_ready;
        if (initial_hits + 1u == kTopK) ++set_policy_metrics_.one_missing_sets;

        if (previous_route_layer_ == UINT32_MAX ||
            previous_route_layer_ + 1u != layer) return;
        std::array<uint64_t, kExperts> scores{};
        bool learned = false;
        for (uint32_t candidate = 0; candidate < kExperts; ++candidate) {
            for (uint32_t source : previous_route_)
                scores[candidate] += adjacent_layer_affinity_[
                    affinity_index(layer, source, candidate)];
            learned = learned || scores[candidate] != 0;
        }
        if (!learned) return;
        ++set_policy_metrics_.adjacent_prediction_sets;
        std::array<bool, kExperts> selected{};
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            uint32_t best = UINT32_MAX;
            uint64_t best_score = 0;
            for (uint32_t candidate = 0; candidate < kExperts; ++candidate)
                if (!selected[candidate] && scores[candidate] > best_score) {
                    best = candidate;
                    best_score = scores[candidate];
                }
            if (best == UINT32_MAX) break;
            selected[best] = true;
            ++set_policy_metrics_.adjacent_prediction_targets;
            for (uint32_t actual : experts)
                if (actual == best) {
                    ++set_policy_metrics_.adjacent_prediction_hits;
                    break;
                }
        }
    }
    void learn_set_route(uint32_t layer,
                         const std::array<uint32_t, kTopK>& experts) {
        for (uint32_t first = 0; first < kTopK; ++first)
            for (uint32_t second = 0; second < kTopK; ++second)
                if (first != second) {
                    saturating_increment(same_layer_affinity_[affinity_index(
                        layer, experts[first], experts[second])]);
                    ++set_policy_metrics_.coactivation_updates;
                }
        if (previous_route_layer_ != UINT32_MAX &&
            previous_route_layer_ + 1u == layer) {
            for (uint32_t source : previous_route_)
                for (uint32_t target : experts) {
                    saturating_increment(adjacent_layer_affinity_[affinity_index(
                        layer, source, target)]);
                    ++set_policy_metrics_.adjacent_updates;
                }
        }
        previous_route_ = experts;
        previous_route_layer_ = layer;
    }

    struct Entry {
        int32_t layer = -1;
        int32_t expert = -1;
        uint64_t age = 0;
    };
    uint32_t chunk_for_slot(uint32_t slot) const {
        for (uint32_t chunk = 0; chunk < kLayers; ++chunk)
            if (slot >= bases_[chunk] && slot < bases_[chunk] + capacities_[chunk])
                return chunk;
        throw std::runtime_error("Global expert slot has no physical chunk");
    }
    const Runtime& runtime_;
    uint32_t total_slots_ = 0;
    uint32_t persistent_slots_ = 0;
    std::array<uint32_t, kLayers> capacities_{};
    std::array<uint32_t, kLayers> bases_{};
    std::vector<Buffer> chunks_;
    std::vector<Entry> entries_;
    std::array<std::array<uint32_t, kExperts>, kLayers> frequency_{};
    std::array<std::array<uint32_t, kExperts>, kLayers> locations_{};
    uint64_t clock_ = 0;
    bool force_admission_ = false;
    bool global_policy_ = false;
    bool profiled_partition_ = false;
    bool set_policy_ = false;
    std::vector<uint16_t> same_layer_affinity_;
    std::vector<uint16_t> adjacent_layer_affinity_;
    std::array<uint32_t, kTopK> previous_route_{};
    uint32_t previous_route_layer_ = UINT32_MAX;
    SetPolicyMetrics set_policy_metrics_{};
    ExpertBlueprint* blueprint_ = nullptr;
};

class LayerHostExpertCache {
public:
    explicit LayerHostExpertCache(const Runtime& runtime, bool external,
                                  uint32_t layer,
                                  ExpertBlueprint* blueprint = nullptr)
        : runtime_(runtime), external_(external), layer_(layer),
          blueprint_(blueprint) {}

    uint32_t resolve(uint32_t expert, const uint8_t* source, bool& hit) {
        const uint32_t limit = external_ ? kTopK : kHostExpertSlotsPerLayer;
        for (uint32_t slot = 0; slot < limit; ++slot) {
            if (slots_[slot].expert == static_cast<int32_t>(expert)) {
                if (blueprint_) blueprint_->touch_ram(layer_, expert,
                    ExpertBlueprint::local_ram_location(layer_, slot));
                slots_[slot].age = ++clock_;
                hit = true;
                return slot;
            }
        }
        uint32_t selected = 0;
        for (uint32_t slot = 1; slot < limit; ++slot)
            if (slots_[slot].expert < 0 ||
                (slots_[selected].expert >= 0 && slots_[slot].age < slots_[selected].age))
                selected = slot;
        if (slots_[selected].expert >= 0 && blueprint_)
            blueprint_->evict_ram(layer_,
                static_cast<uint32_t>(slots_[selected].expert),
                ExpertBlueprint::local_ram_location(layer_, selected));
        if (external_) {
            if (imports_[selected].buffer.handle)
                destroy_buffer(runtime_, imports_[selected].buffer);
            imports_[selected] = import_dsv4_host_range(
                runtime_, source, kExpertRecordBytes);
        } else if (!buffers_[selected].handle) {
            buffers_[selected] = create_host_buffer_uninitialized(
                runtime_, kExpertRecordBytes);
        }
        slots_[selected].expert = static_cast<int32_t>(expert);
        slots_[selected].age = ++clock_;
        if (blueprint_) blueprint_->reserve_ram(layer_, expert,
            ExpertBlueprint::local_ram_location(layer_, selected), external_);
        hit = false;
        return selected;
    }

    DescriptorRange record(uint32_t slot) const {
        const uint32_t limit = external_ ? kTopK : kHostExpertSlotsPerLayer;
        if (slot >= limit)
            throw std::runtime_error("Invalid DeepSeek host expert slot");
        if (external_) {
            if (!imports_[slot].buffer.handle)
                throw std::runtime_error("DeepSeek imported expert slot is empty");
            return arena_range(imports_[slot].buffer,
                               imports_[slot].data_offset, kExpertRecordBytes);
        }
        if (!buffers_[slot].handle)
            throw std::runtime_error("DeepSeek host expert slot is empty");
        return arena_range(buffers_[slot], 0, kExpertRecordBytes);
    }
    void* mapped(uint32_t slot) const {
        if (external_ || slot >= kHostExpertSlotsPerLayer ||
            !buffers_[slot].mapped)
            throw std::runtime_error("Invalid mapped DeepSeek host expert slot");
        return buffers_[slot].mapped;
    }

    void flush(uint32_t slot) const {
        if (external_ || slot >= kHostExpertSlotsPerLayer ||
            !buffers_[slot].handle)
            throw std::runtime_error("Invalid flushed DeepSeek host expert slot");
        flush_buffer_range(runtime_, buffers_[slot], 0, kExpertRecordBytes);
    }

    void destroy_resources() {
        for (Dsv4ImportedRange& imported : imports_)
            if (imported.buffer.handle) destroy_buffer(runtime_, imported.buffer);
        for (Buffer& buffer : buffers_)
            if (buffer.handle) destroy_buffer(runtime_, buffer);
    }

    bool external() const { return external_; }

private:
    const Runtime& runtime_;
    bool external_ = false;
    uint32_t layer_ = 0;
    ExpertBlueprint* blueprint_ = nullptr;
    std::array<CacheSlot, kHostExpertSlotsPerLayer> slots_{};
    std::array<Buffer, kHostExpertSlotsPerLayer> buffers_{};
    std::array<Dsv4ImportedRange, kHostExpertSlotsPerLayer> imports_{};
    uint64_t clock_ = 0;
};

// The fixed per-layer RAM partition wastes slots in low-diversity layers while
// high-diversity learned routers churn. This pool preserves the exact same
// 1376-record / 17.1-GiB upper bound but lets every layer share it.
class GlobalHostExpertCache {
public:
    static constexpr uint32_t kLegacySlots = kLayers * kHostExpertSlotsPerLayer;
    struct HintMetrics {
        uint64_t installed = 0;
        uint64_t matched_requests = 0;
        uint64_t protected_victims = 0;
    };

    GlobalHostExpertCache(const Runtime& runtime, uint32_t capacity,
                          bool blocked_backing,
                          ExpertBlueprint* blueprint = nullptr)
        : runtime_(runtime), capacity_(capacity), blocked_backing_(blocked_backing),
          slots_(capacity), blueprint_(blueprint) {
        if (capacity_ < kTopK || capacity_ > kLayers * kExperts)
            throw std::runtime_error("Invalid global host expert-cache capacity");
        for (auto& locations : locations_) locations.fill(UINT32_MAX);
        if (blocked_backing_) {
            const uint32_t blocks = (capacity_ + kHostCacheRecordsPerBlock - 1u) /
                                    kHostCacheRecordsPerBlock;
            blocks_.resize(blocks);
        } else {
            buffers_.resize(capacity_);
        }
        tiny_lfu_admission_ = std::getenv("DSV4_HOST_TINYLFU") != nullptr;
        lru_policy_ = std::getenv("DSV4_HOST_LRU") != nullptr;
        plain_backing_ = std::getenv("DSV4_PLAIN_HOST_CACHE") != nullptr;
        hybrid_backing_ = std::getenv("DSV4_HYBRID_HOST_CACHE") != nullptr;
        import_plain_l2_ = std::getenv("DSV4_IMPORT_PLAIN_L2") != nullptr;
        foreign_l2_chunk_ = std::getenv("DSV4_FOREIGN_L2_CHUNK") != nullptr;
        foreign_l2_block_limit_ = foreign_l2_chunk_ ? 1u : 0u;
        if (const char* configured = std::getenv("DSV4_FOREIGN_L2_BLOCKS")) {
            foreign_l2_chunk_ = true;
            if (std::strcmp(configured, "all") == 0) {
                foreign_l2_block_limit_ = UINT32_MAX;
            } else {
                const unsigned long long requested = std::stoull(configured);
                if (requested == 0 || requested > UINT32_MAX)
                    throw std::runtime_error(
                        "DSV4_FOREIGN_L2_BLOCKS must be a positive count or all");
                foreign_l2_block_limit_ = static_cast<uint32_t>(requested);
            }
        }
        if (plain_backing_ && hybrid_backing_)
            throw std::runtime_error(
                "DeepSeek plain and hybrid host-cache modes are mutually exclusive");
        if (plain_backing_ && !blocked_backing_)
            throw std::runtime_error(
                "DSV4_PLAIN_HOST_CACHE requires a DSV4_RAM_GIB budget");
        if (hybrid_backing_ &&
            (!blocked_backing_ || capacity_ <= kHybridHostL1Slots))
            throw std::runtime_error(
                "DSV4_HYBRID_HOST_CACHE requires a budget above the 24-GiB tier");
        if (import_plain_l2_ && !hybrid_backing_)
            throw std::runtime_error(
                "DSV4_IMPORT_PLAIN_L2 requires DSV4_HYBRID_HOST_CACHE");
        if (foreign_l2_chunk_ && !hybrid_backing_)
            throw std::runtime_error(
                "DSV4_FOREIGN_L2_CHUNK requires DSV4_HYBRID_HOST_CACHE");
        if (foreign_l2_chunk_ && import_plain_l2_)
            throw std::runtime_error(
                "DSV4_FOREIGN_L2_CHUNK and DSV4_IMPORT_PLAIN_L2 are mutually exclusive");
        l1_capacity_ = hybrid_backing_ ? kHybridHostL1Slots :
            (plain_backing_ ? 0u : capacity_);
        if (plain_backing_ || hybrid_backing_)
            plain_blocks_.resize(blocks_.size());
        if (import_plain_l2_ || foreign_l2_chunk_)
            plain_imports_.resize(blocks_.size());
        if (foreign_l2_chunk_) foreign_mappings_.resize(blocks_.size());
    }
    void begin_route_resolve() {
        if (route_resolve_active_)
            throw std::runtime_error("Nested global host-cache route resolve");
        route_resolve_active_ = true;
        route_reserved_count_ = 0;
    }
    void end_route_resolve() {
        route_resolve_active_ = false;
        route_reserved_count_ = 0;
    }
    ~GlobalHostExpertCache() {
        for (Buffer& buffer : buffers_)
            if (buffer.handle) destroy_buffer(runtime_, buffer);
        for (Buffer& block : blocks_)
            if (block.handle) destroy_buffer(runtime_, block);
        for (Dsv4ImportedRange& imported : plain_imports_)
            if (imported.buffer.handle) destroy_buffer(runtime_, imported.buffer);
        for (size_t index = 0; index < plain_blocks_.size(); ++index) {
            if (!plain_blocks_[index]) continue;
            if (index < foreign_mappings_.size() && foreign_mappings_[index]) {
                UnmapViewOfFile(plain_blocks_[index]);
                CloseHandle(foreign_mappings_[index]);
            } else {
                VirtualFree(plain_blocks_[index], 0, MEM_RELEASE);
            }
        }
    }

    uint32_t resolve(uint32_t layer, uint32_t expert, uint32_t transient,
                     bool& hit) {
        if (transient >= kTopK)
            throw std::runtime_error("Invalid transient host-cache slot");
        uint32_t& requested_frequency = frequency_[layer][expert];
        ++requested_frequency;
        if (retention_hint_valid_ && retention_hint_layer_ == layer &&
            retention_hint_expert_ == expert) {
            // The authoritative route has now confirmed the speculation.  A
            // single extra admission count is bounded and never admits a
            // wrong expert or causes additional I/O.
            if (requested_frequency != UINT32_MAX) ++requested_frequency;
            ++hint_metrics_.matched_requests;
        }
        if ((++resolves_ & 4095u) == 0u) {
            for (auto& layer_frequency : frequency_)
                for (uint32_t& count : layer_frequency) count = (count + 1u) >> 1u;
        }
        const uint32_t location = locations_[layer][expert];
        if (location != UINT32_MAX) {
            if (blueprint_) blueprint_->touch_ram(layer, expert, location);
            slots_[location].age = ++clock_;
            reserve_route_slot(location);
            hit = true;
            return location;
        }
        uint32_t selected = UINT32_MAX;
        uint32_t least_used = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        auto first_empty = [&](uint32_t begin, uint32_t end) {
            for (uint32_t slot = begin; slot < end; ++slot)
                if (!route_slot_reserved(slot) && slots_[slot].layer < 0)
                    return slot;
            return UINT32_MAX;
        };
        auto select_lfu = [&](uint32_t begin, uint32_t end) {
            const uint32_t protected_slot = retention_hint_valid_ ?
                locations_[retention_hint_layer_][retention_hint_expert_] :
                UINT32_MAX;
            uint32_t baseline = UINT32_MAX;
            uint32_t baseline_used = UINT32_MAX;
            uint64_t baseline_age = UINT64_MAX;
            for (uint32_t slot = begin; slot < end; ++slot) {
                if (route_slot_reserved(slot)) continue;
                uint32_t used = 0;
                if (slots_[slot].layer >= 0) {
                    used = frequency_[static_cast<uint32_t>(slots_[slot].layer)]
                                     [static_cast<uint32_t>(slots_[slot].expert)];
                }
                if ((lru_policy_ && slots_[slot].age < baseline_age) ||
                    (!lru_policy_ && (used < baseline_used ||
                     (used == baseline_used && slots_[slot].age < baseline_age)))) {
                    baseline = slot;
                    baseline_used = used;
                    baseline_age = slots_[slot].age;
                }
            }
            if (baseline == UINT32_MAX)
                throw std::runtime_error(
                    "Global host cache has no unreserved route victim");
            if (baseline != protected_slot) {
                selected = baseline;
                least_used = baseline_used;
                oldest = baseline_age;
                return;
            }
            for (uint32_t slot = begin; slot < end; ++slot) {
                if (slot == protected_slot || route_slot_reserved(slot)) continue;
                uint32_t used = 0;
                if (slots_[slot].layer >= 0)
                    used = frequency_[static_cast<uint32_t>(slots_[slot].layer)]
                                     [static_cast<uint32_t>(slots_[slot].expert)];
                if ((lru_policy_ && slots_[slot].age < oldest) ||
                    (!lru_policy_ && (used < least_used ||
                     (used == least_used && slots_[slot].age < oldest)))) {
                    selected = slot;
                    least_used = used;
                    oldest = slots_[slot].age;
                }
            }
            if (selected == UINT32_MAX) {
                selected = baseline;
                least_used = baseline_used;
                oldest = baseline_age;
            } else {
                ++hint_metrics_.protected_victims;
            }
        };
        if (hybrid_backing_) {
            selected = first_empty(0, l1_capacity_);
            if (selected == UINT32_MAX)
                selected = first_empty(l1_capacity_, capacity_);
            if (selected == UINT32_MAX)
                select_lfu(l1_capacity_, capacity_);
            else least_used = 0;
        } else {
            select_lfu(0, capacity_);
        }
        // A strict cache budget should not let a sequential scan of one-shot
        // routes evict experts that have already proved useful.  Rejected
        // records are still read and executed through one of six bounded
        // staging buffers, but never enter the persistent RAM namespace.
        if (tiny_lfu_admission_ && slots_[selected].layer >= 0 &&
            requested_frequency <= least_used) {
            ++admission_rejections_;
            hit = false;
            return capacity_ + transient;
        }
        ensure_backing(selected);
        if (slots_[selected].layer >= 0) {
            if (blueprint_) blueprint_->evict_ram(
                static_cast<uint32_t>(slots_[selected].layer),
                static_cast<uint32_t>(slots_[selected].expert), selected);
            locations_[static_cast<uint32_t>(slots_[selected].layer)]
                      [static_cast<uint32_t>(slots_[selected].expert)] = UINT32_MAX;
        }
        slots_[selected].layer = static_cast<int32_t>(layer);
        slots_[selected].expert = static_cast<int32_t>(expert);
        slots_[selected].age = ++clock_;
        locations_[layer][expert] = selected;
        reserve_route_slot(selected);
        if (blueprint_) blueprint_->reserve_ram(layer, expert, selected, false);
        hit = false;
        return selected;
    }

    uint32_t promote_from_l2(uint32_t layer, uint32_t expert, uint32_t l2_slot,
                             void* swap_staging, bool staged_valid,
                             const std::vector<uint32_t>& protected_l1) {
        if (!hybrid_backing_ || l2_slot >= capacity_ || !plain_slot(l2_slot) ||
            !swap_staging || locations_[layer][expert] != l2_slot)
            return UINT32_MAX;
        uint32_t victim = UINT32_MAX;
        uint32_t least_used = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t slot = 0; slot < l1_capacity_; ++slot) {
            if (std::find(protected_l1.begin(), protected_l1.end(), slot) !=
                protected_l1.end()) continue;
            const Slot& entry = slots_[slot];
            const uint32_t used = entry.layer < 0 ? 0u :
                frequency_[static_cast<uint32_t>(entry.layer)]
                          [static_cast<uint32_t>(entry.expert)];
            if (used < least_used || (used == least_used && entry.age < oldest)) {
                victim = slot;
                least_used = used;
                oldest = entry.age;
            }
        }
        if (victim == UINT32_MAX || frequency_[layer][expert] <= least_used + 1u)
            return UINT32_MAX;

        ensure_backing(victim);
        const Slot demoted = slots_[victim];
        void* const l1_data = mapped(victim);
        void* const l2_data = mapped(l2_slot);
        uint64_t copied = kExpertRecordBytes;
        if (!staged_valid) {
            std::memcpy(swap_staging, l2_data, kExpertRecordBytes);
            copied += kExpertRecordBytes;
        }
        if (demoted.layer >= 0) {
            std::memcpy(l2_data, l1_data, kExpertRecordBytes);
            copied += kExpertRecordBytes;
            locations_[static_cast<uint32_t>(demoted.layer)]
                      [static_cast<uint32_t>(demoted.expert)] = l2_slot;
            slots_[l2_slot] = demoted;
            slots_[l2_slot].age = ++clock_;
        } else {
            slots_[l2_slot] = Slot{};
        }
        std::memcpy(l1_data, swap_staging, kExpertRecordBytes);
        flush(victim);
        slots_[victim].layer = static_cast<int32_t>(layer);
        slots_[victim].expert = static_cast<int32_t>(expert);
        slots_[victim].age = ++clock_;
        locations_[layer][expert] = victim;
        if (blueprint_) blueprint_->swap_ram(layer, expert, l2_slot,
            demoted.layer, demoted.expert, victim);
        ++hybrid_promotions_;
        hybrid_copy_bytes_ += copied;
        return victim;
    }

    void* mapped(uint32_t slot) {
        if (slot >= capacity_)
            throw std::runtime_error("Invalid global host-cache slot");
        if (blocked_backing_) {
            const uint32_t block_index = slot / kHostCacheRecordsPerBlock;
            void* const base = plain_slot(slot) ? plain_blocks_[block_index] :
                blocks_[block_index].mapped;
            if (!base) throw std::runtime_error("Empty global host-cache block");
            return static_cast<uint8_t*>(base) +
                   (slot % kHostCacheRecordsPerBlock) * kExpertRecordBytes;
        }
        if (!buffers_[slot].mapped)
            throw std::runtime_error("Empty global host-cache record");
        return buffers_[slot].mapped;
    }
    void flush(uint32_t slot) {
        if (slot >= capacity_)
            throw std::runtime_error("Invalid global host-cache flush");
        if (blocked_backing_) {
            if (plain_slot(slot)) return;
            const Buffer& block = blocks_[slot / kHostCacheRecordsPerBlock];
            if (!block.handle) throw std::runtime_error("Empty global host-cache block");
            flush_buffer_range(runtime_, block,
                (slot % kHostCacheRecordsPerBlock) * kExpertRecordBytes,
                kExpertRecordBytes);
        } else {
            if (!buffers_[slot].handle)
                throw std::runtime_error("Empty global host-cache record");
            flush_buffer(runtime_, buffers_[slot]);
        }
    }
    DescriptorRange record(uint32_t slot) const {
        if (slot >= capacity_)
            throw std::runtime_error("Invalid global host-cache record");
        if (plain_slot(slot)) {
            if (imported_plain_slot(slot)) {
                const uint32_t block_index = slot / kHostCacheRecordsPerBlock;
                return arena_range(plain_imports_[block_index].buffer,
                    plain_imports_[block_index].data_offset +
                        (slot % kHostCacheRecordsPerBlock) * kExpertRecordBytes,
                    kExpertRecordBytes);
            }
            throw std::runtime_error(
                "Plain host-cache memory must be copied through Vulkan staging");
        }
        if (blocked_backing_) {
            const Buffer& block = blocks_[slot / kHostCacheRecordsPerBlock];
            if (!block.handle) throw std::runtime_error("Empty global host-cache block");
            return arena_range(block,
                (slot % kHostCacheRecordsPerBlock) * kExpertRecordBytes,
                kExpertRecordBytes);
        }
        if (!buffers_[slot].handle)
            throw std::runtime_error("Empty global host-cache record");
        return arena_range(buffers_[slot], 0, kExpertRecordBytes);
    }
    uint32_t capacity() const { return capacity_; }
    bool set_retention_hint(uint32_t layer, uint32_t expert) {
        if (layer >= kLayers || expert >= kExperts)
            throw std::runtime_error("Invalid global host retention hint");
        retention_hint_layer_ = layer;
        retention_hint_expert_ = expert;
        retention_hint_valid_ = true;
        ++hint_metrics_.installed;
        return locations_[layer][expert] != UINT32_MAX;
    }
    void clear_retention_hint() { retention_hint_valid_ = false; }
    HintMetrics hint_metrics() const { return hint_metrics_; }
    void reset_hint_metrics() { hint_metrics_ = {}; }
    bool uses_plain_backing() const { return plain_backing_ || hybrid_backing_; }
    bool all_plain_backing() const { return plain_backing_; }
    bool plain_slot(uint32_t slot) const {
        if (slot >= capacity_)
            throw std::runtime_error("Invalid plain host-cache query");
        return slot >= l1_capacity_;
    }
    bool imported_plain_slot(uint32_t slot) const {
        if ((!import_plain_l2_ && !foreign_l2_chunk_) || slot >= capacity_ ||
            !plain_slot(slot)) return false;
        return plain_imports_[slot / kHostCacheRecordsPerBlock].buffer.handle !=
               VK_NULL_HANDLE;
    }
    uint64_t admission_rejections() const { return admission_rejections_; }
    uint64_t hybrid_promotions() const { return hybrid_promotions_; }
    uint64_t hybrid_copy_bytes() const { return hybrid_copy_bytes_; }
    uint64_t imported_plain_bytes() const { return imported_plain_bytes_; }
    uint32_t imported_plain_blocks() const { return imported_plain_blocks_; }
    bool transient(uint32_t slot) const {
        return slot >= capacity_ && slot < capacity_ + kTopK;
    }
    uint32_t transient_index(uint32_t slot) const {
        if (!transient(slot))
            throw std::runtime_error("Invalid transient host-cache record");
        return slot - capacity_;
    }

    uint32_t fill_remaining_uniform(const ExpertIndex& experts,
                                    double& seconds) {
        if (hybrid_backing_)
            throw std::runtime_error(
                "DeepSeek RAM-cache top-off does not support hybrid backing");
        const auto started = std::chrono::steady_clock::now();
        HANDLE file = CreateFileA(experts.path().c_str(), GENERIC_READ,
            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            throw std::runtime_error(
                "Could not open DeepSeek top-off expert stream");
        uint32_t added = 0;
        uint32_t candidate = 0;
        constexpr uint32_t kPermutationMultiplier = 73;
        constexpr uint32_t kLayerOffset = 29;
        constexpr uint32_t kTotalKeys = kLayers * kExperts;
        try {
            for (uint32_t slot = 0; slot < capacity_; ++slot) {
                if (slots_[slot].layer >= 0) continue;
                uint32_t layer = UINT32_MAX, expert = UINT32_MAX;
                while (candidate < kTotalKeys) {
                    const uint32_t proposed_layer = candidate % kLayers;
                    const uint32_t round = candidate / kLayers;
                    ++candidate;
                    const uint32_t proposed_expert =
                        (round * kPermutationMultiplier +
                         proposed_layer * kLayerOffset) % kExperts;
                    if (locations_[proposed_layer][proposed_expert] ==
                        UINT32_MAX) {
                        layer = proposed_layer;
                        expert = proposed_expert;
                        break;
                    }
                }
                if (layer == UINT32_MAX)
                    throw std::runtime_error(
                        "DeepSeek RAM-cache top-off exhausted expert keys");
                ensure_backing(slot);
                void* const destination = mapped(slot);
                const uint64_t file_offset =
                    experts.core_record_offset(layer, expert);
                if ((reinterpret_cast<uintptr_t>(destination) & 4095u) != 0u ||
                    (file_offset & 4095u) != 0u ||
                    (kExpertRecordBytes & 4095u) != 0u)
                    throw std::runtime_error(
                        "Unaligned DeepSeek RAM-cache top-off record");
                LARGE_INTEGER offset{};
                offset.QuadPart = static_cast<LONGLONG>(file_offset);
                DWORD transferred = 0;
                if (!SetFilePointerEx(file, offset, nullptr, FILE_BEGIN) ||
                    !ReadFile(file, destination,
                              static_cast<DWORD>(kExpertRecordBytes),
                              &transferred, nullptr) ||
                    transferred != kExpertRecordBytes)
                    throw std::runtime_error(
                        "DeepSeek RAM-cache top-off read failed (Win32 " +
                        std::to_string(GetLastError()) + ")");
                flush(slot);
                slots_[slot].layer = static_cast<int32_t>(layer);
                slots_[slot].expert = static_cast<int32_t>(expert);
                slots_[slot].age = ++clock_;
                locations_[layer][expert] = slot;
                if (blueprint_)
                    blueprint_->reserve_ram(layer, expert, slot, true);
                ++added;
            }
        } catch (...) {
            CloseHandle(file);
            throw;
        }
        CloseHandle(file);
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return added;
    }

private:
    void ensure_backing(uint32_t slot) {
        if (blocked_backing_) {
            const uint32_t block_index = slot / kHostCacheRecordsPerBlock;
            const uint32_t first = block_index * kHostCacheRecordsPerBlock;
            const uint32_t records = std::min<uint32_t>(
                kHostCacheRecordsPerBlock, capacity_ - first);
            const uint64_t bytes = static_cast<uint64_t>(records) * kExpertRecordBytes;
            if (plain_slot(slot)) {
                if (!plain_blocks_[block_index]) {
                    const uint32_t first_l2_block =
                        l1_capacity_ / kHostCacheRecordsPerBlock;
                    const bool foreign_block = foreign_l2_chunk_ &&
                        block_index >= first_l2_block &&
                        block_index - first_l2_block < foreign_l2_block_limit_;
                    if (foreign_block) {
                        const DWORD high = static_cast<DWORD>(bytes >> 32u);
                        const DWORD low = static_cast<DWORD>(bytes);
                        HANDLE mapping = CreateFileMappingW(INVALID_HANDLE_VALUE,
                            nullptr, PAGE_READWRITE, high, low, nullptr);
                        if (!mapping)
                            throw std::runtime_error(
                                "DeepSeek pagefile L2 mapping failed (" +
                                std::to_string(bytes) + " bytes, Win32 " +
                                std::to_string(GetLastError()) + ")");
                        void* const block = MapViewOfFile(mapping,
                            FILE_MAP_READ | FILE_MAP_WRITE, 0, 0,
                            static_cast<SIZE_T>(bytes));
                        if (!block) {
                            const DWORD error = GetLastError();
                            CloseHandle(mapping);
                            throw std::runtime_error(
                                "DeepSeek pagefile L2 view failed (Win32 " +
                                std::to_string(error) + ")");
                        }
                        plain_blocks_[block_index] = block;
                        foreign_mappings_[block_index] = mapping;
                        dsv4_plain_host_allocated_bytes += bytes;
                        try {
                            plain_imports_[block_index] = import_dsv4_host_range(
                                runtime_, static_cast<const uint8_t*>(block), bytes,
                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
                                true);
                        } catch (const std::exception& error) {
                            dsv4_plain_host_allocated_bytes -= bytes;
                            plain_blocks_[block_index] = nullptr;
                            foreign_mappings_[block_index] = nullptr;
                            UnmapViewOfFile(block);
                            CloseHandle(mapping);
                            throw std::runtime_error(
                                "DeepSeek FOREIGN L2 import failed at backing block " +
                                std::to_string(block_index) + " after " +
                                std::to_string(imported_plain_blocks_) +
                                " successful imports: " + error.what());
                        }
                        imported_plain_bytes_ += bytes;
                        ++imported_plain_blocks_;
                    } else {
                        void* const block = VirtualAlloc(nullptr,
                            static_cast<SIZE_T>(bytes), MEM_RESERVE | MEM_COMMIT,
                            PAGE_READWRITE);
                        if (!block)
                            throw std::runtime_error(
                                "Plain DeepSeek host-cache allocation failed (" +
                                std::to_string(bytes) + " bytes, Win32 " +
                                std::to_string(GetLastError()) + ")");
                        plain_blocks_[block_index] = block;
                        dsv4_plain_host_allocated_bytes += bytes;
                    }
                }
                if (import_plain_l2_ &&
                    !plain_imports_[block_index].buffer.handle) {
                    plain_imports_[block_index] = import_dsv4_host_range(
                        runtime_, static_cast<const uint8_t*>(
                            plain_blocks_[block_index]), bytes,
                        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT);
                    imported_plain_bytes_ += bytes;
                    ++imported_plain_blocks_;
                }
            } else {
                Buffer& block = blocks_[block_index];
                if (!block.handle) {
                    block = create_host_buffer_uninitialized(runtime_, bytes);
                }
            }
        } else if (!buffers_[slot].handle) {
            buffers_[slot] = create_host_buffer_uninitialized(
                runtime_, kExpertRecordBytes);
        }
    }

    bool route_slot_reserved(uint32_t slot) const {
        if (!route_resolve_active_) return false;
        for (uint32_t index = 0; index < route_reserved_count_; ++index)
            if (route_reserved_slots_[index] == slot) return true;
        return false;
    }
    void reserve_route_slot(uint32_t slot) {
        if (!route_resolve_active_ || slot >= capacity_ ||
            route_slot_reserved(slot)) return;
        if (route_reserved_count_ >= kTopK)
            throw std::runtime_error("Global host-cache route reservation overflow");
        route_reserved_slots_[route_reserved_count_++] = slot;
    }

    struct Slot {
        int32_t layer = -1;
        int32_t expert = -1;
        uint64_t age = 0;
    };
    const Runtime& runtime_;
    uint32_t capacity_ = 0;
    bool blocked_backing_ = false;
    std::vector<Slot> slots_;
    std::vector<Buffer> buffers_;
    std::vector<Buffer> blocks_;
    std::vector<void*> plain_blocks_;
    std::vector<Dsv4ImportedRange> plain_imports_;
    std::vector<HANDLE> foreign_mappings_;
    std::array<std::array<uint32_t, kExperts>, kLayers> frequency_{};
    std::array<std::array<uint32_t, kExperts>, kLayers> locations_{};
    uint64_t clock_ = 0;
    uint64_t resolves_ = 0;
    uint64_t admission_rejections_ = 0;
    uint64_t hybrid_promotions_ = 0;
    uint64_t hybrid_copy_bytes_ = 0;
    uint64_t imported_plain_bytes_ = 0;
    uint32_t imported_plain_blocks_ = 0;
    bool tiny_lfu_admission_ = false;
    bool lru_policy_ = false;
    bool plain_backing_ = false;
    bool hybrid_backing_ = false;
    bool import_plain_l2_ = false;
    bool foreign_l2_chunk_ = false;
    uint32_t foreign_l2_block_limit_ = 0;
    uint32_t l1_capacity_ = 0;
    ExpertBlueprint* blueprint_ = nullptr;
    bool retention_hint_valid_ = false;
    uint32_t retention_hint_layer_ = UINT32_MAX;
    uint32_t retention_hint_expert_ = UINT32_MAX;
    HintMetrics hint_metrics_{};
    bool route_resolve_active_ = false;
    uint32_t route_reserved_count_ = 0;
    std::array<uint32_t, kTopK> route_reserved_slots_{};
};

static void transfer_barrier(VkCommandBuffer command, const Buffer& destination) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = destination.handle;
    barrier.offset = 0;
    barrier.size = destination.size;
    vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

class WeightStreamer {
public:
    struct ExpertReadiness {
        bool all_vram_or_ram = true;
        bool ssd_miss = false;
        uint32_t read_count = 0;
    };
    struct BundleMetrics {
        uint64_t read_count = 0;
        uint64_t read_bytes = 0;
        uint64_t stalled_layers = 0;
    };
    // One finite layer acquisition.  Disk workers may fill records in
    // parallel, but only the decode thread mutates cache metadata or submits
    // Vulkan work.  This preserves Vulkan queue external synchronization even
    // on drivers which expose only one queue from the selected family.
    struct ProgressiveExpertBatch {
        uint32_t layer = UINT32_MAX;
        std::array<uint32_t, kTopK> experts{};
        std::array<uint32_t, kTopK> slots{};
        std::array<uint32_t, kTopK> host_slots{};
        std::array<uint32_t, kTopK> fill_job{};
        std::array<bool, kTopK> missing{};
        std::array<bool, kTopK> host_hit{};
        std::array<bool, kTopK> disk_pending{};
        std::array<bool, kTopK> submitted{};
        uint32_t fill_count = 0;
        uint32_t disk_remaining = 0;
        bool active = false;
        ExpertReadiness readiness{};
        double disk_wait_seconds = 0.0;
    };
    WeightStreamer(const Runtime& runtime, const ReadOnlyMapping& shared_file,
                   const SharedIndex& shared_index, const ExpertIndex& experts,
                   FiniteQueue& transfer, ExpertBlueprint* blueprint = nullptr)
        : runtime_(runtime), shared_file_(shared_file), shared_index_(shared_index),
          experts_(experts), transfer_(transfer), blueprint_(blueprint) {
        uint32_t host_cache_slots = GlobalHostExpertCache::kLegacySlots;
        if (const char* configured = std::getenv("DSV4_RAM_GIB")) {
            const uint64_t gib = std::stoull(configured);
            if (gib < 2u || gib > 60u)
                throw std::runtime_error("DSV4_RAM_GIB must be in [2,60]");
            host_budget_bytes_ = gib << 30;
            budgeted_io_ = true;
            active_fill_workers_ = kMaximumExpertFillWorkers;
            const uint64_t available = host_budget_bytes_ - kHostBudgetReserveBytes;
            const uint64_t block_bytes = static_cast<uint64_t>(
                kHostCacheRecordsPerBlock) * kExpertRecordBytes;
            const uint64_t blocks = available / block_bytes;
            if (blocks == 0)
                throw std::runtime_error("DSV4_RAM_GIB leaves no expert-cache block");
            host_cache_slots = static_cast<uint32_t>(std::min<uint64_t>(
                blocks * kHostCacheRecordsPerBlock,
                static_cast<uint64_t>(kLayers) * kExperts));
            shared_direct_file_ = CreateFileA(shared_file_.path().c_str(),
                GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
            if (shared_direct_file_ == INVALID_HANDLE_VALUE)
                throw std::runtime_error("Could not open budgeted DeepSeek shared stream");
        }
        const uint64_t maximum = shared_index.maximum_layer_bytes();
        if (maximum == 0 || maximum > 512ull * 1024 * 1024)
            throw std::runtime_error("Implausible DeepSeek shared-layer range");
        constexpr uint64_t chunk_capacity = 64ull * 1024 * 1024;
        // load_shared_layer performs one exact layer copy in fallback mode, so
        // this staging allocation must cover the largest layer. Global uploads
        // use their separate explicitly chunked staging buffer below.
        shared_staging_ = create_host_buffer_uninitialized(runtime, maximum);
        if (const char* external = std::getenv("DSV4_EXTERNAL_HOST")) {
            external_host_mode_ = true;
            if (std::strcmp(external, "all") == 0) external_host_layer_ = -1;
            else {
                external_host_layer_ = std::stoi(external);
                if (external_host_layer_ < 0 ||
                    external_host_layer_ >= static_cast<int32_t>(kLayers))
                    throw std::runtime_error("Invalid DSV4_EXTERNAL_HOST layer");
            }
        }
        if (!external_host_mode_ && std::getenv("DSV4_LOCAL_HOST_CACHE") == nullptr)
            global_host_cache_ = std::make_unique<GlobalHostExpertCache>(
                runtime_, host_cache_slots, budgeted_io_, blueprint_);
        host_expert_caches_.reserve(kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            const bool external = external_host_mode_ &&
                (external_host_layer_ < 0 ||
                 external_host_layer_ == static_cast<int32_t>(layer));
            host_expert_caches_.emplace_back(
                runtime, external, layer, blueprint_);
        }
        const auto global = shared_index.global_range();
        global_file_base_ = global.first;
        const uint64_t global_bytes = global.second - global.first;
        global_device_ = create_device_buffer(runtime, global_bytes);
        global_staging_ = create_host_buffer_uninitialized(
            runtime, std::min(global_bytes, chunk_capacity));
        upload_range(global_device_, global_staging_, global.first, global_bytes);

        // Shared/non-expert weights are only roughly seven GiB in row-Q8.  Keep
        // one exact-sized allocation per layer resident so decode never moves
        // those bytes over PCIe.  If the driver cannot fit that layout, retain a
        // safe two-buffer fallback; expert streaming remains identical.
        try {
            resident_layers_.resize(kLayers);
            for (uint32_t layer = 0; layer < kLayers; ++layer) {
                const auto range = shared_index_.layer_range(layer);
                resident_layers_[layer] = create_device_buffer(
                    runtime_, range.second - range.first);
            }
            for (uint32_t layer = 0; layer < kLayers; ++layer) {
                const auto range = shared_index_.layer_range(layer);
                upload_range(resident_layers_[layer], shared_staging_, range.first,
                             range.second - range.first);
            }
            resident_shared_ = true;
        } catch (const std::exception& error) {
            for (Buffer& buffer : resident_layers_) destroy_buffer(runtime_, buffer);
            resident_layers_.clear();
            std::cerr << "DeepSeek shared residency unavailable (" << error.what()
                      << "); using bounded layer double buffering\n";
            for (Buffer& buffer : shared_device_)
                buffer = create_device_buffer(runtime_, maximum);
        }
        unbuffered_files_.fill(INVALID_HANDLE_VALUE);
        bundle_files_.fill(INVALID_HANDLE_VALUE);
        read_events_.fill(nullptr);
        unbuffered_reads_ = budgeted_io_ ||
                            std::getenv("DSV4_UNBUFFERED_READ") != nullptr;
        overlapped_reads_ = budgeted_io_;
        split_expert_transfer_ =
            std::getenv("DSV4_SPLIT_EXPERT_TRANSFER") != nullptr;
        const char* bundle_index_path = std::getenv("DSV4_EXPERT_BUNDLE_INDEX");
        const char* bundle_store_path = std::getenv("DSV4_EXPERT_BUNDLE_STORE");
        if ((bundle_index_path == nullptr) != (bundle_store_path == nullptr))
            throw std::runtime_error(
                "DSV4_EXPERT_BUNDLE_INDEX and DSV4_EXPERT_BUNDLE_STORE must be set together");
        if (bundle_index_path) {
            if (!budgeted_io_ || !unbuffered_reads_ || !overlapped_reads_)
                throw std::runtime_error(
                    "DeepSeek expert bundles require a DSV4_RAM_GIB budget");
            bundle_index_file_ = std::make_unique<ReadOnlyMapping>(bundle_index_path);
            bundle_index_ = std::make_unique<ExpertBundleIndex>(
                *bundle_index_file_, experts_);
            bundle_store_path_ = bundle_store_path;
            HANDLE probe = CreateFileA(bundle_store_path_.c_str(), GENERIC_READ,
                FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, nullptr);
            if (probe == INVALID_HANDLE_VALUE)
                throw std::runtime_error("Could not open DeepSeek expert-bundle store");
            LARGE_INTEGER length{};
            const bool valid_length = GetFileSizeEx(probe, &length) != FALSE &&
                length.QuadPart >= 0 &&
                static_cast<uint64_t>(length.QuadPart) ==
                    bundle_index_->data_file_bytes();
            CloseHandle(probe);
            if (!valid_length)
                throw std::runtime_error("DeepSeek expert-bundle store size drift");
            bundle_staging_bytes_ = static_cast<uint64_t>(kTopK) *
                                    kExpertRecordBytes;
            bundle_staging_ = VirtualAlloc(nullptr,
                static_cast<SIZE_T>(bundle_staging_bytes_),
                MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
            if (!bundle_staging_)
                throw std::runtime_error("Could not allocate bounded bundle-read staging");
            dsv4_plain_host_allocated_bytes += bundle_staging_bytes_;
        }
        if (unbuffered_reads_) {
            for (uint32_t worker = 0; worker < active_fill_workers_; ++worker) {
                DWORD flags = FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS;
                if (overlapped_reads_) flags |= FILE_FLAG_OVERLAPPED;
                unbuffered_files_[worker] = CreateFileA(experts_.path().c_str(),
                    GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                    flags, nullptr);
                if (unbuffered_files_[worker] == INVALID_HANDLE_VALUE)
                    throw std::runtime_error("Could not open unbuffered DeepSeek expert stream");
                if (bundle_index_) {
                    bundle_files_[worker] = CreateFileA(bundle_store_path_.c_str(),
                        GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        flags, nullptr);
                    if (bundle_files_[worker] == INVALID_HANDLE_VALUE)
                        throw std::runtime_error(
                            "Could not open unbuffered DeepSeek expert-bundle stream");
                }
                if (overlapped_reads_) {
                    read_events_[worker] = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                    if (!read_events_[worker])
                        throw std::runtime_error("Could not create DeepSeek read event");
                }
            }
        }
        for (uint32_t worker = 0; worker < active_fill_workers_; ++worker) {
            fill_workers_[worker] = std::thread([this, worker] {
                uint64_t observed_generation = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lock(fill_mutex_);
                    fill_start_.wait(lock, [&] {
                        return fill_stop_ || fill_generation_ != observed_generation;
                    });
                    if (fill_stop_) return;
                    observed_generation = fill_generation_;
                    const uint32_t layer = fill_layer_;
                    lock.unlock();
                    auto read_range = [&](HANDLE file, void* destination,
                                          uint64_t file_offset,
                                          uint64_t bytes) -> DWORD {
                        if (file == INVALID_HANDLE_VALUE ||
                            bytes == 0 ||
                            bytes > static_cast<uint64_t>(
                                std::numeric_limits<DWORD>::max()) ||
                            (reinterpret_cast<uintptr_t>(destination) & 4095u) != 0u ||
                            (file_offset & 4095u) != 0u || (bytes & 4095u) != 0u)
                            return ERROR_INVALID_PARAMETER;
                        DWORD transferred = 0;
                        bool read_ok = false;
                        DWORD error = ERROR_SUCCESS;
                        if (overlapped_reads_) {
                            OVERLAPPED operation{};
                            operation.Offset = static_cast<DWORD>(file_offset);
                            operation.OffsetHigh = static_cast<DWORD>(file_offset >> 32u);
                            operation.hEvent = read_events_[worker];
                            ResetEvent(operation.hEvent);
                            const BOOL started = ReadFile(file, destination,
                                static_cast<DWORD>(bytes), nullptr, &operation);
                            const DWORD start_error = started ?
                                ERROR_SUCCESS : GetLastError();
                            if (started || start_error == ERROR_IO_PENDING) {
                                const DWORD waited = WaitForSingleObject(
                                    operation.hEvent, 10000u);
                                if (waited == WAIT_OBJECT_0) {
                                    read_ok = GetOverlappedResult(
                                        file, &operation, &transferred, FALSE) != FALSE;
                                    if (!read_ok) error = GetLastError();
                                } else {
                                    CancelIoEx(file, &operation);
                                    // The OVERLAPPED object and staging range must remain
                                    // alive until cancellation itself completes.
                                    WaitForSingleObject(operation.hEvent, 10000u);
                                    DWORD ignored = 0;
                                    GetOverlappedResult(file, &operation, &ignored, FALSE);
                                    error = ERROR_TIMEOUT;
                                }
                            } else {
                                error = start_error;
                            }
                        } else {
                            LARGE_INTEGER offset{};
                            offset.QuadPart = static_cast<LONGLONG>(file_offset);
                            const BOOL positioned = SetFilePointerEx(
                                file, offset, nullptr, FILE_BEGIN);
                            const BOOL read = positioned && ReadFile(
                                file, destination, static_cast<DWORD>(bytes),
                                &transferred, nullptr);
                            read_ok = read != FALSE;
                            if (!read_ok) error = GetLastError();
                        }
                        if (!read_ok || transferred != bytes)
                            return error == ERROR_SUCCESS ? ERROR_READ_FAULT : error;
                        return ERROR_SUCCESS;
                    };
                    const uint32_t io_count = bundle_index_ ?
                        fill_bundle_read_count_ : fill_count_;
                    for (uint32_t job = worker; job < io_count;
                          job += active_fill_workers_) {
                        DWORD error = ERROR_SUCCESS;
                        if (bundle_index_) {
                            void* const destination = static_cast<uint8_t*>(
                                bundle_staging_) +
                                static_cast<uint64_t>(fill_bundle_staging_[job]) *
                                    kExpertRecordBytes;
                            error = read_range(bundle_files_[worker], destination,
                                fill_bundle_offsets_[job], fill_bundle_bytes_[job]);
                        } else {
                            void* const destination = host_cache_mapped(
                                layer, fill_slots_[job]);
                            if (unbuffered_reads_)
                                error = read_range(unbuffered_files_[worker],
                                    destination,
                                    experts_.core_record_offset(
                                        layer, fill_experts_[job]),
                                    kExpertRecordBytes);
                            else
                                std::memcpy(destination,
                                    experts_.core_record(layer, fill_experts_[job]),
                                    static_cast<size_t>(kExpertRecordBytes));
                        }
                        if (error != ERROR_SUCCESS) {
                            std::lock_guard<std::mutex> error_lock(fill_mutex_);
                            fill_failed_ = true;
                            fill_error_ = error;
                        }
                        {
                            std::lock_guard<std::mutex> progress_lock(fill_mutex_);
                            fill_job_done_[job] = true;
                        }
                        fill_progress_.notify_one();
                    }
                    lock.lock();
                    if (++fill_completed_ == active_fill_workers_) fill_done_.notify_one();
                }
            });
        }
        for (Buffer& buffer : batch_staging_)
            buffer = create_host_buffer_uninitialized(runtime_, kExpertRecordBytes);
    }

    ~WeightStreamer() {
        {
            std::lock_guard<std::mutex> lock(fill_mutex_);
            fill_stop_ = true;
        }
        fill_start_.notify_all();
        for (std::thread& worker : fill_workers_)
            if (worker.joinable()) worker.join();
        for (HANDLE file : unbuffered_files_)
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        for (HANDLE file : bundle_files_)
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        for (HANDLE event : read_events_)
            if (event) CloseHandle(event);
        if (bundle_staging_) VirtualFree(bundle_staging_, 0, MEM_RELEASE);
        if (shared_direct_file_ != INVALID_HANDLE_VALUE)
            CloseHandle(shared_direct_file_);
        for (Buffer& buffer : batch_staging_) destroy_buffer(runtime_, buffer);
        destroy_buffer(runtime_, global_staging_);
        destroy_buffer(runtime_, global_device_);
        for (LayerHostExpertCache& cache : host_expert_caches_)
            cache.destroy_resources();
        destroy_buffer(runtime_, shared_staging_);
        for (Buffer& buffer : resident_layers_) destroy_buffer(runtime_, buffer);
        for (Buffer& buffer : shared_device_) destroy_buffer(runtime_, buffer);
    }

    uint64_t load_shared_layer(uint32_t layer, uint32_t parity) {
        if (parity >= kSharedBuffers) throw std::runtime_error("Invalid shared buffer parity");
        if (resident_shared_) return 0;
        const auto range = shared_index_.layer_range(layer);
        const uint64_t bytes = range.second - range.first;
        read_shared_range(shared_staging_.mapped, range.first, bytes);
        flush_buffer(runtime_, shared_staging_);
        shared_file_base_[parity] = range.first;
        shared_file_bytes_[parity] = bytes;
        return transfer_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{};
            copy.size = bytes;
            vkfn::CmdCopyBuffer(command, shared_staging_.handle,
                                shared_device_[parity].handle, 1, &copy);
            transfer_barrier(command, shared_device_[parity]);
        });
    }

    uint64_t prepare_missing_experts(uint32_t layer, GlobalExpertCache& cache,
                                  const std::array<uint32_t, kTopK>& experts,
                                  const std::array<uint32_t, kTopK>& slots,
                                  const std::vector<uint32_t>& missing_ranks,
                                  const std::vector<uint32_t>& promotion_ranks,
                                  std::array<uint32_t, kTopK>& host_slots,
                                  ExpertReadiness* readiness = nullptr) {
        if (missing_ranks.empty()) return 0;
        std::vector<uint32_t> fill_ranks;
        std::array<bool, kTopK> host_hits{};
        for (uint32_t rank : missing_ranks) {
            bool hit = false;
            host_slots[rank] = resolve_host_record(
                layer, experts[rank], experts_.core_record(layer, experts[rank]),
                rank, hit);
            host_hits[rank] = hit;
            if (hit) ++host_cache_hits_;
            else {
                ++host_cache_misses_;
                if (!host_expert_caches_[layer].external())
                    fill_ranks.push_back(rank);
            }
            if (readiness)
                readiness->all_vram_or_ram &=
                    hit || host_expert_caches_[layer].external();
        }
        auto submit_promotions = [&](const std::vector<uint32_t>& ranks) {
            if (ranks.empty()) return uint64_t{0};
            const uint64_t timeline = transfer_.submit([&](VkCommandBuffer command) {
                for (uint32_t rank : ranks) {
                    const DescriptorRange source = host_transfer_record(
                        layer, host_slots[rank], rank);
                    const DescriptorRange destination = cache.record(slots[rank]);
                    VkBufferCopy copy{};
                    copy.srcOffset = source.offset;
                    copy.dstOffset = destination.offset;
                    copy.size = kExpertRecordBytes;
                    vkfn::CmdCopyBuffer(command, source.buffer,
                                        destination.buffer, 1, &copy);
                    transfer_barrier(command, cache.arena(slots[rank]));
                }
            });
            if (blueprint_) for (uint32_t rank : ranks) {
                uint32_t staging_slot = UINT32_MAX;
                if (global_host_cache_ &&
                    (global_host_cache_->transient(host_slots[rank]) ||
                     (global_host_cache_->plain_slot(host_slots[rank]) &&
                      !global_host_cache_->imported_plain_slot(host_slots[rank]))))
                    staging_slot = rank;
                blueprint_->begin_h2d(layer, experts[rank], slots[rank],
                                      staging_slot, timeline);
            }
            return timeline;
        };
        uint64_t early_ready = 0;
        std::vector<uint32_t> early_ranks;
        if (split_expert_transfer_ && global_host_cache_ &&
            !global_host_cache_->uses_plain_backing()) {
            for (uint32_t rank : promotion_ranks)
                if (host_hits[rank]) early_ranks.push_back(rank);
            early_ready = submit_promotions(early_ranks);
            // A later miss can select the same LFU record that supplied an
            // earlier hit in this six-route batch.  Do not let direct I/O
            // overwrite that range until its finite copy has completed.
            bool aliases_fill = false;
            for (uint32_t early : early_ranks)
                for (uint32_t fill : fill_ranks)
                    aliases_fill |= host_slots[early] == host_slots[fill];
            if (aliases_fill && early_ready) transfer_.wait(early_ready);
            if (aliases_fill && blueprint_)
                blueprint_->complete_timeline(early_ready);
        }
        // Exact routes expose up to six independent 12.75-MiB records.  Fault
        // them from the mapped checkpoint concurrently so the NVMe queue and
        // CPU memory channels are not serialized behind one memcpy.
        if (!fill_ranks.empty()) {
            if (bundle_index_)
                plan_bundle_reads(layer, experts, fill_ranks);
            if (readiness) {
                readiness->ssd_miss = true;
                readiness->read_count += bundle_index_ ?
                    fill_bundle_read_count_ :
                    static_cast<uint32_t>(fill_ranks.size());
            }
            if (blueprint_) for (uint32_t rank : fill_ranks)
                blueprint_->begin_disk_read(layer, experts[rank], rank);
            std::unique_lock<std::mutex> lock(fill_mutex_);
            fill_layer_ = layer;
            fill_count_ = static_cast<uint32_t>(fill_ranks.size());
            fill_completed_ = 0;
            fill_failed_ = false;
            fill_error_ = ERROR_SUCCESS;
            for (uint32_t job = 0; job < fill_count_; ++job) {
                const uint32_t rank = fill_ranks[job];
                fill_ranks_[job] = rank;
                fill_slots_[job] = host_slots[rank];
                fill_experts_[job] = experts[rank];
            }
            ++fill_generation_;
            fill_start_.notify_all();
            fill_done_.wait(lock, [&] {
                return fill_completed_ == active_fill_workers_;
            });
            if (fill_failed_)
                throw std::runtime_error("Unbuffered DeepSeek expert read failed (Win32 " +
                    std::to_string(fill_error_) + ")");
            if (bundle_index_) {
                for (uint32_t job = 0; job < fill_count_; ++job) {
                    const uint32_t rank = fill_ranks_[job];
                    const uint32_t scratch = fill_bundle_rank_staging_[rank];
                    if (scratch >= kTopK)
                        throw std::runtime_error(
                            "DeepSeek bundle scratch rank was not populated");
                    std::memcpy(host_cache_mapped(layer, fill_slots_[job]),
                        static_cast<const uint8_t*>(bundle_staging_) +
                            static_cast<uint64_t>(scratch) * kExpertRecordBytes,
                        static_cast<size_t>(kExpertRecordBytes));
                }
            }
        }
        for (uint32_t rank : fill_ranks) {
            if (blueprint_) {
                const uint32_t slot = host_slots[rank];
                const bool persistent = !global_host_cache_ ||
                    !global_host_cache_->transient(slot);
                const uint32_t location = global_host_cache_ ? slot :
                    ExpertBlueprint::local_ram_location(layer, slot);
                blueprint_->complete_disk_read(
                    layer, experts[rank], location, persistent);
            }
            flush_host_record(layer, host_slots[rank]);
        }
        // Ordinary RAM can exceed the AMD driver's host-visible Vulkan
        // allocation ceiling.  Copy each selected plain-RAM record through its
        // rank's existing bounded Vulkan staging buffer before the H2D submit.
        for (uint32_t rank : missing_ranks)
            stage_plain_host_record(
                layer, experts[rank], host_slots[rank], rank);
        if (global_host_cache_ && global_host_cache_->uses_plain_backing()) {
            std::vector<uint32_t> protected_l1;
            for (uint32_t rank : missing_ranks) {
                const uint32_t slot = host_slots[rank];
                if (!global_host_cache_->transient(slot) &&
                    !global_host_cache_->plain_slot(slot))
                    protected_l1.push_back(slot);
            }
            for (uint32_t rank : missing_ranks) {
                const uint32_t slot = host_slots[rank];
                if (!host_hits[rank] || global_host_cache_->transient(slot) ||
                    !global_host_cache_->plain_slot(slot)) continue;
                const bool staged_valid =
                    !global_host_cache_->imported_plain_slot(slot);
                const uint32_t promoted = global_host_cache_->promote_from_l2(
                    layer, experts[rank], slot, batch_staging_[rank].mapped,
                    staged_valid, protected_l1);
                if (promoted != UINT32_MAX) {
                    host_slots[rank] = promoted;
                    protected_l1.push_back(promoted);
                }
            }
        }
        if (promotion_ranks.empty()) return early_ready;
        if (early_ranks.empty()) return submit_promotions(promotion_ranks);
        std::vector<uint32_t> late_ranks;
        for (uint32_t rank : promotion_ranks)
            if (!host_hits[rank]) late_ranks.push_back(rank);
        const uint64_t late_ready = submit_promotions(late_ranks);
        return late_ready ? late_ready : early_ready;
    }

    ProgressiveExpertBatch begin_progressive_experts(
        uint32_t layer, GlobalExpertCache& cache,
        const std::array<uint32_t, kTopK>& experts,
        const std::array<uint32_t, kTopK>& slots,
        const std::vector<uint32_t>& missing_ranks,
        const std::vector<uint32_t>& promotion_ranks) {
        if (missing_ranks.empty())
            throw std::runtime_error("Progressive acquisition requires a cache miss");
        if (progressive_fill_active_)
            throw std::runtime_error("Nested progressive expert acquisition");
        // Canonical host records transfer directly.  All-plain records use one
        // bounded per-rank Vulkan staging record before the same finite H2D
        // submission.  Hybrid/bundle backing retains its separate lifetime
        // rules and stays on the all-six acquisition path.
        if (!global_host_cache_ ||
            (global_host_cache_->uses_plain_backing() &&
             !global_host_cache_->all_plain_backing()) ||
            bundle_index_ || host_expert_caches_[layer].external())
            throw std::runtime_error(
                "DSV4_PROGRESSIVE_EXPERTS requires canonical or all-plain global host backing");
        ProgressiveExpertBatch batch{};
        batch.layer = layer;
        batch.experts = experts;
        batch.slots = slots;
        batch.host_slots.fill(UINT32_MAX);
        batch.fill_job.fill(UINT32_MAX);
        batch.active = true;
        std::vector<uint32_t> fill_ranks;
        for (uint32_t rank : missing_ranks) {
            if (rank >= kTopK ||
                std::find(promotion_ranks.begin(), promotion_ranks.end(), rank) ==
                    promotion_ranks.end())
                throw std::runtime_error(
                    "Progressive V1 requires a full-record promotion per miss");
        }
        global_host_cache_->begin_route_resolve();
        try {
          for (uint32_t rank : missing_ranks) {
            batch.missing[rank] = true;
            bool hit = false;
            batch.host_slots[rank] = resolve_host_record(
                layer, experts[rank], experts_.core_record(layer, experts[rank]),
                rank, hit);
            batch.host_hit[rank] = hit;
            if (hit) {
                ++host_cache_hits_;
            } else {
                ++host_cache_misses_;
                fill_ranks.push_back(rank);
                batch.disk_pending[rank] = true;
            }
            batch.readiness.all_vram_or_ram &= hit;
          }
        } catch (...) {
            global_host_cache_->end_route_resolve();
            throw;
        }
        global_host_cache_->end_route_resolve();
        for (uint32_t first : missing_ranks)
            for (uint32_t second : missing_ranks)
                if (first < second &&
                    batch.host_slots[first] == batch.host_slots[second])
                    throw std::runtime_error(
                        "Progressive acquisition resolved aliased host records");
        batch.fill_count = static_cast<uint32_t>(fill_ranks.size());
        batch.disk_remaining = batch.fill_count;
        batch.readiness.ssd_miss = !fill_ranks.empty();
        batch.readiness.read_count = batch.fill_count;
        if (fill_ranks.empty()) return batch;

        if (blueprint_) for (uint32_t rank : fill_ranks)
            blueprint_->begin_disk_read(layer, experts[rank], rank);
        {
            std::lock_guard<std::mutex> lock(fill_mutex_);
            fill_layer_ = layer;
            fill_count_ = batch.fill_count;
            fill_completed_ = 0;
            fill_failed_ = false;
            fill_error_ = ERROR_SUCCESS;
            fill_job_done_.fill(false);
            for (uint32_t job = 0; job < batch.fill_count; ++job) {
                const uint32_t rank = fill_ranks[job];
                fill_ranks_[job] = rank;
                fill_slots_[job] = batch.host_slots[rank];
                fill_experts_[job] = experts[rank];
                batch.fill_job[rank] = job;
            }
            progressive_fill_active_ = true;
            ++fill_generation_;
        }
        fill_start_.notify_all();
        return batch;
    }

    // Returns whichever disk rank has completed first.  The host wait is
    // finite and no shader polls this state.
    uint32_t wait_next_progressive_disk(ProgressiveExpertBatch& batch) {
        if (!batch.active || batch.disk_remaining == 0u) return UINT32_MAX;
        const auto started = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(fill_mutex_);
        const bool ready = fill_progress_.wait_for(lock, std::chrono::seconds(10), [&] {
            if (fill_failed_) return true;
            for (uint32_t rank = 0; rank < kTopK; ++rank)
                if (batch.disk_pending[rank] &&
                    fill_job_done_[batch.fill_job[rank]]) return true;
            return false;
        });
        batch.disk_wait_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        if (!ready)
            throw std::runtime_error("Progressive expert read exceeded 10 seconds");
        if (fill_failed_)
            throw std::runtime_error("Unbuffered progressive expert read failed (Win32 " +
                std::to_string(fill_error_) + ")");
        uint32_t selected = UINT32_MAX;
        for (uint32_t rank = 0; rank < kTopK; ++rank)
            if (batch.disk_pending[rank] &&
                fill_job_done_[batch.fill_job[rank]]) {
                selected = rank;
                break;
            }
        if (selected == UINT32_MAX)
            throw std::runtime_error("Progressive expert completion drift");
        batch.disk_pending[selected] = false;
        --batch.disk_remaining;
        lock.unlock();
        if (blueprint_) {
            const uint32_t slot = batch.host_slots[selected];
            const bool persistent = !global_host_cache_->transient(slot);
            blueprint_->complete_disk_read(
                batch.layer, batch.experts[selected], slot, persistent);
        }
        flush_host_record(batch.layer, batch.host_slots[selected]);
        return selected;
    }

    uint64_t submit_progressive_promotion(ProgressiveExpertBatch& batch,
                                          GlobalExpertCache& cache,
                                          uint32_t rank) {
        if (!batch.active || rank >= kTopK || !batch.missing[rank] ||
            batch.disk_pending[rank] || batch.submitted[rank])
            throw std::runtime_error("Invalid progressive expert promotion");
        stage_plain_host_record(batch.layer, batch.experts[rank],
                                batch.host_slots[rank], rank);
        const DescriptorRange source = host_transfer_record(
            batch.layer, batch.host_slots[rank], rank);
        const DescriptorRange destination = cache.record(batch.slots[rank]);
        const uint64_t timeline = transfer_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{};
            copy.srcOffset = source.offset;
            copy.dstOffset = destination.offset;
            copy.size = kExpertRecordBytes;
            vkfn::CmdCopyBuffer(command, source.buffer, destination.buffer, 1, &copy);
            transfer_barrier(command, cache.arena(batch.slots[rank]));
        });
        batch.submitted[rank] = true;
        if (blueprint_) {
            uint32_t staging_slot = UINT32_MAX;
            if (global_host_cache_->transient(batch.host_slots[rank]) ||
                global_host_cache_->plain_slot(batch.host_slots[rank]))
                staging_slot = rank;
            blueprint_->begin_h2d(batch.layer, batch.experts[rank],
                                  batch.slots[rank], staging_slot, timeline);
        }
        return timeline;
    }

    void finish_progressive_experts(ProgressiveExpertBatch& batch) {
        if (!batch.active) return;
        if (batch.disk_remaining != 0u)
            throw std::runtime_error("Progressive acquisition left disk ranks pending");
        for (uint32_t rank = 0; rank < kTopK; ++rank)
            if (batch.missing[rank] && !batch.submitted[rank])
                throw std::runtime_error("Progressive acquisition left a rank unsubmitted");
        if (batch.fill_count) {
            std::unique_lock<std::mutex> lock(fill_mutex_);
            if (!fill_done_.wait_for(lock, std::chrono::seconds(10), [&] {
                    return fill_completed_ == active_fill_workers_;
                }))
                throw std::runtime_error("Progressive read workers exceeded 10 seconds");
            progressive_fill_active_ = false;
        }
        batch.active = false;
    }

    uint64_t prepare_batch_experts(uint32_t layer, GlobalExpertCache& cache,
                                   const uint32_t* route_words,
                                   const GlobalExpertCache::BatchSlots& slots,
                                   const std::vector<uint32_t>& missing_flat) {
        uint64_t ready = 0;
        for (uint32_t begin = 0; begin < missing_flat.size(); begin += kTopK) {
            if (ready) {
                transfer_.wait(ready);
                if (blueprint_) blueprint_->complete_timeline(ready);
            }
            const uint32_t count = std::min<uint32_t>(kTopK,
                static_cast<uint32_t>(missing_flat.size()) - begin);
            const uint32_t workers = std::min<uint32_t>(active_fill_workers_, count);
            if (blueprint_) for (uint32_t local = 0; local < count; ++local) {
                const uint32_t flat = missing_flat[begin + local];
                const uint32_t token = flat / kTopK, rank = flat % kTopK;
                blueprint_->begin_disk_read(
                    layer, route_words[token * 64u + rank], local);
            }
            std::atomic<DWORD> read_error{ERROR_SUCCESS};
            std::vector<std::thread> threads;
            for (uint32_t worker = 0; worker < workers; ++worker)
                threads.emplace_back([&, worker] {
                    for (uint32_t local = worker; local < count; local += workers) {
                        const uint32_t flat = missing_flat[begin + local];
                        const uint32_t token = flat / kTopK, rank = flat % kTopK;
                        const uint32_t expert = route_words[token * 64u + rank];
                        if (!budgeted_io_) {
                            std::memcpy(batch_staging_[local].mapped,
                                experts_.core_record(layer, expert), kExpertRecordBytes);
                            continue;
                        }
                        const uint64_t file_offset =
                            experts_.core_record_offset(layer, expert);
                        const bool aligned =
                            (reinterpret_cast<uintptr_t>(batch_staging_[local].mapped) &
                             4095u) == 0u && (file_offset & 4095u) == 0u &&
                            (kExpertRecordBytes & 4095u) == 0u;
                        if (!aligned) {
                            read_error.store(ERROR_INVALID_PARAMETER);
                            continue;
                        }
                        OVERLAPPED operation{};
                        operation.Offset = static_cast<DWORD>(file_offset);
                        operation.OffsetHigh = static_cast<DWORD>(file_offset >> 32u);
                        operation.hEvent = read_events_[worker];
                        ResetEvent(operation.hEvent);
                        const BOOL started = ReadFile(unbuffered_files_[worker],
                            batch_staging_[local].mapped,
                            static_cast<DWORD>(kExpertRecordBytes), nullptr, &operation);
                        const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
                        DWORD transferred = 0;
                        bool read_ok = false;
                        if (started || start_error == ERROR_IO_PENDING) {
                            const DWORD waited = WaitForSingleObject(operation.hEvent, 10000u);
                            if (waited == WAIT_OBJECT_0) {
                                read_ok = GetOverlappedResult(unbuffered_files_[worker],
                                    &operation, &transferred, FALSE) != FALSE;
                            } else {
                                CancelIoEx(unbuffered_files_[worker], &operation);
                                read_error.store(ERROR_TIMEOUT);
                            }
                        } else {
                            read_error.store(start_error);
                        }
                        if (!read_ok || transferred != kExpertRecordBytes) {
                            DWORD error = GetLastError();
                            if (error == ERROR_SUCCESS) error = ERROR_READ_FAULT;
                            read_error.store(error);
                        }
                    }
                });
            for (std::thread& thread : threads) thread.join();
            if (read_error.load() != ERROR_SUCCESS)
                throw std::runtime_error(
                    "Budgeted batched DeepSeek expert read failed (Win32 " +
                    std::to_string(read_error.load()) + ")");
            if (budgeted_io_)
                batch_direct_disk_bytes_ += static_cast<uint64_t>(count) *
                                            kExpertRecordBytes;
            for (uint32_t local = 0; local < count; ++local) {
                if (blueprint_) {
                    const uint32_t flat = missing_flat[begin + local];
                    const uint32_t token = flat / kTopK, rank = flat % kTopK;
                    blueprint_->complete_disk_read(
                        layer, route_words[token * 64u + rank], local, false);
                }
                flush_buffer(runtime_, batch_staging_[local]);
            }
            ready = transfer_.submit([&](VkCommandBuffer command) {
                for (uint32_t local = 0; local < count; ++local) {
                    const uint32_t flat = missing_flat[begin + local];
                    const uint32_t token = flat / kTopK, rank = flat % kTopK;
                    const DescriptorRange destination = cache.record(slots[token][rank]);
                    VkBufferCopy copy{0, destination.offset, kExpertRecordBytes};
                    vkfn::CmdCopyBuffer(command, batch_staging_[local].handle,
                        destination.buffer, 1, &copy);
                    transfer_barrier(command, cache.arena(slots[token][rank]));
                }
            });
            if (blueprint_) for (uint32_t local = 0; local < count; ++local) {
                const uint32_t flat = missing_flat[begin + local];
                const uint32_t token = flat / kTopK, rank = flat % kTopK;
                blueprint_->begin_h2d(layer, route_words[token * 64u + rank],
                                      slots[token][rank], local, ready);
            }
        }
        return ready;
    }

    DescriptorRange host_expert_record(uint32_t layer, uint32_t slot,
                                       uint32_t rank) const {
        if (layer >= kLayers) throw std::runtime_error("Invalid host expert layer");
        if (rank >= kTopK) throw std::runtime_error("Invalid host expert rank");
        if (global_host_cache_ && !global_host_cache_->transient(slot) &&
            global_host_cache_->plain_slot(slot)) {
            if (global_host_cache_->imported_plain_slot(slot))
                throw std::runtime_error(
                    "Imported L2 expert cannot be bound to a shader");
            return arena_range(batch_staging_[rank], 0, kExpertRecordBytes);
        }
        return host_record(layer, slot);
    }
    DescriptorRange host_expert_placeholder(uint32_t layer) const {
        if (layer >= kLayers) throw std::runtime_error("Invalid host expert layer");
        return arena_range(shared_staging_, 0, shared_staging_.size);
    }
    uint64_t host_cache_hits() const { return host_cache_hits_; }
    uint64_t host_cache_misses() const { return host_cache_misses_; }
    uint64_t host_cache_admission_rejections() const {
        return global_host_cache_ ? global_host_cache_->admission_rejections() : 0;
    }
    uint64_t plain_stage_bytes() const { return plain_stage_bytes_; }
    uint64_t hybrid_promotions() const {
        return global_host_cache_ ? global_host_cache_->hybrid_promotions() : 0;
    }
    uint64_t hybrid_copy_bytes() const {
        return global_host_cache_ ? global_host_cache_->hybrid_copy_bytes() : 0;
    }
    uint64_t imported_plain_bytes() const {
        return global_host_cache_ ? global_host_cache_->imported_plain_bytes() : 0;
    }
    uint32_t imported_plain_blocks() const {
        return global_host_cache_ ? global_host_cache_->imported_plain_blocks() : 0;
    }
    uint64_t imported_plain_transfer_bytes() const {
        return imported_plain_transfer_bytes_;
    }
    bool plain_host_cache() const {
        return global_host_cache_ && global_host_cache_->uses_plain_backing();
    }
    uint64_t batch_direct_disk_bytes() const { return batch_direct_disk_bytes_; }
    uint32_t host_cache_slots() const {
        return global_host_cache_ ? global_host_cache_->capacity() :
                                    kLayers * kHostExpertSlotsPerLayer;
    }
    uint32_t fill_host_cache_remaining(double& seconds) {
        if (!global_host_cache_)
            throw std::runtime_error(
                "DeepSeek RAM-cache top-off requires the global cache");
        return global_host_cache_->fill_remaining_uniform(experts_, seconds);
    }
    bool set_host_retention_hint(uint32_t layer, uint32_t expert) {
        if (!global_host_cache_) return false;
        return global_host_cache_->set_retention_hint(layer, expert);
    }
    void clear_host_retention_hint() {
        if (global_host_cache_) global_host_cache_->clear_retention_hint();
    }
    GlobalHostExpertCache::HintMetrics host_hint_metrics() const {
        return global_host_cache_ ? global_host_cache_->hint_metrics() :
                                    GlobalHostExpertCache::HintMetrics{};
    }
    void reset_host_hint_metrics() {
        if (global_host_cache_) global_host_cache_->reset_hint_metrics();
    }
    uint64_t host_budget_bytes() const { return host_budget_bytes_; }
    bool budgeted_io() const { return budgeted_io_; }
    bool expert_bundles() const { return bundle_index_ != nullptr; }
    BundleMetrics bundle_metrics() const { return bundle_metrics_; }
    void reset_bundle_metrics() { bundle_metrics_ = {}; }

    const Buffer& shared_buffer(uint32_t layer, uint32_t parity) const {
        if (layer >= kLayers || parity >= kSharedBuffers)
            throw std::runtime_error("Invalid DeepSeek shared-buffer key");
        return resident_shared_ ? resident_layers_[layer] : shared_device_[parity];
    }
    uint64_t shared_file_base(uint32_t layer, uint32_t parity) const {
        if (layer >= kLayers || parity >= kSharedBuffers)
            throw std::runtime_error("Invalid DeepSeek shared-buffer key");
        // A fallback buffer always receives its layer range at offset zero.
        // Return that prospective base even before the first load so descriptor
        // sets can be built once during executor construction.
        return shared_index_.layer_range(layer).first;
    }
    bool shared_resident() const { return resident_shared_; }
    const Buffer& global_buffer() const { return global_device_; }
    uint64_t global_file_base() const { return global_file_base_; }

private:
    struct SelectedBundleCopy {
        uint32_t rank = 0;
        ExpertBundleIndex::Copy copy{};
    };
    static uint32_t bundle_run_count(
        std::vector<SelectedBundleCopy> selected) {
        if (selected.empty()) return 0;
        std::sort(selected.begin(), selected.end(),
            [](const SelectedBundleCopy& first,
               const SelectedBundleCopy& second) {
                if (first.copy.bundle_id != second.copy.bundle_id)
                    return first.copy.bundle_id < second.copy.bundle_id;
                return first.copy.ordinal < second.copy.ordinal;
            });
        uint32_t runs = 0;
        uint32_t prior_bundle = UINT32_MAX, prior_ordinal = UINT32_MAX;
        for (const SelectedBundleCopy& item : selected) {
            if (item.copy.bundle_id != prior_bundle ||
                prior_ordinal == UINT32_MAX ||
                item.copy.ordinal != prior_ordinal + 1u)
                ++runs;
            prior_bundle = item.copy.bundle_id;
            prior_ordinal = item.copy.ordinal;
        }
        return runs;
    }
    void plan_bundle_reads(uint32_t layer,
                           const std::array<uint32_t, kTopK>& experts,
                           const std::vector<uint32_t>& fill_ranks) {
        if (!bundle_index_ || !bundle_staging_ || fill_ranks.empty())
            throw std::runtime_error("Invalid DeepSeek bundle-read plan request");
        std::vector<SelectedBundleCopy> selected;
        selected.reserve(fill_ranks.size());
        // Multiple copies are expected to remain rare.  Choose incrementally by
        // the actual finite-read count; ties prefer the cheaper/probable copy.
        for (uint32_t rank : fill_ranks) {
            const auto& candidates = bundle_index_->copies(layer, experts[rank]);
            const ExpertBundleIndex::Copy* best = nullptr;
            uint32_t best_runs = UINT32_MAX;
            for (const ExpertBundleIndex::Copy& candidate : candidates) {
                auto trial = selected;
                trial.push_back({rank, candidate});
                const uint32_t runs = bundle_run_count(std::move(trial));
                if (!best || runs < best_runs ||
                    (runs == best_runs &&
                     (candidate.read_cost_bytes < best->read_cost_bytes ||
                      (candidate.read_cost_bytes == best->read_cost_bytes &&
                       (candidate.predicted_probability >
                            best->predicted_probability ||
                        (candidate.predicted_probability ==
                            best->predicted_probability &&
                         candidate.data_offset < best->data_offset)))))) {
                    best = &candidate;
                    best_runs = runs;
                }
            }
            if (!best) throw std::runtime_error("DeepSeek expert has no bundle copy");
            selected.push_back({rank, *best});
        }
        std::sort(selected.begin(), selected.end(),
            [](const SelectedBundleCopy& first,
               const SelectedBundleCopy& second) {
                if (first.copy.bundle_id != second.copy.bundle_id)
                    return first.copy.bundle_id < second.copy.bundle_id;
                return first.copy.ordinal < second.copy.ordinal;
            });
        fill_bundle_rank_staging_.fill(UINT32_MAX);
        fill_bundle_read_count_ = 0;
        uint32_t scratch_record = 0;
        uint32_t prior_bundle = UINT32_MAX, prior_ordinal = UINT32_MAX;
        for (const SelectedBundleCopy& item : selected) {
            const bool new_run = item.copy.bundle_id != prior_bundle ||
                prior_ordinal == UINT32_MAX ||
                item.copy.ordinal != prior_ordinal + 1u;
            if (new_run) {
                if (fill_bundle_read_count_ >= kTopK)
                    throw std::runtime_error("DeepSeek bundle plan exceeded Top-6 reads");
                const uint32_t run = fill_bundle_read_count_++;
                fill_bundle_offsets_[run] = item.copy.data_offset;
                fill_bundle_bytes_[run] = kExpertRecordBytes;
                fill_bundle_staging_[run] = scratch_record;
            } else {
                fill_bundle_bytes_[fill_bundle_read_count_ - 1u] +=
                    kExpertRecordBytes;
            }
            fill_bundle_rank_staging_[item.rank] = scratch_record++;
            if (blueprint_)
                blueprint_->select_nvme_copy(layer, experts[item.rank],
                    item.copy.copy_index, item.copy.bundle_id,
                    item.copy.data_offset, item.copy.read_cost_bytes,
                    item.copy.predicted_probability);
            prior_bundle = item.copy.bundle_id;
            prior_ordinal = item.copy.ordinal;
        }
        if (scratch_record != fill_ranks.size() || scratch_record > kTopK)
            throw std::runtime_error("DeepSeek bundle scratch plan drift");
        uint64_t bytes = 0;
        for (uint32_t run = 0; run < fill_bundle_read_count_; ++run)
            bytes += fill_bundle_bytes_[run];
        if (bytes != static_cast<uint64_t>(fill_ranks.size()) *
                     kExpertRecordBytes)
            throw std::runtime_error("DeepSeek bundle plan would overread experts");
        bundle_metrics_.read_count += fill_bundle_read_count_;
        bundle_metrics_.read_bytes += bytes;
        ++bundle_metrics_.stalled_layers;
    }

    void read_shared_range(void* destination, uint64_t file_offset, uint64_t bytes) {
        if (!budgeted_io_) {
            std::memcpy(destination, shared_file_.data() + file_offset,
                        static_cast<size_t>(bytes));
            return;
        }
        if (shared_direct_file_ == INVALID_HANDLE_VALUE ||
            (reinterpret_cast<uintptr_t>(destination) & 4095u) != 0u ||
            (file_offset & 4095u) != 0u || (bytes & 4095u) != 0u ||
            bytes > static_cast<uint64_t>(std::numeric_limits<DWORD>::max())) {
            throw std::runtime_error("Unaligned budgeted DeepSeek shared read");
        }
        LARGE_INTEGER offset{};
        offset.QuadPart = static_cast<LONGLONG>(file_offset);
        DWORD transferred = 0;
        const BOOL positioned = SetFilePointerEx(
            shared_direct_file_, offset, nullptr, FILE_BEGIN);
        const BOOL read = positioned && ReadFile(shared_direct_file_, destination,
            static_cast<DWORD>(bytes), &transferred, nullptr);
        if (!read || transferred != bytes)
            throw std::runtime_error("Budgeted DeepSeek shared read failed (Win32 " +
                std::to_string(GetLastError()) + ")");
    }

    uint32_t resolve_host_record(uint32_t layer, uint32_t expert,
                                 const uint8_t* source, uint32_t transient,
                                 bool& hit) {
        if (global_host_cache_)
            return global_host_cache_->resolve(layer, expert, transient, hit);
        return host_expert_caches_[layer].resolve(expert, source, hit);
    }
    void* host_cache_mapped(uint32_t layer, uint32_t slot) {
        if (global_host_cache_) {
            if (global_host_cache_->transient(slot))
                return batch_staging_[global_host_cache_->transient_index(slot)].mapped;
            return global_host_cache_->mapped(slot);
        }
        return host_expert_caches_[layer].mapped(slot);
    }
    void flush_host_record(uint32_t layer, uint32_t slot) {
        if (global_host_cache_) {
            if (global_host_cache_->transient(slot))
                flush_buffer(runtime_,
                    batch_staging_[global_host_cache_->transient_index(slot)]);
            else global_host_cache_->flush(slot);
        }
        else host_expert_caches_[layer].flush(slot);
    }
    void stage_plain_host_record(uint32_t layer, uint32_t expert,
                                 uint32_t slot, uint32_t rank) {
        if (rank >= kTopK) throw std::runtime_error("Invalid plain host staging rank");
        if (!global_host_cache_ || global_host_cache_->transient(slot) ||
            !global_host_cache_->plain_slot(slot)) return;
        if (global_host_cache_->imported_plain_slot(slot)) return;
        std::memcpy(batch_staging_[rank].mapped,
                    global_host_cache_->mapped(slot), kExpertRecordBytes);
        flush_buffer(runtime_, batch_staging_[rank]);
        if (blueprint_) blueprint_->mark_staging(layer, expert, rank);
        plain_stage_bytes_ += kExpertRecordBytes;
    }
    DescriptorRange host_transfer_record(uint32_t layer, uint32_t slot,
                                         uint32_t rank) {
        if (rank >= kTopK) throw std::runtime_error("Invalid host transfer rank");
        if (global_host_cache_ && !global_host_cache_->transient(slot) &&
            global_host_cache_->plain_slot(slot)) {
            if (global_host_cache_->imported_plain_slot(slot)) {
                imported_plain_transfer_bytes_ += kExpertRecordBytes;
                return global_host_cache_->record(slot);
            }
            return arena_range(batch_staging_[rank], 0, kExpertRecordBytes);
        }
        return host_record(layer, slot);
    }
    DescriptorRange host_record(uint32_t layer, uint32_t slot) const {
        if (global_host_cache_) {
            if (global_host_cache_->transient(slot))
                return arena_range(
                    batch_staging_[global_host_cache_->transient_index(slot)],
                    0, kExpertRecordBytes);
            return global_host_cache_->record(slot);
        }
        return host_expert_caches_[layer].record(slot);
    }

    void upload_range(Buffer& destination, Buffer& staging, uint64_t file_offset,
                      uint64_t bytes) {
        uint64_t final_copy = 0;
        for (uint64_t offset = 0; offset < bytes; offset += staging.size) {
            // The staging allocation is reused for every chunk.  Wait before
            // overwriting it; FiniteQueue::submit waits only after record-time,
            // which is too late because the memcpy happens here first.
            if (final_copy) transfer_.wait(final_copy);
            const uint64_t count = std::min<uint64_t>(staging.size, bytes - offset);
            read_shared_range(staging.mapped, file_offset + offset, count);
            flush_buffer(runtime_, staging);
            final_copy = transfer_.submit([&, offset, count](VkCommandBuffer command) {
                VkBufferCopy copy{};
                copy.dstOffset = offset;
                copy.size = count;
                vkfn::CmdCopyBuffer(command, staging.handle, destination.handle, 1, &copy);
                transfer_barrier(command, destination);
            });
        }
        if (final_copy) transfer_.wait(final_copy);
    }

    const Runtime& runtime_;
    const ReadOnlyMapping& shared_file_;
    const SharedIndex& shared_index_;
    const ExpertIndex& experts_;
    FiniteQueue& transfer_;
    std::array<Buffer, kSharedBuffers> shared_device_{};
    std::vector<Buffer> resident_layers_;
    bool resident_shared_ = false;
    Buffer shared_staging_{};
    std::vector<LayerHostExpertCache> host_expert_caches_;
    uint64_t host_cache_hits_ = 0;
    uint64_t host_cache_misses_ = 0;
    uint64_t plain_stage_bytes_ = 0;
    uint64_t imported_plain_transfer_bytes_ = 0;
    uint64_t batch_direct_disk_bytes_ = 0;
    std::array<uint64_t, kSharedBuffers> shared_file_base_{};
    std::array<uint64_t, kSharedBuffers> shared_file_bytes_{};
    Buffer global_device_{};
    Buffer global_staging_{};
    uint64_t global_file_base_ = 0;
    bool external_host_mode_ = false;
    int32_t external_host_layer_ = -1;
    std::unique_ptr<GlobalHostExpertCache> global_host_cache_;
    std::array<std::thread, kMaximumExpertFillWorkers> fill_workers_{};
    std::array<Buffer, kTopK> batch_staging_{};
    std::array<HANDLE, kMaximumExpertFillWorkers> unbuffered_files_{};
    std::array<HANDLE, kMaximumExpertFillWorkers> bundle_files_{};
    std::array<HANDLE, kMaximumExpertFillWorkers> read_events_{};
    HANDLE shared_direct_file_ = INVALID_HANDLE_VALUE;
    uint64_t host_budget_bytes_ = 0;
    uint32_t active_fill_workers_ = kLegacyExpertFillWorkers;
    bool budgeted_io_ = false;
    bool unbuffered_reads_ = false;
    bool overlapped_reads_ = false;
    bool split_expert_transfer_ = false;
    ExpertBlueprint* blueprint_ = nullptr;
    std::unique_ptr<ReadOnlyMapping> bundle_index_file_;
    std::unique_ptr<ExpertBundleIndex> bundle_index_;
    std::string bundle_store_path_;
    void* bundle_staging_ = nullptr;
    uint64_t bundle_staging_bytes_ = 0;
    BundleMetrics bundle_metrics_{};
    std::mutex fill_mutex_;
    std::condition_variable fill_start_;
    std::condition_variable fill_done_;
    std::condition_variable fill_progress_;
    bool fill_stop_ = false;
    bool progressive_fill_active_ = false;
    uint64_t fill_generation_ = 0;
    uint32_t fill_completed_ = 0;
    uint32_t fill_count_ = 0;
    bool fill_failed_ = false;
    DWORD fill_error_ = ERROR_SUCCESS;
    uint32_t fill_layer_ = 0;
    std::array<uint32_t, kTopK> fill_ranks_{};
    std::array<uint32_t, kTopK> fill_slots_{};
    std::array<uint32_t, kTopK> fill_experts_{};
    std::array<bool, kTopK> fill_job_done_{};
    uint32_t fill_bundle_read_count_ = 0;
    std::array<uint64_t, kTopK> fill_bundle_offsets_{};
    std::array<uint64_t, kTopK> fill_bundle_bytes_{};
    std::array<uint32_t, kTopK> fill_bundle_staging_{};
    std::array<uint32_t, kTopK> fill_bundle_rank_staging_{};
};

static std::array<uint32_t, kTopK> read_route(const Runtime& runtime,
                                               const Buffer& routing) {
    invalidate_buffer(runtime, routing);
    const auto* words = static_cast<const uint32_t*>(routing.mapped);
    std::array<uint32_t, kTopK> experts{};
    for (uint32_t rank = 0; rank < kTopK; ++rank) {
        experts[rank] = words[rank];
        if (experts[rank] >= kExperts)
            throw std::runtime_error("DeepSeek router emitted an out-of-range expert");
        for (uint32_t prior = 0; prior < rank; ++prior)
            if (experts[prior] == experts[rank])
                throw std::runtime_error("DeepSeek router emitted a duplicate expert");
    }
    return experts;
}

// Shader bindings are installed after the individually bounded kernels have
// passed static SPIR-V validation.  Keeping the execution split explicit here
// prevents a future optimization from accidentally reintroducing a device-side
// wait for host expert acquisition.
class ExecutorScaffold {
public:
    using SharedPreRecorder =
        std::function<void(VkCommandBuffer, uint32_t, uint32_t)>;
    struct ProgressiveExpertRecorders {
        std::function<void(VkCommandBuffer, uint32_t)> prefix;
        std::function<void(VkCommandBuffer, uint32_t, uint32_t)> rank;
        std::function<void(VkCommandBuffer, uint32_t, uint32_t)> finish;
    };
    struct ProgressiveMetrics {
        uint64_t mixed_layers = 0;
        uint64_t all_ready_fast_layers = 0;
        uint64_t resident_ranks = 0;
        uint64_t ram_ranks = 0;
        uint64_t disk_ranks = 0;
        uint64_t finite_submissions = 0;
        double disk_wait_seconds = 0.0;
        double schedule_seconds = 0.0;
    };
    struct BatchExpertResolution {
        GlobalExpertCache::BatchSlots slots{};
        uint64_t ready = 0;
        uint32_t hits = 0;
        uint32_t misses = 0;
    };
    ExecutorScaffold(const Runtime& runtime, const ReadOnlyMapping& shared_file,
                     const SharedIndex& shared_index, const ExpertIndex& expert_index)
        : runtime_(runtime), compute_(runtime, runtime.queue),
          transfer_(runtime, runtime.secondary_queue),
          blueprint_(), streamer_(runtime, shared_file, shared_index, expert_index,
                                  transfer_, &blueprint_) {
        token_ = create_buffer(runtime, sizeof(uint32_t));
        hidden_ = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(kHcMultiplicity) * kDimension * sizeof(float));
        normalized_ = create_device_buffer(runtime, kDimension * sizeof(float));
        q_rank_ = create_device_buffer(runtime, 1024ull * sizeof(float));
        q_ = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(kHeads) * kHeadDimension * sizeof(float));
        kv_ = create_device_buffer(runtime, kHeadDimension * sizeof(float));
        context_ = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(kHeads) * kHeadDimension * sizeof(float));
        ffn_ = create_device_buffer(runtime,
            static_cast<VkDeviceSize>(kTopK + 1) * kMoeDimension * sizeof(float));
        routing_ = create_buffer(runtime, 64 * sizeof(uint32_t));
        logits_ = create_buffer(runtime, static_cast<VkDeviceSize>(kVocabulary) * sizeof(float));
        const VkDeviceSize kv_layer = static_cast<VkDeviceSize>(kWindow + kShortContext / 4) *
                                      kHeadDimension * sizeof(float);
        kv_cache_ = create_device_buffer(runtime, static_cast<VkDeviceSize>(kLayers) * kv_layer);
        diagnostic_ = create_buffer(runtime,
            static_cast<VkDeviceSize>(kHcMultiplicity) * kDimension * sizeof(float));
        main_targets_ = create_device_buffer(runtime,
            3ull * kHcMultiplicity * kDimension * sizeof(float));
        trace_ = std::getenv("DSV4_TRACE") != nullptr;
        route_trace_ = std::getenv("DSV4_ROUTE_TRACE") != nullptr;
        progressive_experts_ =
            std::getenv("DSV4_PROGRESSIVE_EXPERTS") != nullptr;
        if (progressive_experts_)
            progressive_compute_ = std::make_unique<
                experiment::FiniteQueueRing<8>>(
                    runtime_.device, runtime_.queue, runtime_.queue_family,
                    finite_queue_ring_api());
        uint32_t requested_cache_slots =
            shared_index.header().shared_format == 3 ? 10u : 6u;
        if (const char* configured = std::getenv("DSV4_CACHE_SLOTS")) {
            requested_cache_slots = static_cast<uint32_t>(std::stoul(configured));
            if (requested_cache_slots == 0 ||
                requested_cache_slots > kPersistentExpertSlotsPerLayer)
                throw std::runtime_error("DSV4_CACHE_SLOTS must be in [1,11]");
        }
        cache_ = std::make_unique<GlobalExpertCache>(
            runtime, requested_cache_slots, &blueprint_);
        expert_trace_.open(2u, kLayers, kExperts, kTopK,
                           static_cast<uint32_t>(kExpertRecordBytes),
                           cache_->capacity(), streamer_.host_cache_slots());
    }

    ~ExecutorScaffold() {
        destroy_buffer(runtime_, main_targets_);
        destroy_buffer(runtime_, diagnostic_);
        destroy_buffer(runtime_, kv_cache_);
        destroy_buffer(runtime_, logits_);
        destroy_buffer(runtime_, routing_);
        destroy_buffer(runtime_, ffn_);
        destroy_buffer(runtime_, context_);
        destroy_buffer(runtime_, kv_);
        destroy_buffer(runtime_, q_);
        destroy_buffer(runtime_, q_rank_);
        destroy_buffer(runtime_, normalized_);
        destroy_buffer(runtime_, hidden_);
        destroy_buffer(runtime_, token_);
    }

    // This is the safety-critical outer schedule.  record_pre must end after
    // router selection and make routing host-visible.  An optional finite
    // shared-expert prefix can then execute while the CPU acquires routed
    // experts; the final record consumes both results after the ordinary
    // timeline/transfer dependencies.  No shader waits for host progress.
    template <typename RecordPre, typename RecordPost>
    void run_layer(uint32_t layer, RecordPre&& record_pre, RecordPost&& record_post,
                   const SharedPreRecorder* record_shared_pre = nullptr,
                   const ProgressiveExpertRecorders* progressive = nullptr) {
        if (progressive_experts_ && record_shared_pre)
            throw std::runtime_error(
                "DSV4_PROGRESSIVE_EXPERTS is incompatible with the scalar shared-prefix schedule");
        if (progressive_experts_ &&
            (!progressive || !progressive->prefix || !progressive->rank ||
             !progressive->finish))
            throw std::runtime_error(
                "DSV4_PROGRESSIVE_EXPERTS requires finite expert recorders");
        const auto pre_started = std::chrono::steady_clock::now();
        const uint32_t parity = layer & 1u;
        const uint64_t shared_ready = streamer_.load_shared_layer(layer, parity);
        const uint64_t pre_done = compute_.submit(
            [&](VkCommandBuffer command) { record_pre(command, layer, parity); },
            shared_ready ? transfer_.semaphore() : VK_NULL_HANDLE, shared_ready);
        compute_.wait(pre_done);
        pre_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - pre_started).count();
        if (record_shared_pre) {
            compute_.submit([&](VkCommandBuffer command) {
                (*record_shared_pre)(command, layer, parity);
            });
        }
        const auto acquire_started = std::chrono::steady_clock::now();
        const std::array<uint32_t, kTopK> experts = read_route(runtime_, routing_);
        std::array<float, kTopK> route_weights{};
        const auto* trace_route_words =
            static_cast<const uint32_t*>(routing_.mapped);
        for (uint32_t rank = 0; rank < kTopK; ++rank)
            std::memcpy(&route_weights[rank], trace_route_words + kTopK + rank,
                        sizeof(float));
        if (route_trace_) {
            std::cout << "route " << current_position_ << ' ' << layer;
            for (uint32_t expert : experts) std::cout << ' ' << expert;
            std::cout << '\n';
        }
        const AdjacentRank1Predictor::Prediction next_hint =
            adjacent_predictor_.observe(layer, experts);
        std::vector<uint32_t> missing;
        std::vector<uint32_t> promotions;
        const std::array<uint32_t, kTopK> slots =
            cache_->resolve(layer, experts, missing, promotions);
        current_slots_[layer] = slots;
        current_host_[layer].fill(false);
        current_host_slots_[layer].fill(UINT32_MAX);
        cache_hits_ += kTopK - missing.size();
        cache_misses_ += missing.size();
        expert_transfer_bytes_ += promotions.size() * kExpertRecordBytes;
        auto* route_words = static_cast<uint32_t*>(routing_.mapped);
        for (uint32_t rank = 0; rank < kTopK; ++rank)
            route_words[16 + rank] = slots[rank];
        flush_buffer(runtime_, routing_);
        for (uint32_t rank : missing) current_host_[layer][rank] = true;
        // A promoted record is available in its final device slot once the
        // finite transfer submission signals. Execute it there immediately;
        // this avoids reading the same bytes over PCIe in the shader as well
        // as copying them for the next token.
        for (uint32_t rank : promotions) current_host_[layer][rank] = false;
        for (uint32_t rank : missing)
            if (current_host_[layer][rank]) direct_host_expert_bytes_ += kExpertRecordBytes;
        if (progressive_experts_ && !missing.empty()) {
            const auto schedule_started = std::chrono::steady_clock::now();
            WeightStreamer::ProgressiveExpertBatch batch =
                streamer_.begin_progressive_experts(
                    layer, *cache_, experts, slots, missing, promotions);
            current_host_slots_[layer] = batch.host_slots;
            WeightStreamer::ExpertReadiness readiness = batch.readiness;
            std::array<uint64_t, kTopK> transfer_timelines{};

            progressive_compute_->submit([&](VkCommandBuffer command) {
                progressive->prefix(command, layer);
            });
            ++progressive_metrics_.finite_submissions;
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                if (batch.missing[rank]) continue;
                progressive_compute_->submit([&](VkCommandBuffer command) {
                    progressive->rank(command, layer, rank);
                });
                ++progressive_metrics_.resident_ranks;
                ++progressive_metrics_.finite_submissions;
            }
            // RAM hits can be promoted while cold records are still being read.
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                if (!batch.missing[rank] || !batch.host_hit[rank]) continue;
                transfer_timelines[rank] = streamer_.submit_progressive_promotion(
                    batch, *cache_, rank);
                progressive_compute_->submit(
                    [&](VkCommandBuffer command) {
                        progressive->rank(command, layer, rank);
                    }, transfer_.semaphore(), transfer_timelines[rank]);
                ++progressive_metrics_.ram_ranks;
                ++progressive_metrics_.finite_submissions;
            }
            while (batch.disk_remaining) {
                const uint32_t rank = streamer_.wait_next_progressive_disk(batch);
                transfer_timelines[rank] = streamer_.submit_progressive_promotion(
                    batch, *cache_, rank);
                progressive_compute_->submit(
                    [&](VkCommandBuffer command) {
                        progressive->rank(command, layer, rank);
                    }, transfer_.semaphore(), transfer_timelines[rank]);
                ++progressive_metrics_.disk_ranks;
                ++progressive_metrics_.finite_submissions;
            }
            streamer_.finish_progressive_experts(batch);
            progressive_metrics_.disk_wait_seconds += batch.disk_wait_seconds;
            bool resident_hint = false;
            if (next_hint.valid)
                resident_hint = streamer_.set_host_retention_hint(
                    next_hint.layer, next_hint.expert);
            else
                streamer_.clear_host_retention_hint();
            adjacent_predictor_.record_host_hint(resident_hint);
            blueprint_.record_route(false, readiness.all_vram_or_ram,
                                    readiness.ssd_miss);
            acquire_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - acquire_started).count();

            const auto post_started = std::chrono::steady_clock::now();
            const uint64_t finished = progressive_compute_->submit(
                [&](VkCommandBuffer command) {
                    progressive->finish(command, layer, parity);
                });
            ++progressive_metrics_.finite_submissions;
            progressive_compute_->wait(finished);
            for (uint64_t timeline : transfer_timelines)
                if (timeline) blueprint_.complete_timeline(timeline);
            if (expert_trace_.enabled()) {
                uint8_t device_mask = 0, ram_mask = 0, disk_mask = 0;
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    const uint8_t bit = static_cast<uint8_t>(1u << rank);
                    if (!batch.missing[rank]) device_mask |= bit;
                    else if (batch.host_hit[rank]) ram_mask |= bit;
                    else disk_mask |= bit;
                }
                expert_trace_.event(current_position_, layer, experts,
                    route_weights, device_mask, ram_mask, disk_mask, slots,
                    acquire_started, std::chrono::steady_clock::now());
            }
            post_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - post_started).count();
            ++progressive_metrics_.mixed_layers;
            progressive_metrics_.schedule_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - schedule_started).count();
            return;
        }
        if (progressive_experts_) ++progressive_metrics_.all_ready_fast_layers;
        WeightStreamer::ExpertReadiness readiness;
        const uint64_t experts_ready = streamer_.prepare_missing_experts(
            layer, *cache_, experts, slots, missing,
            promotions, current_host_slots_[layer], &readiness);
        bool resident_hint = false;
        if (next_hint.valid)
            resident_hint = streamer_.set_host_retention_hint(
                next_hint.layer, next_hint.expert);
        else
            streamer_.clear_host_retention_hint();
        adjacent_predictor_.record_host_hint(resident_hint);
        blueprint_.record_route(missing.empty(), readiness.all_vram_or_ram,
                                readiness.ssd_miss);
        acquire_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - acquire_started).count();
        const auto post_started = std::chrono::steady_clock::now();
        const uint64_t post_done = compute_.submit(
            [&](VkCommandBuffer command) { record_post(command, layer, parity); },
            experts_ready ? transfer_.semaphore() : VK_NULL_HANDLE,
            experts_ready);
        compute_.wait(post_done);
        blueprint_.complete_timeline(experts_ready);
        if (expert_trace_.enabled()) {
            uint8_t device_mask = 0, ram_mask = 0, disk_mask = 0;
            std::array<bool, kTopK> was_missing{};
            for (uint32_t rank : missing) was_missing[rank] = true;
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const uint8_t bit = static_cast<uint8_t>(1u << rank);
                if (!was_missing[rank]) device_mask |= bit;
                else if (readiness.ssd_miss) disk_mask |= bit;
                else ram_mask |= bit;
            }
            expert_trace_.event(current_position_, layer, experts,
                route_weights, device_mask, ram_mask, disk_mask, slots,
                acquire_started, std::chrono::steady_clock::now());
        }
        post_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - post_started).count();
    }

    template <typename RecordEmbedding, typename RecordPre,
              typename RecordPost, typename RecordFinal>
    uint32_t run_token(uint32_t token, uint32_t position,
                       RecordEmbedding&& record_embedding,
                       RecordPre&& record_pre, RecordPost&& record_post,
                       RecordFinal&& record_final,
                       const SharedPreRecorder* record_shared_pre = nullptr,
                       const ProgressiveExpertRecorders* progressive = nullptr) {
        if (position >= kShortContext)
            throw std::runtime_error("Initial DeepSeek path is capped below 128 tokens");
        *static_cast<uint32_t*>(token_.mapped) = token;
        current_position_ = position;
        flush_buffer(runtime_, token_);
        const uint64_t embedded = compute_.submit([&](VkCommandBuffer command) {
            record_embedding(command, token, position);
        });
        compute_.wait(embedded);
        if (trace_ && position == 0u) trace_hidden("embedding", UINT32_MAX);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            run_layer(layer, record_pre, record_post, record_shared_pre, progressive);
            if (trace_ && position == 0u) trace_hidden("layer", layer);
        }
        const uint64_t finalized = compute_.submit([&](VkCommandBuffer command) {
            record_final(command, position);
        });
        compute_.wait(finalized);
        invalidate_buffer(runtime_, token_);
        const uint32_t prediction = *static_cast<const uint32_t*>(token_.mapped);
        if (prediction >= kVocabulary)
            throw std::runtime_error("Greedy shader emitted an invalid token");
        return prediction;
    }

    template <typename RecordEmbedding, typename RecordPre,
              typename RecordPost, typename RecordFinal>
    std::vector<uint32_t> generate(const Tokenizer& tokenizer,
                                   const std::vector<uint32_t>& prompt,
                                   uint32_t maximum_new_tokens,
                                   RecordEmbedding&& record_embedding,
                                   RecordPre&& record_pre, RecordPost&& record_post,
                                   RecordFinal&& record_final,
                                   const SharedPreRecorder* record_shared_pre = nullptr,
                                   const ProgressiveExpertRecorders* progressive = nullptr) {
        if (prompt.empty()) throw std::runtime_error("DeepSeek prompt is empty");
        if (prompt.size() + maximum_new_tokens > kShortContext)
            throw std::runtime_error("Prompt plus generation exceeds short-context cap");
        uint32_t prediction = 0;
        for (uint32_t position = 0; position < prompt.size(); ++position) {
            prediction = run_token(prompt[position], position, record_embedding,
                                   record_pre, record_post, record_final,
                                   record_shared_pre, progressive);
        }
        if (std::getenv("DSV4_FILL_RAM_CACHE")) {
            double fill_seconds = 0.0;
            const uint32_t filled =
                streamer_.fill_host_cache_remaining(fill_seconds);
            std::cout << "RAM cache top-off: " << filled << " records, "
                      << double(streamer_.host_cache_slots()) *
                             double(kExpertRecordBytes) / double(1ull << 30)
                      << " GiB cache, " << fill_seconds << " s\n";
        }
        cache_hits_ = 0;
        cache_misses_ = 0;
        expert_transfer_bytes_ = 0;
        direct_host_expert_bytes_ = 0;
        pre_seconds_ = 0.0;
        post_seconds_ = 0.0;
        acquire_seconds_ = 0.0;
        host_hits_base_ = streamer_.host_cache_hits();
        host_misses_base_ = streamer_.host_cache_misses();
        host_rejections_base_ = streamer_.host_cache_admission_rejections();
        plain_stage_base_ = streamer_.plain_stage_bytes();
        imported_transfer_base_ = streamer_.imported_plain_transfer_bytes();
        hybrid_promotions_base_ = streamer_.hybrid_promotions();
        hybrid_copy_base_ = streamer_.hybrid_copy_bytes();
        blueprint_.reset_metrics();
        cache_->reset_set_policy_metrics();
        adjacent_predictor_.reset_metrics();
        streamer_.reset_host_hint_metrics();
        streamer_.reset_bundle_metrics();
        progressive_metrics_ = {};
        decode_passes_ = 0;
        expert_trace_.set_decode(true);
        if (route_trace_) std::cout << "decode_start " << prompt.size() << '\n';
        const auto decode_started = std::chrono::steady_clock::now();
        std::vector<uint32_t> generated;
        generated.reserve(maximum_new_tokens);
        for (uint32_t item = 0; item < maximum_new_tokens; ++item) {
            if (prediction == tokenizer.eos()) break;
            generated.push_back(prediction);
            std::cout << tokenizer.decode_piece(prediction) << std::flush;
            if (item + 1u == maximum_new_tokens) break;
            const uint32_t position = static_cast<uint32_t>(prompt.size() + item);
            prediction = run_token(prediction, position, record_embedding,
                                   record_pre, record_post, record_final,
                                   record_shared_pre, progressive);
            ++decode_passes_;
        }
        decode_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - decode_started).count();
        return generated;
    }

    Buffer& routing() { return routing_; }
    Buffer& hidden() { return hidden_; }
    Buffer& normalized() { return normalized_; }
    Buffer& q_rank() { return q_rank_; }
    Buffer& query() { return q_; }
    Buffer& kv() { return kv_; }
    Buffer& context() { return context_; }
    Buffer& ffn() { return ffn_; }
    Buffer& logits() { return logits_; }
    Buffer& kv_cache() { return kv_cache_; }
    Buffer& main_targets() { return main_targets_; }
    VkSemaphore expert_transfer_semaphore() const { return transfer_.semaphore(); }
    BatchExpertResolution acquire_batch_experts(uint32_t layer,
                                                 const uint32_t* route_words,
                                                 uint32_t batch) {
        if (layer >= kLayers || !route_words)
            throw std::runtime_error("Invalid batched expert acquisition");
        BatchExpertResolution result;
        std::vector<uint32_t> missing_flat;
        result.slots = cache_->resolve_batch(layer, route_words, batch, missing_flat);
        result.misses = static_cast<uint32_t>(missing_flat.size());
        result.hits = kTopK * batch - result.misses;
        cache_hits_ += result.hits;
        cache_misses_ += result.misses;
        expert_transfer_bytes_ += static_cast<uint64_t>(result.misses) * kExpertRecordBytes;
        if (std::getenv("DSV4_BATCH_DIRECT_STAGING")) {
            result.ready = streamer_.prepare_batch_experts(
                layer, *cache_, route_words, result.slots, missing_flat);
        } else for (uint32_t begin = 0; begin < missing_flat.size(); begin += kTopK) {
            if (result.ready) transfer_.wait(result.ready);
            std::array<uint32_t, kTopK> experts{};
            std::array<uint32_t, kTopK> slots{};
            std::array<uint32_t, kTopK> host_slots{};
            std::vector<uint32_t> missing_ranks, promotions;
            const uint32_t count = std::min<uint32_t>(kTopK,
                static_cast<uint32_t>(missing_flat.size()) - begin);
            for (uint32_t local = 0; local < count; ++local) {
                const uint32_t flat = missing_flat[begin + local];
                const uint32_t token = flat / kTopK, rank = flat % kTopK;
                experts[local] = route_words[token * 64u + rank];
                slots[local] = result.slots[token][rank];
                missing_ranks.push_back(local);
                promotions.push_back(local);
            }
            result.ready = streamer_.prepare_missing_experts(layer, *cache_, experts,
                slots, missing_ranks, promotions, host_slots);
        }
        return result;
    }
    DescriptorRange device_expert_record(uint32_t slot) const {
        return cache_->record(slot);
    }
    DescriptorRange selected_expert_record(uint32_t layer, uint32_t rank) const {
        if (layer >= kLayers || rank >= kTopK)
            throw std::runtime_error("Invalid selected DeepSeek expert record");
        const uint32_t slot = current_slots_[layer][rank];
        if (slot >= cache_->capacity())
            throw std::runtime_error("Invalid selected DeepSeek cache slot");
        return cache_->record(slot);
    }
    bool selected_expert_uses_host(uint32_t layer, uint32_t rank) const {
        return current_host_.at(layer).at(rank);
    }
    DescriptorRange selected_host_expert_record(uint32_t layer, uint32_t rank) const {
        if (!selected_expert_uses_host(layer, rank))
            throw std::runtime_error("Selected DeepSeek expert is not host-resident");
        return streamer_.host_expert_record(
            layer, current_host_slots_.at(layer).at(rank), rank);
    }
    DescriptorRange host_expert_record(uint32_t layer, uint32_t slot,
                                       uint32_t rank) const {
        return streamer_.host_expert_record(layer, slot, rank);
    }
    DescriptorRange host_expert_placeholder(uint32_t layer) const {
        return streamer_.host_expert_placeholder(layer);
    }
    const Buffer& shared_arena(uint32_t layer, uint32_t parity) const {
        return streamer_.shared_buffer(layer, parity);
    }
    const Buffer& global_arena() const { return streamer_.global_buffer(); }
    uint64_t global_file_base() const { return streamer_.global_file_base(); }
    uint64_t shared_file_base(uint32_t layer, uint32_t parity) const {
        return streamer_.shared_file_base(layer, parity);
    }
    bool shared_weights_resident() const { return streamer_.shared_resident(); }
    uint32_t expert_cache_slots() const {
        return cache_->capacity();
    }
    Buffer& token_parameter() { return token_; }
    uint64_t cache_hits() const { return cache_hits_; }
    uint64_t cache_misses() const { return cache_misses_; }
    uint64_t expert_transfer_bytes() const { return expert_transfer_bytes_; }
    uint64_t direct_host_expert_bytes() const { return direct_host_expert_bytes_; }
    double pre_seconds() const { return pre_seconds_; }
    double post_seconds() const { return post_seconds_; }
    double acquire_seconds() const { return acquire_seconds_; }
    double decode_seconds() const { return decode_seconds_; }
    uint64_t decode_passes() const { return decode_passes_; }
    uint64_t host_cache_hits() const {
        return streamer_.host_cache_hits() - host_hits_base_;
    }
    uint64_t host_cache_misses() const {
        return streamer_.host_cache_misses() - host_misses_base_;
    }
    uint64_t batch_direct_disk_bytes() const {
        return streamer_.batch_direct_disk_bytes();
    }
    uint64_t host_cache_admission_rejections() const {
        return streamer_.host_cache_admission_rejections() - host_rejections_base_;
    }
    uint32_t host_cache_slots() const { return streamer_.host_cache_slots(); }
    uint64_t host_budget_bytes() const { return streamer_.host_budget_bytes(); }
    bool budgeted_io() const { return streamer_.budgeted_io(); }
    bool expert_bundles() const { return streamer_.expert_bundles(); }
    WeightStreamer::BundleMetrics bundle_metrics() const {
        return streamer_.bundle_metrics();
    }
    uint64_t plain_stage_bytes() const {
        return streamer_.plain_stage_bytes() - plain_stage_base_;
    }
    uint64_t hybrid_promotions() const {
        return streamer_.hybrid_promotions() - hybrid_promotions_base_;
    }
    uint64_t hybrid_copy_bytes() const {
        return streamer_.hybrid_copy_bytes() - hybrid_copy_base_;
    }
    uint64_t imported_plain_bytes() const { return streamer_.imported_plain_bytes(); }
    uint32_t imported_plain_blocks() const { return streamer_.imported_plain_blocks(); }
    uint64_t imported_plain_transfer_bytes() const {
        return streamer_.imported_plain_transfer_bytes() - imported_transfer_base_;
    }
    bool plain_host_cache() const { return streamer_.plain_host_cache(); }
    bool expert_blueprint_enabled() const { return blueprint_.enabled(); }
    ExpertBlueprint::Metrics expert_blueprint_metrics() const {
        return blueprint_.metrics();
    }
    bool top6_set_policy_enabled() const { return cache_->set_policy_enabled(); }
    GlobalExpertCache::SetPolicyMetrics top6_set_policy_metrics() const {
        return cache_->set_policy_metrics();
    }
    bool adjacent_rank1_hint_enabled() const {
        return adjacent_predictor_.enabled();
    }
    AdjacentRank1Predictor::Metrics adjacent_rank1_metrics() const {
        return adjacent_predictor_.metrics();
    }
    GlobalHostExpertCache::HintMetrics host_hint_metrics() const {
        return streamer_.host_hint_metrics();
    }
    bool progressive_experts_enabled() const { return progressive_experts_; }
    ProgressiveMetrics progressive_metrics() const { return progressive_metrics_; }
    void complete_expert_transfer(uint64_t timeline) {
        blueprint_.complete_timeline(timeline);
    }

private:
    void trace_hidden(const char* stage, uint32_t layer) {
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(kHcMultiplicity) *
                                   kDimension * sizeof(float);
        const uint64_t copied = compute_.submit([&](VkCommandBuffer command) {
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = hidden_.handle;
            barrier.size = bytes;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                     0, nullptr, 1, &barrier, 0, nullptr);
            VkBufferCopy copy{};
            copy.size = bytes;
            vkfn::CmdCopyBuffer(command, hidden_.handle, diagnostic_.handle, 1, &copy);
        });
        compute_.wait(copied);
        invalidate_buffer(runtime_, diagnostic_);
        const auto* values = static_cast<const float*>(diagnostic_.mapped);
        double squares = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < kHcMultiplicity * kDimension; ++i) {
            const float value = values[i];
            if (!std::isfinite(value)) { ++nonfinite; continue; }
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            squares += static_cast<double>(value) * value;
        }
        std::cout << "trace " << stage;
        if (layer != UINT32_MAX) std::cout << ' ' << layer;
        std::cout << " rms=" << std::sqrt(squares /
            static_cast<double>(kHcMultiplicity * kDimension))
                  << " min=" << minimum << " max=" << maximum
                  << " nonfinite=" << nonfinite << "\n";
    }

    const Runtime& runtime_;
    FiniteQueue compute_;
    FiniteQueue transfer_;
    ExpertBlueprint blueprint_;
    AdjacentRank1Predictor adjacent_predictor_;
    WeightStreamer streamer_;
    std::unique_ptr<experiment::FiniteQueueRing<8>> progressive_compute_;
    bool progressive_experts_ = false;
    ProgressiveMetrics progressive_metrics_{};
    Buffer token_{};
    Buffer hidden_{};
    Buffer normalized_{};
    Buffer q_rank_{};
    Buffer q_{};
    Buffer kv_{};
    Buffer context_{};
    Buffer ffn_{};
    Buffer routing_{};
    Buffer logits_{};
    Buffer kv_cache_{};
    Buffer diagnostic_{};
    Buffer main_targets_{};
    bool trace_ = false;
    bool route_trace_ = false;
    ovllm_trace::Writer expert_trace_;
    uint32_t current_position_ = 0;
    std::unique_ptr<GlobalExpertCache> cache_;
    std::array<std::array<bool, kTopK>, kLayers> current_host_{};
    std::array<std::array<uint32_t, kTopK>, kLayers> current_host_slots_{};
    uint64_t cache_hits_ = 0;
    uint64_t cache_misses_ = 0;
    uint64_t expert_transfer_bytes_ = 0;
    uint64_t direct_host_expert_bytes_ = 0;
    double pre_seconds_ = 0.0;
    double post_seconds_ = 0.0;
    double acquire_seconds_ = 0.0;
    double decode_seconds_ = 0.0;
    uint64_t decode_passes_ = 0;
    uint64_t host_hits_base_ = 0;
    uint64_t host_misses_base_ = 0;
    uint64_t host_rejections_base_ = 0;
    uint64_t plain_stage_base_ = 0;
    uint64_t imported_transfer_base_ = 0;
    uint64_t hybrid_promotions_base_ = 0;
    uint64_t hybrid_copy_base_ = 0;
    std::array<std::array<uint32_t, kTopK>, kLayers> current_slots_{};
};

static uint32_t float_word(float value) {
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

static uint32_t divide_up(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static std::array<DescriptorRange, 6> dsv4_bindings(
    const DescriptorRange& dummy,
    std::initializer_list<DescriptorRange> active) {
    if (active.size() > 6) throw std::runtime_error("Too many DeepSeek bindings");
    std::array<DescriptorRange, 6> result{};
    result.fill(dummy);
    size_t index = 0;
    for (const DescriptorRange& range : active) result[index++] = range;
    return result;
}

static void copy_compute_result(VkCommandBuffer command,
                                const Buffer& source, VkDeviceSize source_offset,
                                const Buffer& destination, VkDeviceSize destination_offset,
                                VkDeviceSize bytes) {
    VkBufferMemoryBarrier before{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.buffer = source.handle;
    before.offset = source_offset;
    before.size = bytes;
    vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 1, &before, 0, nullptr);
    VkBufferCopy copy{};
    copy.srcOffset = source_offset;
    copy.dstOffset = destination_offset;
    copy.size = bytes;
    vkfn::CmdCopyBuffer(command, source.handle, destination.handle, 1, &copy);
    VkBufferMemoryBarrier after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    after.buffer = destination.handle;
    after.offset = destination_offset;
    after.size = bytes;
    VkBufferMemoryBarrier source_after{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    source_after.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    source_after.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    source_after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    source_after.buffer = source.handle;
    source_after.offset = source_offset;
    source_after.size = bytes;
    const std::array<VkBufferMemoryBarrier, 2> after_barriers{after, source_after};
    vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr,
                             static_cast<uint32_t>(after_barriers.size()),
                             after_barriers.data(), 0, nullptr);
}

class DeepSeekProgram {
public:
    DeepSeekProgram(const Runtime& runtime, ExecutorScaffold& executor,
                    const SharedIndex& index, const ReadOnlyMapping& shared_file,
                    const std::filesystem::path& shader_directory)
        : runtime_(runtime), executor_(executor), index_(index),
          kernels_(runtime, shader_directory) {
        expert_r1x4_ = std::getenv("DSV4_EXPERT_R1X4") != nullptr;
        expert_r4_ = !expert_r1x4_ && std::getenv("DSV4_EXPERT_R4") != nullptr;
        q4_experts_ = std::getenv("DSV4_Q4_EXPERTS") != nullptr;
        q4_swar_ = q4_experts_ && std::getenv("DSV4_Q4_SWAR") != nullptr;
        q4_expert_batch_bda_ = q4_experts_ &&
            std::getenv("DSV4_Q4_EXPERT_BATCH_BDA") != nullptr;
        hc_fused_ = std::getenv("DSV4_HC_FUSED") != nullptr;
        rms_rope_fused_ = std::getenv("DSV4_RMS_ROPE_FUSED") != nullptr;
        shared_r1x4_ = std::getenv("DSV4_SHARED_R1X4") != nullptr;
        ratio4_slot_.fill(-1);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            if (index_.compression_ratio(layer) == 4)
                ratio4_slot_[layer] = static_cast<int32_t>(ratio4_layers_++);
        }
        hidden_alt_ = create_device_buffer(runtime_,
            static_cast<VkDeviceSize>(kHcMultiplicity) * kDimension * sizeof(float));
        reduced_ = create_device_buffer(runtime_, kDimension * sizeof(float));
        hc_mixes_ = create_device_buffer(runtime_, 24 * sizeof(float));
        hc_split_ = create_device_buffer(runtime_, 24 * sizeof(float));
        quantized_ = create_device_buffer(runtime_, (8192 + 256) * sizeof(uint32_t));
        expert_quantized_ = create_device_buffer(runtime_, (1024 + 128) * sizeof(uint32_t));
        intermediate_quantized_ = create_device_buffer(runtime_,
            static_cast<VkDeviceSize>(kTopK) * (512 + 64) * sizeof(uint32_t));
        o_rank_ = create_device_buffer(runtime_, 8192 * sizeof(float));
        dense_gate_up_ = create_device_buffer(runtime_, 4096 * sizeof(float));
        ffn_accumulator_ = create_device_buffer(runtime_,
            static_cast<VkDeviceSize>(kTopK) * kDimension * sizeof(float));
        router_logits_ = create_device_buffer(runtime_, kExperts * sizeof(float));
        const VkDeviceSize projection_values = static_cast<VkDeviceSize>(ratio4_layers_) *
            128u * 1024u;
        compressor_kv_ = create_device_buffer(runtime_, projection_values * sizeof(float));
        compressor_score_ = create_device_buffer(runtime_, projection_values * sizeof(float));
        compressor_raw_ = create_device_buffer(runtime_,
            static_cast<VkDeviceSize>(ratio4_layers_) * 32u * 512u * sizeof(float));
        indices_ = create_buffer(runtime_, 160 * sizeof(uint32_t));
        ones_ = create_buffer(runtime_, 512 * sizeof(float));
        hc_head_params_ = create_buffer(runtime_, 5 * sizeof(float));
        rope_cos_plain_ = create_buffer(runtime_, kShortContext * 32 * sizeof(float));
        rope_sin_plain_ = create_buffer(runtime_, kShortContext * 32 * sizeof(float));
        rope_cos_compressed_ = create_buffer(runtime_, kShortContext * 32 * sizeof(float));
        rope_sin_compressed_ = create_buffer(runtime_, kShortContext * 32 * sizeof(float));
        argmax_workspace_ = create_device_buffer(runtime_, 256 * 2 * sizeof(uint32_t));
        capture_dspark_ = std::getenv("DSV4_DSPARK_CAPTURE") != nullptr;

        initialize_constants(shared_file);
        build_common_sets();
        build_layer_sets();
        build_global_sets();
    }

    DeepSeekProgram(const DeepSeekProgram&) = delete;
    DeepSeekProgram& operator=(const DeepSeekProgram&) = delete;
    ~DeepSeekProgram() {
        destroy_buffer(runtime_, argmax_workspace_);
        destroy_buffer(runtime_, rope_sin_compressed_);
        destroy_buffer(runtime_, rope_cos_compressed_);
        destroy_buffer(runtime_, rope_sin_plain_);
        destroy_buffer(runtime_, rope_cos_plain_);
        destroy_buffer(runtime_, hc_head_params_);
        destroy_buffer(runtime_, ones_);
        destroy_buffer(runtime_, indices_);
        destroy_buffer(runtime_, compressor_raw_);
        destroy_buffer(runtime_, compressor_score_);
        destroy_buffer(runtime_, compressor_kv_);
        destroy_buffer(runtime_, router_logits_);
        destroy_buffer(runtime_, ffn_accumulator_);
        destroy_buffer(runtime_, dense_gate_up_);
        destroy_buffer(runtime_, o_rank_);
        destroy_buffer(runtime_, intermediate_quantized_);
        destroy_buffer(runtime_, expert_quantized_);
        destroy_buffer(runtime_, quantized_);
        destroy_buffer(runtime_, hc_split_);
        destroy_buffer(runtime_, hc_mixes_);
        destroy_buffer(runtime_, reduced_);
        destroy_buffer(runtime_, hidden_alt_);
    }

    void record_embedding(VkCommandBuffer command, uint32_t, uint32_t position) {
        current_command_ = command;
        current_position_ = position;
        update_indices(position);
        const EmbeddingPush push{kVocabulary, kDimension, kDimension / 4u, 0};
        for (uint32_t hc = 0; hc < kHcMultiplicity; ++hc) {
            EmbeddingPush item = push;
            item.output_offset = hc * kDimension;
            dispatch(global_.embedding, global_embedding_q4_ ?
                     kernels_.pipelines().embedding_q4g64t : kernels_.pipelines().embedding, &item,
                     divide_up(kDimension, 64u));
        }
        compute_barrier(command);
    }

    void record_pre(VkCommandBuffer command, uint32_t layer, uint32_t) {
        if (layer >= kLayers) throw std::runtime_error("Invalid DeepSeek layer record");
        LayerSets& sets = layers_[layer];
        record_hc_pre(command, sets.hc_attn_mix, sets.hc_attn_split,
                      sets.hc_attn_pre, sets.attn_norm,
                      sets.hc_attn_mix_sinkhorn_fused,
                      sets.hc_attn_pre_norm_fused);

        quantize(command, common_.quant_normalized, kDimension, 128u,
                 1024u, 1024u);
        gemv(command, sets.wq_a, 1024u, kDimension, 1024u, 0u);
        gemv(command, sets.wkv, 512u, kDimension, 1024u, 0u);
        if (sets.ratio4) {
            const uint32_t slot = static_cast<uint32_t>(ratio4_slot_[layer]);
            const uint32_t output = (slot * 128u + current_position_) * 1024u;
            gemv(command, sets.compressor_wkv, 1024u, kDimension, 1024u, output);
            gemv(command, sets.compressor_gate, 1024u, kDimension, 1024u, output);
        }
        compute_barrier(command);

        const RmsPush qnorm{1u, 1024u, rms_bits(), 0u};
        dispatch(sets.q_norm, kernels_.pipelines().rmsnorm, &qnorm, 1u);
        compute_barrier(command);
        quantize(command, common_.quant_reduced, 1024u, 128u, 256u, 256u);
        gemv(command, sets.wq_b, kHeads * kHeadDimension, 1024u, 256u, 0u);
        compute_barrier(command);
        if (rms_rope_fused_) {
            const HcFusedPush fused{rms_bits(), current_position_, 0u, 0u};
            dispatch(sets.query_norm_rope_fused,
                     kernels_.pipelines().rmsnorm_rope_fused, &fused, kHeads);
            dispatch(sets.kv_norm_rope_fused,
                     kernels_.pipelines().rmsnorm_rope_fused, &fused, 1u);
            compute_barrier(command);
        } else {
            const RmsPush qhead{kHeads, kHeadDimension, rms_bits(), 0u};
            dispatch(common_.query_norm, kernels_.pipelines().rmsnorm, &qhead, kHeads);
            compute_barrier(command);
            const RopePush qrope{kHeads, kHeadDimension, kRopeDimension, current_position_};
            dispatch(sets.query_rope, kernels_.pipelines().partial_rope, &qrope,
                     divide_up(kHeads * kHeadDimension, 64u));

            const RmsPush kvnorm{1u, kHeadDimension, rms_bits(), 0u};
            dispatch(sets.kv_norm, kernels_.pipelines().rmsnorm, &kvnorm, 1u);
            compute_barrier(command);
            const RopePush kvrope{1u, kHeadDimension, kRopeDimension, current_position_};
            dispatch(sets.kv_rope, kernels_.pipelines().partial_rope, &kvrope,
                     divide_up(kHeadDimension, 64u));
            compute_barrier(command);
        }
        copy_kv(command, layer, current_position_);

        if (sets.ratio4 && (current_position_ & 3u) == 3u) {
            const uint32_t groups = (current_position_ + 1u) / 4u;
            const CompressPush compress{groups, current_position_ + 1u, 512u, 4u};
            dispatch(sets.compress, kernels_.pipelines().compress_ratio4,
                     &compress, groups);
            compute_barrier(command);
            const uint32_t group = groups - 1u;
            const RmsPush compressed_norm{1u, 512u, rms_bits(), 0u};
            dispatch(sets.compressed_norm[group], kernels_.pipelines().rmsnorm,
                     &compressed_norm, 1u);
            compute_barrier(command);
            const RopePush compressed_rope{1u, 512u, kRopeDimension, group * 4u};
            dispatch(sets.kv_rope, kernels_.pipelines().partial_rope,
                     &compressed_rope, divide_up(512u, 64u));
            compute_barrier(command);
            copy_kv(command, layer, kWindow + group);
        }

        const uint32_t compressed = sets.ratio4 ? (current_position_ + 1u) / 4u : 0u;
        const AttentionPush attention{kHeads, kHeadDimension,
                                      current_position_ + 1u + compressed, 0u};
        dispatch(sets.attention, kernels_.pipelines().attention_short,
                 &attention, kHeads);
        compute_barrier(command);
        RopePush inverse{kHeads, kHeadDimension, kRopeDimension,
                         current_position_ | 0x80000000u};
        dispatch(sets.query_rope, kernels_.pipelines().partial_rope,
                 &inverse, divide_up(kHeads * kHeadDimension, 64u));
        compute_barrier(command);
        quantize(command, common_.quant_query, kHeads * kHeadDimension,
                 128u, 8192u, 8192u);
        const GroupedGemvPush wo_a{8u, 1024u, 4096u, 8192u};
        const bool wo_a_q4 = q4_sets_.count(sets.wo_a) != 0;
        const VkPipeline wo_a_pipeline = wo_a_q4 ?
            (shared_r1x4_ ? kernels_.pipelines().q4g64t_grouped_gemv_r1x4 :
                            kernels_.pipelines().q4g64t_grouped_gemv) :
            kernels_.pipelines().q8_grouped_gemv;
        dispatch(sets.wo_a, wo_a_pipeline, &wo_a,
                 divide_up(1024u, (wo_a_q4 && !shared_r1x4_) ? 8u : 4u), 8u);
        compute_barrier(command);
        quantize(command, common_.quant_o_rank, 8192u, 128u, 2048u, 2048u);
        gemv(command, sets.wo_b, kDimension, 8192u, 2048u, 0u);
        compute_barrier(command);

        const HcApplyPush hc_apply{kDimension, kHcMultiplicity, 24u, 0u};
        dispatch(sets.hc_attn_post, kernels_.pipelines().hc_post,
                 &hc_apply, divide_up(kHcMultiplicity * kDimension, 64u));
        compute_barrier(command);
        record_hc_pre(command, sets.hc_ffn_mix, sets.hc_ffn_split,
                      sets.hc_ffn_pre, sets.ffn_norm,
                      sets.hc_ffn_mix_sinkhorn_fused,
                      sets.hc_ffn_pre_norm_fused);

        quantize(command, common_.quant_normalized, kDimension, 128u,
                 1024u, 1024u);
        gemv(command, sets.router_gemv, kExperts, kDimension, 1024u, 0u);
        compute_barrier(command);
        RouterPush router{kExperts, kTopK, float_word(index_.header().route_scale),
                          current_token_flags(layer)};
        dispatch(sets.router, kernels_.pipelines().router_top6, &router, 1u);
        compute_barrier(command);
    }

    void record_routed_experts(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_.at(layer);
        bool batch_q4_experts = q4_expert_batch_bda_;
        std::array<DescriptorRange, kTopK> expert_records{};
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const bool host = executor_.selected_expert_uses_host(layer, rank);
            expert_records[rank] = host ?
                executor_.selected_host_expert_record(layer, rank) :
                executor_.selected_expert_record(layer, rank);
            batch_q4_experts = batch_q4_experts && !host;
        }
        if (batch_q4_experts) {
            auto* route_words = static_cast<uint32_t*>(executor_.routing().mapped);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const DescriptorRange& record = expert_records[rank];
                VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                info.buffer = record.buffer;
                const uint64_t address =
                    vkfn::GetBufferDeviceAddress(runtime_.device, &info) + record.offset;
                if (address == 0u)
                    throw std::runtime_error("Selected Q4 expert has no Vulkan device address");
                std::memcpy(route_words + 32u + rank * 2u, &address, sizeof(address));
            }
            flush_buffer(runtime_, executor_.routing());
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                kernels_.update_binding(sets.expert_gate_host[rank], 1u,
                                        expert_records[rank]);
                kernels_.update_binding(sets.expert_down_host[rank], 1u,
                                        expert_records[rank]);
            }
        }
        quantize(command, common_.quant_normalized_expert, kDimension, 32u,
                 1024u, 1024u);
        const uint32_t limit = float_word(index_.header().swiglu_limit);
        const VkPipeline gate_pipeline = q4_experts_ ?
            (q4_swar_ ? kernels_.pipelines().expert_gate_up_q4g64t_swar :
                        kernels_.pipelines().expert_gate_up_q4g64t) :
            (expert_r1x4_ ? kernels_.pipelines().expert_gate_up_fp4_r1x4 :
            (expert_r4_ ? kernels_.pipelines().expert_gate_up_fp4_r4 :
                          kernels_.pipelines().expert_gate_up_fp4));
        const VkPipeline down_pipeline = q4_experts_ ?
            (q4_swar_ ? kernels_.pipelines().expert_down_q4g64t_swar :
                        kernels_.pipelines().expert_down_q4g64t) :
            (expert_r1x4_ ? kernels_.pipelines().expert_down_fp4_r1x4 :
            (expert_r4_ ? kernels_.pipelines().expert_down_fp4_r4 :
                          kernels_.pipelines().expert_down_fp4));
        const uint32_t expert_rows_per_group =
            (q4_experts_ || expert_r1x4_ || expert_r4_) ? 4u : 8u;
        if (batch_q4_experts) {
            const ExpertPush gate{0u, 1024u, limit, 0u};
            dispatch(common_.expert_gate_batch,
                kernels_.pipelines().expert_gate_up_q4g64t_bda_batch,
                &gate, divide_up(kMoeDimension, 4u), kTopK);
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const ExpertPush gate{rank, 1024u, limit, 0u};
                dispatch(sets.expert_gate_host[rank], gate_pipeline,
                          &gate, divide_up(kMoeDimension, expert_rows_per_group));
            }
        }
        compute_barrier(command);
        // Quantize all six independent intermediates into disjoint slices,
        // then issue one dependency before the six down projections.
        if (batch_q4_experts) {
            const QuantizePush quant{kMoeDimension, 32u, 512u, 512u};
            dispatch(common_.quant_expert_intermediate_batch,
                kernels_.pipelines().quantize_q8_strided_batch, &quant,
                divide_up(kMoeDimension, 32u), kTopK);
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const QuantizePush quant{kMoeDimension, 32u, 512u, 512u};
                dispatch(common_.quant_expert_intermediate[rank],
                         kernels_.pipelines().quantize_q8, &quant,
                         divide_up(kMoeDimension, 32u));
            }
        }
        compute_barrier(command);
        if (batch_q4_experts) {
            const ExpertPush down{0u, 512u, 0u, 0u};
            dispatch(common_.expert_down_batch,
                kernels_.pipelines().expert_down_q4g64t_bda_batch,
                &down, divide_up(kDimension, 4u), kTopK);
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const ExpertPush down{rank, 512u, 0u, 0u};
                dispatch(sets.expert_down_host[rank], down_pipeline,
                          &down, divide_up(kDimension, expert_rows_per_group));
            }
        }
        compute_barrier(command);
        const QuantizePush reduce{kDimension, kTopK, 0u, 0u};
        dispatch(common_.reduce_experts, kernels_.pipelines().reduce_experts,
                 &reduce, divide_up(kDimension, 64u));
        compute_barrier(command);
    }

    // Progressive V1 keeps the exact scalar-Q4 kernels and record ABI.  Only
    // command construction changes: one common activation quantization, six
    // independent finite rank submissions, then the original deterministic
    // rank-order reduction.  The all-ready path continues to use the faster
    // six-rank BDA kernels through record_routed_experts above.
    void record_progressive_expert_prefix(VkCommandBuffer command,
                                          uint32_t layer) {
        if (!q4_experts_)
            throw std::runtime_error(
                "DSV4_PROGRESSIVE_EXPERTS currently requires Q4 experts");
        LayerSets& sets = layers_.at(layer);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const DescriptorRange record = executor_.selected_expert_record(layer, rank);
            kernels_.update_binding(sets.expert_gate_host[rank], 1u, record);
            kernels_.update_binding(sets.expert_down_host[rank], 1u, record);
        }
        quantize(command, common_.quant_normalized_expert, kDimension, 32u,
                 1024u, 1024u);
    }

    void record_progressive_expert_rank(VkCommandBuffer command,
                                        uint32_t layer, uint32_t rank) {
        if (!q4_experts_ || rank >= kTopK)
            throw std::runtime_error("Invalid progressive Q4 expert rank");
        current_command_ = command;
        LayerSets& sets = layers_.at(layer);
        const VkPipeline gate_pipeline = q4_swar_ ?
            kernels_.pipelines().expert_gate_up_q4g64t_swar :
            kernels_.pipelines().expert_gate_up_q4g64t;
        const VkPipeline down_pipeline = q4_swar_ ?
            kernels_.pipelines().expert_down_q4g64t_swar :
            kernels_.pipelines().expert_down_q4g64t;
        const uint32_t limit = float_word(index_.header().swiglu_limit);
        const ExpertPush gate{rank, 1024u, limit, 0u};
        dispatch(sets.expert_gate_host[rank], gate_pipeline,
                 &gate, divide_up(kMoeDimension, 4u));
        compute_barrier(command);
        const QuantizePush quant{kMoeDimension, 32u, 512u, 512u};
        dispatch(common_.quant_expert_intermediate[rank],
                 kernels_.pipelines().quantize_q8, &quant,
                 divide_up(kMoeDimension, 32u));
        compute_barrier(command);
        const ExpertPush down{rank, 512u, 0u, 0u};
        dispatch(sets.expert_down_host[rank], down_pipeline,
                 &down, divide_up(kDimension, 4u));
        compute_barrier(command);
    }

    void record_progressive_expert_finish(VkCommandBuffer command,
                                          uint32_t layer, uint32_t parity) {
        current_command_ = command;
        const QuantizePush reduce{kDimension, kTopK, 0u, 0u};
        dispatch(common_.reduce_experts, kernels_.pipelines().reduce_experts,
                 &reduce, divide_up(kDimension, 64u));
        compute_barrier(command);
        record_shared_pre(command, layer, parity);
        record_shared_finish(command, layer);
    }

    void record_shared_pre(VkCommandBuffer command, uint32_t layer, uint32_t) {
        LayerSets& sets = layers_.at(layer);
        const uint32_t limit = float_word(index_.header().swiglu_limit);
        gemv(command, sets.shared_w1, kMoeDimension, kDimension, 1024u, 0u);
        gemv(command, sets.shared_w3, kMoeDimension, kDimension, 1024u,
             kMoeDimension);
        compute_barrier(command);
        const SwigluPush swiglu{kMoeDimension, limit, 0u, kMoeDimension};
        dispatch(common_.shared_swiglu, kernels_.pipelines().swiglu,
                 &swiglu, divide_up(kMoeDimension, 64u));
        compute_barrier(command);
        quantize(command, common_.quant_shared_intermediate, kMoeDimension,
                 128u, 512u, 512u);
    }

    void record_shared_finish(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_.at(layer);
        gemv_residual(command, sets.shared_w2, kDimension, kMoeDimension,
                      512u, 0u);
        compute_barrier(command);
        const HcApplyPush hc_apply{kDimension, kHcMultiplicity, 24u, 0u};
        dispatch(sets.hc_ffn_post, kernels_.pipelines().hc_post,
                  &hc_apply, divide_up(kHcMultiplicity * kDimension, 64u));
        compute_barrier(command);
        if (capture_dspark_ && layer >= 40u && layer <= 42u) {
            const VkDeviceSize bytes = static_cast<VkDeviceSize>(kHcMultiplicity) *
                kDimension * sizeof(float);
            copy_compute_result(command, executor_.hidden(), 0,
                executor_.main_targets(), static_cast<VkDeviceSize>(layer - 40u) * bytes,
                bytes);
        }
    }

    void record_post(VkCommandBuffer command, uint32_t layer, uint32_t parity) {
        record_routed_experts(command, layer);
        record_shared_pre(command, layer, parity);
        record_shared_finish(command, layer);
    }

    void record_post_after_shared(VkCommandBuffer command, uint32_t layer,
                                  uint32_t) {
        record_routed_experts(command, layer);
        record_shared_finish(command, layer);
    }

    void record_final(VkCommandBuffer command, uint32_t) {
        current_command_ = command;
        const HcApplyPush head{kDimension, kHcMultiplicity, rms_bits(), 0u};
        dispatch(global_.hc_head, kernels_.pipelines().hc_head,
                 &head, 1u);
        compute_barrier(command);
        const RmsPush norm{1u, kDimension, rms_bits(), 0u};
        dispatch(global_.norm, kernels_.pipelines().rmsnorm, &norm, 1u);
        compute_barrier(command);
        quantize(command, common_.quant_normalized, kDimension, 128u,
                 1024u, 1024u);
        gemv(command, global_.head, kVocabulary, kDimension, 1024u, 0u);
        compute_barrier(command);
        const uint32_t groups = divide_up(kVocabulary, 64u * 8u);
        const ArgmaxPush first{kVocabulary, groups, 0u, 0u};
        dispatch(global_.argmax, kernels_.pipelines().greedy_argmax,
                 &first, groups);
        compute_barrier(command);
        const ArgmaxPush second{kVocabulary, groups, 1u, 0u};
        dispatch(global_.argmax, kernels_.pipelines().greedy_argmax,
                 &second, 1u);
    }

    Buffer& compressor_kv_history() { return compressor_kv_; }
    Buffer& compressor_score_history() { return compressor_score_; }
    Buffer& compressor_raw_history() { return compressor_raw_; }

private:
    struct CommonSets {
        VkDescriptorSet quant_normalized = VK_NULL_HANDLE;
        VkDescriptorSet quant_reduced = VK_NULL_HANDLE;
        VkDescriptorSet quant_query = VK_NULL_HANDLE;
        VkDescriptorSet quant_o_rank = VK_NULL_HANDLE;
        VkDescriptorSet quant_normalized_expert = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kTopK> quant_expert_intermediate{};
        VkDescriptorSet quant_expert_intermediate_batch = VK_NULL_HANDLE;
        VkDescriptorSet expert_gate_batch = VK_NULL_HANDLE;
        VkDescriptorSet expert_down_batch = VK_NULL_HANDLE;
        VkDescriptorSet quant_shared_intermediate = VK_NULL_HANDLE;
        VkDescriptorSet query_norm = VK_NULL_HANDLE;
        VkDescriptorSet shared_swiglu = VK_NULL_HANDLE;
        VkDescriptorSet reduce_experts = VK_NULL_HANDLE;
    };

    struct LayerSets {
        bool ratio4 = false;
        VkDescriptorSet hc_attn_mix = VK_NULL_HANDLE;
        VkDescriptorSet hc_attn_split = VK_NULL_HANDLE;
        VkDescriptorSet hc_attn_pre = VK_NULL_HANDLE;
        VkDescriptorSet attn_norm = VK_NULL_HANDLE;
        VkDescriptorSet hc_attn_mix_sinkhorn_fused = VK_NULL_HANDLE;
        VkDescriptorSet hc_attn_pre_norm_fused = VK_NULL_HANDLE;
        VkDescriptorSet wq_a = VK_NULL_HANDLE;
        VkDescriptorSet q_norm = VK_NULL_HANDLE;
        VkDescriptorSet wq_b = VK_NULL_HANDLE;
        VkDescriptorSet query_rope = VK_NULL_HANDLE;
        VkDescriptorSet query_norm_rope_fused = VK_NULL_HANDLE;
        VkDescriptorSet wkv = VK_NULL_HANDLE;
        VkDescriptorSet kv_norm = VK_NULL_HANDLE;
        VkDescriptorSet kv_rope = VK_NULL_HANDLE;
        VkDescriptorSet kv_norm_rope_fused = VK_NULL_HANDLE;
        VkDescriptorSet attention = VK_NULL_HANDLE;
        VkDescriptorSet wo_a = VK_NULL_HANDLE;
        VkDescriptorSet wo_b = VK_NULL_HANDLE;
        VkDescriptorSet compressor_wkv = VK_NULL_HANDLE;
        VkDescriptorSet compressor_gate = VK_NULL_HANDLE;
        VkDescriptorSet compress = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, 32> compressed_norm{};
        VkDescriptorSet hc_attn_post = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_mix = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_split = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_pre = VK_NULL_HANDLE;
        VkDescriptorSet ffn_norm = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_mix_sinkhorn_fused = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_pre_norm_fused = VK_NULL_HANDLE;
        VkDescriptorSet router_gemv = VK_NULL_HANDLE;
        VkDescriptorSet router = VK_NULL_HANDLE;
        std::array<VkDescriptorSet, kTopK> expert_gate_host{};
        std::array<VkDescriptorSet, kTopK> expert_down_host{};
        VkDescriptorSet shared_w1 = VK_NULL_HANDLE;
        VkDescriptorSet shared_w3 = VK_NULL_HANDLE;
        VkDescriptorSet shared_w2 = VK_NULL_HANDLE;
        VkDescriptorSet hc_ffn_post = VK_NULL_HANDLE;
    };

    struct GlobalSets {
        VkDescriptorSet embedding = VK_NULL_HANDLE;
        VkDescriptorSet hc_head = VK_NULL_HANDLE;
        VkDescriptorSet norm = VK_NULL_HANDLE;
        VkDescriptorSet head = VK_NULL_HANDLE;
        VkDescriptorSet argmax = VK_NULL_HANDLE;
    };

    VkDescriptorSet make_set(std::initializer_list<DescriptorRange> active) {
        return kernels_.set(dsv4_bindings(kernels_.dummy(), active));
    }

    VkDescriptorSet q8_set(const DescriptorRange& activation,
                           const TensorView& matrix, const Buffer& arena,
                           uint64_t file_base, const DescriptorRange& output,
                           const DescriptorRange* residual = nullptr) {
        std::array<DescriptorRange, 6> ranges{};
        ranges.fill(kernels_.dummy());
        ranges[0] = activation;
        ranges[1] = tensor_data_range(arena, file_base, matrix);
        ranges[2] = tensor_auxiliary_range(arena, file_base, matrix);
        ranges[3] = output;
        if (residual) ranges[4] = *residual;
        VkDescriptorSet set = kernels_.set(ranges);
        if (matrix.format == TensorFormat::q4g64t) q4_sets_.insert(set);
        return set;
    }

    void initialize_constants(const ReadOnlyMapping& shared_file) {
        auto* ones = static_cast<float*>(ones_.mapped);
        std::fill(ones, ones + 512, 1.0f);
        flush_buffer(runtime_, ones_);

        const TensorView& scale = index_.require("hc_head_scale");
        const TensorView& base = index_.require("hc_head_base");
        if (scale.format != TensorFormat::f32 || scale.bytes != sizeof(float) ||
            base.format != TensorFormat::f32 || base.bytes != 4 * sizeof(float))
            throw std::runtime_error("Invalid DeepSeek HC-head parameters");
        auto* params = static_cast<float*>(hc_head_params_.mapped);
        std::memcpy(params, shared_file.data() + scale.offset, sizeof(float));
        std::memcpy(params + 1, shared_file.data() + base.offset, 4 * sizeof(float));
        flush_buffer(runtime_, hc_head_params_);

        fill_rope(rope_cos_plain_, rope_sin_plain_, index_.header().rope_theta,
                  false);
        fill_rope(rope_cos_compressed_, rope_sin_compressed_,
                  index_.header().compress_rope_theta, true);
    }

    void fill_rope(Buffer& cosine, Buffer& sine, float base, bool yarn) {
        auto* cos_values = static_cast<float*>(cosine.mapped);
        auto* sin_values = static_cast<float*>(sine.mapped);
        const float pi = 3.14159265358979323846f;
        float low = 0.0f;
        float high = 0.0f;
        if (yarn) {
            const float original = index_.header().original_max_position;
            low = std::floor(kRopeDimension * std::log(original /
                (index_.header().beta_fast * 2.0f * pi)) /
                (2.0f * std::log(base)));
            high = std::ceil(kRopeDimension * std::log(original /
                (index_.header().beta_slow * 2.0f * pi)) /
                (2.0f * std::log(base)));
            low = std::max(low, 0.0f);
            high = std::min(high, static_cast<float>(kRopeDimension - 1));
            if (low == high) high += 0.001f;
        }
        for (uint32_t frequency = 0; frequency < 32u; ++frequency) {
            float value = 1.0f / std::pow(base,
                static_cast<float>(frequency * 2u) / kRopeDimension);
            if (yarn) {
                const float ramp = std::clamp((static_cast<float>(frequency) - low) /
                                              (high - low), 0.0f, 1.0f);
                const float smooth = 1.0f - ramp;
                value = value / index_.header().yarn_factor * (1.0f - smooth) +
                        value * smooth;
            }
            for (uint32_t position = 0; position < kShortContext; ++position) {
                const float angle = static_cast<float>(position) * value;
                cos_values[position * 32u + frequency] = std::cos(angle);
                sin_values[position * 32u + frequency] = std::sin(angle);
            }
        }
        flush_buffer(runtime_, cosine);
        flush_buffer(runtime_, sine);
    }

    void build_common_sets() {
        common_.quant_normalized = make_set({whole(executor_.normalized()),
                                             whole(quantized_)});
        common_.quant_reduced = make_set({whole(reduced_), whole(quantized_)});
        common_.quant_query = make_set({whole(executor_.query()), whole(quantized_)});
        common_.quant_o_rank = make_set({whole(o_rank_), whole(quantized_)});
        common_.quant_normalized_expert = make_set({whole(executor_.normalized()),
                                                    whole(expert_quantized_)});
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            common_.quant_expert_intermediate[rank] = make_set({
                arena_range(executor_.ffn(),
                    static_cast<VkDeviceSize>(rank) * kMoeDimension * sizeof(float),
                    kMoeDimension * sizeof(float)),
                arena_range(intermediate_quantized_,
                    static_cast<VkDeviceSize>(rank) * (512u + 64u) * sizeof(uint32_t),
                    (512u + 64u) * sizeof(uint32_t))});
        }
        common_.quant_expert_intermediate_batch = make_set({
            arena_range(executor_.ffn(), 0,
                static_cast<VkDeviceSize>(kTopK) * kMoeDimension * sizeof(float)),
            whole(intermediate_quantized_)});
        const DescriptorRange expert_addresses = arena_range(executor_.routing(),
            32u * sizeof(uint32_t), kTopK * sizeof(uint64_t));
        common_.expert_gate_batch = make_set({whole(expert_quantized_),
            expert_addresses, whole(executor_.routing()), whole(executor_.ffn())});
        common_.expert_down_batch = make_set({whole(intermediate_quantized_),
            expert_addresses, whole(executor_.routing()), whole(ffn_accumulator_)});
        common_.quant_shared_intermediate = make_set({
            arena_range(executor_.ffn(),
                static_cast<VkDeviceSize>(kTopK) * kMoeDimension * sizeof(float),
                kMoeDimension * sizeof(float)), whole(quantized_)});
        common_.query_norm = make_set({whole(executor_.query()), whole(ones_),
                                       whole(executor_.context())});
        common_.shared_swiglu = make_set({whole(dense_gate_up_), whole(dense_gate_up_),
                                          kernels_.dummy(),
                                          arena_range(executor_.ffn(),
                                            static_cast<VkDeviceSize>(kTopK) *
                                                kMoeDimension * sizeof(float),
                                             kMoeDimension * sizeof(float))});
        common_.reduce_experts = make_set({whole(ffn_accumulator_)});
    }

    void build_layer_sets() {
        layers_.resize(kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            LayerSets& sets = layers_[layer];
            sets.ratio4 = index_.compression_ratio(layer) == 4;
            const uint32_t parity = layer & 1u;
            const Buffer& arena = executor_.shared_arena(layer, parity);
            const uint64_t base = executor_.shared_file_base(layer, parity);
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            const auto tensor = [&](const char* suffix) -> const TensorView& {
                return index_.require(prefix + suffix);
            };
            const auto data = [&](const TensorView& item) {
                return tensor_data_range(arena, base, item);
            };
            const auto q8 = [&](const char* suffix, uint32_t rows, uint32_t inner) ->
                    const TensorView& {
                const TensorView& item = tensor(suffix);
                require_linear_matrix(item, rows, inner, suffix);
                return item;
            };

            sets.hc_attn_mix = make_set({whole(executor_.hidden()),
                data(tensor("hc_attn_fn")), whole(hc_mixes_)});
            sets.hc_attn_split = make_set({whole(hc_mixes_),
                data(tensor("hc_attn_scale")), data(tensor("hc_attn_base")),
                whole(hc_split_)});
            sets.hc_attn_pre = make_set({whole(executor_.hidden()), whole(hc_split_),
                                         whole(reduced_)});
            sets.attn_norm = make_set({whole(reduced_), data(tensor("attn_norm.weight")),
                                       whole(executor_.normalized())});
            sets.hc_attn_mix_sinkhorn_fused = make_set({whole(executor_.hidden()),
                data(tensor("hc_attn_fn")), data(tensor("hc_attn_scale")),
                data(tensor("hc_attn_base")), whole(hc_split_)});
            sets.hc_attn_pre_norm_fused = make_set({whole(executor_.hidden()),
                whole(hc_split_), data(tensor("attn_norm.weight")),
                whole(executor_.normalized())});

            const TensorView& wq_a = q8("attn.wq_a.weight", 1024u, kDimension);
            sets.wq_a = q8_set(whole(quantized_), wq_a, arena, base,
                               whole(executor_.q_rank()));
            sets.q_norm = make_set({whole(executor_.q_rank()),
                data(tensor("attn.q_norm.weight")), whole(reduced_)});
            const TensorView& wq_b = q8("attn.wq_b.weight",
                                        kHeads * kHeadDimension, 1024u);
            sets.wq_b = q8_set(whole(quantized_), wq_b, arena, base,
                               whole(executor_.query()));

            const bool compressed_rope = index_.compression_ratio(layer) != 0;
            Buffer& rope_cos = compressed_rope ? rope_cos_compressed_ : rope_cos_plain_;
            Buffer& rope_sin = compressed_rope ? rope_sin_compressed_ : rope_sin_plain_;
            sets.query_rope = make_set({whole(executor_.context()), whole(rope_cos),
                                        whole(rope_sin), whole(executor_.query())});
            sets.query_norm_rope_fused = make_set({whole(executor_.query()),
                whole(ones_), whole(rope_cos), whole(rope_sin),
                whole(executor_.query())});
            const TensorView& wkv = q8("attn.wkv.weight", 512u, kDimension);
            sets.wkv = q8_set(whole(quantized_), wkv, arena, base,
                              whole(executor_.kv()));
            sets.kv_norm = make_set({whole(executor_.kv()),
                data(tensor("attn.kv_norm.weight")), whole(reduced_)});
            sets.kv_rope = make_set({whole(reduced_), whole(rope_cos), whole(rope_sin),
                                     whole(executor_.kv())});
            sets.kv_norm_rope_fused = make_set({whole(executor_.kv()),
                data(tensor("attn.kv_norm.weight")), whole(rope_cos), whole(rope_sin),
                whole(executor_.kv())});
            const VkDeviceSize kv_layer_bytes =
                static_cast<VkDeviceSize>(kWindow + 32u) * kHeadDimension * sizeof(float);
            sets.attention = make_set({whole(executor_.query()),
                arena_range(executor_.kv_cache(),
                    static_cast<VkDeviceSize>(layer) * kv_layer_bytes, kv_layer_bytes),
                whole(indices_), data(tensor("attn.attn_sink")),
                whole(executor_.context())});
            const TensorView& wo_a = q8("attn.wo_a.weight", 8192u, 4096u);
            sets.wo_a = q8_set(whole(quantized_), wo_a, arena, base, whole(o_rank_));
            const TensorView& wo_b = q8("attn.wo_b.weight", kDimension, 8192u);
            sets.wo_b = q8_set(whole(quantized_), wo_b, arena, base, whole(reduced_));

            if (sets.ratio4) {
                const uint32_t ratio_slot = static_cast<uint32_t>(ratio4_slot_[layer]);
                const VkDeviceSize history_bytes = 128ull * 1024ull * sizeof(float);
                const VkDeviceSize history_offset =
                    static_cast<VkDeviceSize>(ratio_slot) * history_bytes;
                const TensorView& compressor_wkv = q8("attn.compressor.wkv.weight",
                                                       1024u, kDimension);
                sets.compressor_wkv = q8_set(whole(quantized_), compressor_wkv,
                    arena, base, whole(compressor_kv_));
                const TensorView& compressor_gate = q8("attn.compressor.wgate.weight",
                                                        1024u, kDimension);
                sets.compressor_gate = q8_set(whole(quantized_), compressor_gate,
                    arena, base, whole(compressor_score_));
                const VkDeviceSize raw_bytes = 32ull * 512ull * sizeof(float);
                const VkDeviceSize raw_offset =
                    static_cast<VkDeviceSize>(ratio_slot) * raw_bytes;
                sets.compress = make_set({
                    arena_range(compressor_kv_, history_offset, history_bytes),
                    arena_range(compressor_score_, history_offset, history_bytes),
                    data(tensor("attn.compressor.ape")),
                    arena_range(compressor_raw_, raw_offset, raw_bytes)});
                for (uint32_t group = 0; group < 32u; ++group) {
                    sets.compressed_norm[group] = make_set({
                        arena_range(compressor_raw_, raw_offset +
                            static_cast<VkDeviceSize>(group) * 512u * sizeof(float),
                            512u * sizeof(float)),
                        data(tensor("attn.compressor.norm.weight")), whole(reduced_)});
                }
            }

            sets.hc_attn_post = make_set({whole(reduced_), whole(executor_.hidden()),
                                          whole(hc_split_), whole(hidden_alt_)});
            sets.hc_ffn_mix = make_set({whole(hidden_alt_), data(tensor("hc_ffn_fn")),
                                        whole(hc_mixes_)});
            sets.hc_ffn_split = make_set({whole(hc_mixes_),
                data(tensor("hc_ffn_scale")), data(tensor("hc_ffn_base")),
                whole(hc_split_)});
            sets.hc_ffn_pre = make_set({whole(hidden_alt_), whole(hc_split_),
                                        whole(reduced_)});
            sets.ffn_norm = make_set({whole(reduced_), data(tensor("ffn_norm.weight")),
                                      whole(executor_.normalized())});
            sets.hc_ffn_mix_sinkhorn_fused = make_set({whole(hidden_alt_),
                data(tensor("hc_ffn_fn")), data(tensor("hc_ffn_scale")),
                data(tensor("hc_ffn_base")), whole(hc_split_)});
            sets.hc_ffn_pre_norm_fused = make_set({whole(hidden_alt_),
                whole(hc_split_), data(tensor("ffn_norm.weight")),
                whole(executor_.normalized())});
            const TensorView& gate = q8("ffn.gate.weight", kExperts, kDimension);
            sets.router_gemv = q8_set(whole(quantized_), gate, arena, base,
                                      whole(router_logits_));
            DescriptorRange bias = kernels_.dummy();
            DescriptorRange tid2eid = kernels_.dummy();
            if (layer < kHashLayers) tid2eid = data(tensor("ffn.gate.tid2eid"));
            else bias = data(tensor("ffn.gate.bias"));
            sets.router = make_set({whole(router_logits_), bias, tid2eid,
                                    whole(executor_.routing())});

            const DescriptorRange host_placeholder =
                executor_.host_expert_placeholder(layer);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                sets.expert_gate_host[rank] = make_set({whole(expert_quantized_),
                    host_placeholder, whole(executor_.routing()), whole(executor_.ffn())});
                sets.expert_down_host[rank] = make_set({
                    arena_range(intermediate_quantized_,
                        static_cast<VkDeviceSize>(rank) * (512u + 64u) * sizeof(uint32_t),
                        (512u + 64u) * sizeof(uint32_t)),
                    host_placeholder, whole(executor_.routing()), whole(ffn_accumulator_)});
            }
            const TensorView& shared_w1 = q8("ffn.shared_experts.w1.weight",
                                             kMoeDimension, kDimension);
            const TensorView& shared_w3 = q8("ffn.shared_experts.w3.weight",
                                             kMoeDimension, kDimension);
            const TensorView& shared_w2 = q8("ffn.shared_experts.w2.weight",
                                             kDimension, kMoeDimension);
            sets.shared_w1 = q8_set(whole(quantized_), shared_w1, arena, base,
                                    whole(dense_gate_up_));
            sets.shared_w3 = q8_set(whole(quantized_), shared_w3, arena, base,
                                    whole(dense_gate_up_));
            const DescriptorRange residual = whole(ffn_accumulator_);
            sets.shared_w2 = q8_set(whole(quantized_), shared_w2, arena, base,
                                    whole(reduced_), &residual);
            sets.hc_ffn_post = make_set({whole(reduced_), whole(hidden_alt_),
                                         whole(hc_split_), whole(executor_.hidden())});
        }
    }

    void build_global_sets() {
        const Buffer& arena = executor_.global_arena();
        const uint64_t base = executor_.global_file_base();
        const TensorView& embedding = index_.require("embed.weight");
        require_linear_matrix(embedding, kVocabulary, kDimension, "embedding");
        global_embedding_q4_ = embedding.format == TensorFormat::q4g64t;
        global_.embedding = make_set({tensor_data_range(arena, base, embedding),
            tensor_auxiliary_range(arena, base, embedding),
            whole(executor_.token_parameter()), whole(executor_.hidden())});
        const TensorView& hc_fn = index_.require("hc_head_fn");
        global_.hc_head = make_set({whole(executor_.hidden()),
            tensor_data_range(arena, base, hc_fn), whole(hc_head_params_), whole(reduced_)});
        global_.norm = make_set({whole(reduced_),
            tensor_data_range(arena, base, index_.require("norm.weight")),
            whole(executor_.normalized())});
        const TensorView& head = index_.require("head.weight");
        require_linear_matrix(head, kVocabulary, kDimension, "language head");
        global_.head = q8_set(whole(quantized_), head, arena, base,
                              whole(executor_.logits()));
        global_.argmax = make_set({whole(executor_.logits()),
            whole(executor_.token_parameter()), whole(argmax_workspace_)});
    }

    void update_indices(uint32_t position) {
        auto* values = static_cast<uint32_t*>(indices_.mapped);
        uint32_t count = 0;
        for (uint32_t index = 0; index <= position; ++index)
            values[count++] = index;
        const uint32_t compressed = (position + 1u) / 4u;
        for (uint32_t index = 0; index < compressed; ++index)
            values[count++] = kWindow + index;
        while (count < 160u) values[count++] = UINT32_MAX;
        flush_buffer(runtime_, indices_);
    }

    uint32_t current_token_flags(uint32_t layer) const {
        const auto* token = static_cast<const uint32_t*>(executor_.token_parameter().mapped);
        return *token | (layer < kHashLayers ? 0x80000000u : 0u);
    }

    uint32_t rms_bits() const { return float_word(index_.header().rms_epsilon); }

    void dispatch(VkDescriptorSet set, VkPipeline pipeline, const void* push,
                  uint32_t groups_x, uint32_t groups_y = 1u) {
        dispatch_dsv4(current_command_, kernels_.resources(), pipeline, set, push,
                      groups_x, groups_y);
    }

    void quantize(VkCommandBuffer command, VkDescriptorSet set, uint32_t count,
                  uint32_t group_size, uint32_t packed_words, uint32_t scale_u32) {
        current_command_ = command;
        const QuantizePush push{count, group_size, packed_words, scale_u32};
        dispatch(set, kernels_.pipelines().quantize_q8, &push,
                 divide_up(count, group_size));
        compute_barrier(command);
    }

    void gemv(VkCommandBuffer command, VkDescriptorSet set, uint32_t rows,
              uint32_t inner, uint32_t activation_scale, uint32_t output_offset) {
        current_command_ = command;
        const GemvPush push{rows, inner, activation_scale, output_offset};
        const bool q4 = q4_sets_.count(set) != 0;
        const VkPipeline pipeline = q4 ?
            (shared_r1x4_ ? kernels_.pipelines().q4g64t_gemv_r1x4 :
                            kernels_.pipelines().q4g64t_gemv) :
            kernels_.pipelines().q8_gemv;
        dispatch(set, pipeline, &push,
                 divide_up(rows, (q4 && shared_r1x4_) ? 4u : 8u));
    }

    void gemv_residual(VkCommandBuffer command, VkDescriptorSet set, uint32_t rows,
                       uint32_t inner, uint32_t activation_scale,
                       uint32_t output_offset) {
        current_command_ = command;
        const GemvPush push{rows, inner, activation_scale, output_offset};
        const bool q4 = q4_sets_.count(set) != 0;
        const VkPipeline pipeline = q4 ?
            (shared_r1x4_ ? kernels_.pipelines().q4g64t_gemv_residual_r1x4 :
                            kernels_.pipelines().q4g64t_gemv_residual) :
            kernels_.pipelines().q8_gemv_residual;
        dispatch(set, pipeline, &push,
                 divide_up(rows, (q4 && shared_r1x4_) ? 4u : 8u));
    }

    void record_hc_pre(VkCommandBuffer command, VkDescriptorSet mix,
                       VkDescriptorSet split, VkDescriptorSet pre,
                       VkDescriptorSet norm, VkDescriptorSet fused_mix_split,
                       VkDescriptorSet fused_pre_norm) {
        current_command_ = command;
        if (hc_fused_) {
            const HcFusedPush push{rms_bits(),
                float_word(index_.header().hc_epsilon), 0u, 0u};
            dispatch(fused_mix_split, kernels_.pipelines().hc_mix_sinkhorn_fused,
                     &push, 1u);
            compute_barrier(command);
            dispatch(fused_pre_norm, kernels_.pipelines().hc_pre_rmsnorm_fused,
                     &push, 1u);
            compute_barrier(command);
            return;
        }
        const HcMixPush mix_push{kDimension, kHcMultiplicity, 24u, rms_bits()};
        dispatch(mix, kernels_.pipelines().hc_mix, &mix_push, 6u);
        compute_barrier(command);
        const HcSplitPush split_push{1u, kHcMultiplicity, 20u,
                                     float_word(index_.header().hc_epsilon)};
        dispatch(split, kernels_.pipelines().hc_sinkhorn, &split_push, 1u);
        compute_barrier(command);
        const HcApplyPush pre_push{kDimension, kHcMultiplicity, 24u, 0u};
        dispatch(pre, kernels_.pipelines().hc_pre, &pre_push,
                 divide_up(kDimension, 64u));
        compute_barrier(command);
        const RmsPush norm_push{1u, kDimension, rms_bits(), 0u};
        dispatch(norm, kernels_.pipelines().rmsnorm, &norm_push, 1u);
        compute_barrier(command);
    }

    void copy_kv(VkCommandBuffer command, uint32_t layer, uint32_t slot) {
        const VkDeviceSize layer_values =
            static_cast<VkDeviceSize>(kWindow + 32u) * kHeadDimension;
        const VkDeviceSize destination =
            (static_cast<VkDeviceSize>(layer) * layer_values +
             static_cast<VkDeviceSize>(slot) * kHeadDimension) * sizeof(float);
        copy_compute_result(command, executor_.kv(), 0, executor_.kv_cache(),
                            destination, kHeadDimension * sizeof(float));
    }

    const Runtime& runtime_;
    ExecutorScaffold& executor_;
    const SharedIndex& index_;
    Dsv4KernelLibrary kernels_;
    Buffer hidden_alt_{};
    Buffer reduced_{};
    Buffer hc_mixes_{};
    Buffer hc_split_{};
    Buffer quantized_{};
    Buffer expert_quantized_{};
    Buffer intermediate_quantized_{};
    Buffer o_rank_{};
    Buffer dense_gate_up_{};
    Buffer ffn_accumulator_{};
    Buffer router_logits_{};
    Buffer compressor_kv_{};
    Buffer compressor_score_{};
    Buffer compressor_raw_{};
    Buffer indices_{};
    Buffer ones_{};
    Buffer hc_head_params_{};
    Buffer rope_cos_plain_{};
    Buffer rope_sin_plain_{};
    Buffer rope_cos_compressed_{};
    Buffer rope_sin_compressed_{};
    Buffer argmax_workspace_{};
    CommonSets common_{};
    GlobalSets global_{};
    std::vector<LayerSets> layers_;
    std::unordered_set<VkDescriptorSet> q4_sets_;
    std::array<int32_t, kLayers> ratio4_slot_{};
    uint32_t ratio4_layers_ = 0;
    uint32_t current_position_ = 0;
    VkCommandBuffer current_command_ = VK_NULL_HANDLE;
    bool capture_dspark_ = false;
    bool expert_r1x4_ = false;
    bool expert_r4_ = false;
    bool q4_experts_ = false;
    bool q4_swar_ = false;
    bool q4_expert_batch_bda_ = false;
    bool hc_fused_ = false;
    bool rms_rope_fused_ = false;
    bool shared_r1x4_ = false;
    bool global_embedding_q4_ = false;
};

static std::filesystem::path runtime_path(const std::filesystem::path& directory,
                                          const char* name) {
    const std::filesystem::path path = directory / name;
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Missing DeepSeek runtime asset: " + path.string());
    return path;
}

} // namespace dsv4

int dsv4_cli_main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 2) {
            std::cerr << "usage: amd_deepseek_v4.exe <runtime-directory> "
                         "[--inspect | --init | --tokenize <text> | <prompt> [new-tokens]]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file(
            dsv4::runtime_path(directory, "tokenizer.ovb").string());
        dsv4::Tokenizer tokenizer(tokenizer_file);
        if (argc >= 4 && std::strcmp(argv[2], "--tokenize") == 0) {
            const std::vector<uint32_t> tokens = tokenizer.chat_prompt(argv[3]);
            std::cout << "tokens:";
            for (uint32_t token : tokens) std::cout << ' ' << token;
            std::cout << "\nuser round-trip: " << tokenizer.decode(
                tokenizer.encode_text(argv[3])) << "\n";
            return 0;
        }
        const bool force_q8_shared = std::getenv("DSV4_Q8_SHARED") != nullptr;
        const std::filesystem::path q4_shared = directory / "model-q4g64.ovs";
        const std::filesystem::path q4_global = directory / "model-q4g64-global.ovs";
        const std::filesystem::path q4_head = directory / "model-q4g64-head.ovs";
        const std::filesystem::path q4_embed = directory / "model-q4g64-embed.ovs";
        const bool use_q4_global = !force_q8_shared &&
            std::getenv("DSV4_Q4_GLOBAL") != nullptr && std::filesystem::exists(q4_global);
        const bool use_q4_head = !force_q8_shared && !use_q4_global &&
            std::getenv("DSV4_Q4_HEAD") != nullptr && std::filesystem::exists(q4_head);
        const bool use_q4_embed = !force_q8_shared && !use_q4_global && !use_q4_head &&
            std::getenv("DSV4_Q4_EMBED") != nullptr && std::filesystem::exists(q4_embed);
        const char* shared_name = use_q4_global ? "model-q4g64-global.ovs" :
            (use_q4_head ? "model-q4g64-head.ovs" :
            (use_q4_embed ? "model-q4g64-embed.ovs" :
            (!force_q8_shared && std::filesystem::exists(q4_shared) ?
             "model-q4g64.ovs" : "model.ovs")));
        dsv4::ReadOnlyMapping shared(
            dsv4::runtime_path(directory, shared_name).string());
        const bool request_q4_experts = std::getenv("DSV4_Q4_EXPERTS") != nullptr;
        const std::filesystem::path q4_experts = directory / "experts-q4g64.ovx";
        if (request_q4_experts && !std::filesystem::exists(q4_experts))
            throw std::runtime_error("DSV4_Q4_EXPERTS requested but complete experts-q4g64.ovx is absent");
        const char* expert_name = request_q4_experts ? "experts-q4g64.ovx" : "experts.ovx";
        dsv4::ReadOnlyMapping experts(
            dsv4::runtime_path(directory, expert_name).string(),
            std::getenv("DSV4_EXTERNAL_HOST") != nullptr);
        dsv4::SharedIndex shared_index(shared);
        dsv4::ExpertIndex expert_index(experts);
        if (expert_index.q4g64t() != request_q4_experts)
            throw std::runtime_error("DeepSeek expert precision does not match DSV4_Q4_EXPERTS");
        if (argc >= 3 && std::strcmp(argv[2], "--inspect") == 0) {
            std::cout << "DeepSeek-V4-Flash-0731 runtime files validated\n"
                      << "shared bytes: " << shared.size() << "\n"
                      << "expert bytes: " << experts.size() << "\n"
                      << "maximum shared layer bytes: "
                      << shared_index.maximum_layer_bytes() << "\n";
            return 0;
        }
        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << "\n";
        std::cout << "expert precision: " <<
            (expert_index.q4g64t() ? "Q4G64T K64/BF16" : "native E2M1 FP4/UE8M0") << "\n";
        std::cout << "shared precision: " <<
            (use_q4_global ? "Q4G64T layers+embedding+head / Q8 router" :
            (use_q4_head ? "Q4G64T layers+head / Q8 embedding+router" :
            (use_q4_embed ? "Q4G64T layers+embedding / Q8 head+router" :
            (shared_index.header().shared_format == 3 ? "Q4G64T layers / Q8 global+router" :
                                                       "row-Q8")))) << "\n";
        if (argc < 3) throw std::runtime_error("A DeepSeek prompt is required");
        const uint32_t maximum_new_tokens = argc >= 4 ?
            static_cast<uint32_t>(std::stoul(argv[3])) : 8u;
        std::vector<uint32_t> generated;
        uint64_t cache_hits = 0, cache_misses = 0, expert_transfer_bytes = 0;
        uint64_t direct_host_expert_bytes = 0;
        uint64_t host_cache_hits = 0, host_cache_misses = 0;
        uint64_t host_cache_rejections = 0;
        uint64_t host_budget_bytes = 0, explicit_host_bytes = 0;
        uint64_t vulkan_host_bytes = 0, plain_host_bytes = 0;
        uint64_t plain_stage_bytes = 0;
        uint64_t hybrid_promotions = 0, hybrid_copy_bytes = 0;
        uint64_t imported_plain_bytes = 0;
        uint64_t imported_plain_transfer_bytes = 0;
        uint32_t imported_plain_blocks = 0;
        uint64_t process_private_bytes = 0, process_working_set_bytes = 0;
        uint32_t host_cache_slots = 0;
        bool budgeted_io = false;
        bool expert_bundles = false;
        dsv4::WeightStreamer::BundleMetrics bundle_metrics{};
        bool plain_host_cache = false;
        bool expert_blueprint_enabled = false;
        dsv4::ExpertBlueprint::Metrics expert_blueprint_metrics{};
        bool top6_set_policy_enabled = false;
        dsv4::GlobalExpertCache::SetPolicyMetrics top6_set_policy_metrics{};
        bool adjacent_rank1_hint_enabled = false;
        dsv4::AdjacentRank1Predictor::Metrics adjacent_rank1_metrics{};
        dsv4::GlobalHostExpertCache::HintMetrics host_hint_metrics{};
        bool progressive_experts_enabled = false;
        dsv4::ExecutorScaffold::ProgressiveMetrics progressive_metrics{};
        double pre_seconds = 0.0, post_seconds = 0.0, acquire_seconds = 0.0;
        double decode_seconds = 0.0;
        uint64_t decode_passes = 0;
        const auto started = std::chrono::steady_clock::now();
        {
            dsv4::ExecutorScaffold executor(runtime, shared, shared_index, expert_index);
            const std::filesystem::path shader_directory =
                std::filesystem::absolute(argv[0]).parent_path();
            dsv4::DeepSeekProgram program(runtime, executor, shared_index, shared,
                                           shader_directory);
            const bool shared_acquire_overlap =
                std::getenv("DSV4_SHARED_ACQUIRE_OVERLAP") != nullptr;
            const bool progressive_experts =
                std::getenv("DSV4_PROGRESSIVE_EXPERTS") != nullptr;
            std::cout << "shared weights: "
                      << (executor.shared_weights_resident() ? "resident" : "double-buffered")
                      << "\nexpert cache slots: " << executor.expert_cache_slots()
                      << "\nexpert bundle store: "
                      << (executor.expert_bundles() ? "enabled" : "disabled")
                      << "\nshared/acquisition overlap: "
                      << (shared_acquire_overlap ? "yes" : "no")
                      << "\nprogressive per-rank experts: "
                      << (progressive_experts ? "yes" : "no") << "\n";
            if (std::strcmp(argv[2], "--init") != 0) {
                const std::vector<uint32_t> prompt = tokenizer.chat_prompt(argv[2]);
                dsv4::ExecutorScaffold::SharedPreRecorder shared_pre =
                    [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                        program.record_shared_pre(command, layer, parity);
                    };
                dsv4::ExecutorScaffold::ProgressiveExpertRecorders progressive{
                    [&](VkCommandBuffer command, uint32_t layer) {
                        program.record_progressive_expert_prefix(command, layer);
                    },
                    [&](VkCommandBuffer command, uint32_t layer, uint32_t rank) {
                        program.record_progressive_expert_rank(command, layer, rank);
                    },
                    [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                        program.record_progressive_expert_finish(command, layer, parity);
                    }};
                generated = executor.generate(tokenizer, prompt, maximum_new_tokens,
                    [&](VkCommandBuffer command, uint32_t token, uint32_t position) {
                        program.record_embedding(command, token, position);
                    },
                    [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                        program.record_pre(command, layer, parity);
                    },
                    [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                        if (shared_acquire_overlap)
                            program.record_post_after_shared(command, layer, parity);
                        else
                            program.record_post(command, layer, parity);
                    },
                    [&](VkCommandBuffer command, uint32_t position) {
                        program.record_final(command, position);
                    }, shared_acquire_overlap ? &shared_pre : nullptr,
                    progressive_experts ? &progressive : nullptr);
            } else {
                std::cout << "DeepSeek Vulkan resources and descriptors initialized\n";
            }
            cache_hits = executor.cache_hits();
            cache_misses = executor.cache_misses();
            expert_transfer_bytes = executor.expert_transfer_bytes();
            direct_host_expert_bytes = executor.direct_host_expert_bytes();
            pre_seconds = executor.pre_seconds();
            post_seconds = executor.post_seconds();
            acquire_seconds = executor.acquire_seconds();
            decode_seconds = executor.decode_seconds();
            decode_passes = executor.decode_passes();
            host_cache_hits = executor.host_cache_hits();
            host_cache_misses = executor.host_cache_misses();
            host_cache_rejections = executor.host_cache_admission_rejections();
            host_cache_slots = executor.host_cache_slots();
            host_budget_bytes = executor.host_budget_bytes();
            budgeted_io = executor.budgeted_io();
            expert_bundles = executor.expert_bundles();
            bundle_metrics = executor.bundle_metrics();
            vulkan_host_bytes = dsv4::dsv4_host_buffer_allocated_bytes;
            plain_host_bytes = dsv4::dsv4_plain_host_allocated_bytes;
            explicit_host_bytes = vulkan_host_bytes + plain_host_bytes;
            plain_stage_bytes = executor.plain_stage_bytes();
            hybrid_promotions = executor.hybrid_promotions();
            hybrid_copy_bytes = executor.hybrid_copy_bytes();
            imported_plain_bytes = executor.imported_plain_bytes();
            imported_plain_transfer_bytes = executor.imported_plain_transfer_bytes();
            imported_plain_blocks = executor.imported_plain_blocks();
            plain_host_cache = executor.plain_host_cache();
            expert_blueprint_enabled = executor.expert_blueprint_enabled();
            expert_blueprint_metrics = executor.expert_blueprint_metrics();
            top6_set_policy_enabled = executor.top6_set_policy_enabled();
            top6_set_policy_metrics = executor.top6_set_policy_metrics();
            adjacent_rank1_hint_enabled = executor.adjacent_rank1_hint_enabled();
            adjacent_rank1_metrics = executor.adjacent_rank1_metrics();
            host_hint_metrics = executor.host_hint_metrics();
            progressive_experts_enabled = executor.progressive_experts_enabled();
            progressive_metrics = executor.progressive_metrics();
            const dsv4::HostMemorySnapshot process_memory = dsv4::host_memory_snapshot();
            process_private_bytes = process_memory.private_bytes;
            process_working_set_bytes = process_memory.working_set_bytes;
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        std::cout << "\ntoken ids:";
        for (uint32_t token : generated) std::cout << ' ' << token;
        std::cout << "\nelapsed: " << seconds << " s\n"
                  << "generated throughput: "
                  << (seconds > 0.0 ? generated.size() / seconds : 0.0) << " tok/s\n"
                  << "decode throughput: "
                  << (decode_seconds > 0.0 ? decode_passes / decode_seconds : 0.0)
                  << " tok/s\n"
                  << "decode passes: " << decode_passes << "\n"
                  << "cache hits/misses: " << cache_hits << '/' << cache_misses << "\n"
                  << "RAM-cache hits/misses: " << host_cache_hits << '/'
                   << host_cache_misses << "\n"
                  << "RAM-cache admission rejections: " << host_cache_rejections << "\n"
                  << "RAM-cache served / disk expert reads: "
                  << static_cast<double>(host_cache_hits * dsv4::kExpertRecordBytes) /
                         (1024.0 * 1024.0 * 1024.0) << " / "
                  << static_cast<double>(host_cache_misses * dsv4::kExpertRecordBytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "host budget / controlled host backing / RAM slots / direct I/O: "
                  << static_cast<double>(host_budget_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " / "
                  << static_cast<double>(explicit_host_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB / " << host_cache_slots << " / "
                  << (budgeted_io ? "yes" : "no") << "\n"
                  << "Vulkan/plain host backing / plain-cache mode: "
                  << static_cast<double>(vulkan_host_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " / "
                  << static_cast<double>(plain_host_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB / "
                  << (plain_host_cache ? "yes" : "no") << "\n"
                  << "plain RAM staging copies: "
                  << static_cast<double>(plain_stage_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "hybrid L2 promotions / swap copies: "
                  << hybrid_promotions << " / "
                  << static_cast<double>(hybrid_copy_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "imported L2 blocks / bytes: " << imported_plain_blocks << " / "
                  << static_cast<double>(imported_plain_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "imported L2 transfer source bytes: "
                  << static_cast<double>(imported_plain_transfer_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "process private / working set: "
                  << static_cast<double>(process_private_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " / "
                  << static_cast<double>(process_working_set_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "expert transfer: " <<
                     static_cast<double>(expert_transfer_bytes) / (1024.0 * 1024.0 * 1024.0)
                   << " GiB\n"
                  << "direct host expert reads: " <<
                     static_cast<double>(direct_host_expert_bytes) /
                         (1024.0 * 1024.0 * 1024.0)
                  << " GiB\n"
                  << "attention+router / acquisition / expert+shared: "
                  << pre_seconds << " / " << acquire_seconds << " / "
                  << post_seconds << " s\n"
                  << "peak Vulkan allocations: "
                  << static_cast<double>(peak_vulkan_buffer_bytes) / (1024.0 * 1024.0 * 1024.0)
                  << " GiB\n";
        if (progressive_experts_enabled) {
            std::cout
                << "progressive layers (mixed/all-ready-fast): "
                << progressive_metrics.mixed_layers << '/'
                << progressive_metrics.all_ready_fast_layers << "\n"
                << "progressive ranks (resident/RAM/disk): "
                << progressive_metrics.resident_ranks << '/'
                << progressive_metrics.ram_ranks << '/'
                << progressive_metrics.disk_ranks << "\n"
                << "progressive finite submissions / disk-wait / schedule: "
                << progressive_metrics.finite_submissions << " / "
                << progressive_metrics.disk_wait_seconds << " / "
                << progressive_metrics.schedule_seconds << " s\n";
        }
        if (expert_bundles) {
            const uint64_t average = bundle_metrics.read_count ?
                bundle_metrics.read_bytes / bundle_metrics.read_count : 0;
            std::cout << "expert bundle reads (count/bytes/avg/stalled-layers): "
                      << bundle_metrics.read_count << '/'
                      << bundle_metrics.read_bytes << '/'
                      << average << '/'
                      << bundle_metrics.stalled_layers << "\n";
        }
        if (expert_blueprint_enabled) {
            const auto& metrics = expert_blueprint_metrics;
            std::cout
                << "blueprint readiness layers (VRAM/VRAM+RAM/SSD/total): "
                << metrics.top6_vram_ready_layers << '/'
                << metrics.top6_vram_or_ram_ready_layers << '/'
                << metrics.layers_with_ssd_miss << '/'
                << metrics.route_layers << "\n"
                << "blueprint reads (count/bytes): "
                << metrics.expert_read_count << '/' << metrics.expert_read_bytes << "\n"
                << "blueprint H2D (count/bytes/max-timeline): "
                << metrics.h2d_count << '/' << metrics.h2d_bytes << '/'
                << metrics.maximum_timeline << "\n"
                << "blueprint transfer states (idle/read/H2D/pending/ETA-s): "
                << metrics.idle_entries << '/' << metrics.nvme_read_entries << '/'
                << metrics.h2d_entries << '/' << metrics.pending_transfers << '/'
                << metrics.estimated_outstanding_seconds << "\n";
        }
        if (top6_set_policy_enabled) {
            const auto& metrics = top6_set_policy_metrics;
            std::cout
                << "Top-6 set readiness (complete/one-missing/sets): "
                << metrics.complete_set_ready << '/'
                << metrics.one_missing_sets << '/'
                << metrics.route_sets << "\n"
                << "Top-6 set hits/misses/max-misses: "
                << metrics.initial_hits << '/'
                << metrics.total_misses << '/'
                << metrics.maximum_set_misses << "\n"
                << "set-policy victim changes (changed/decisions/coact-saved/adjacent-saved): "
                << metrics.changed_victims << '/'
                << metrics.victim_decisions << '/'
                << metrics.coactivation_saved << '/'
                << metrics.adjacent_affinity_saved << "\n"
                << "adjacent Top-6 prediction (hits/targets/sets): "
                << metrics.adjacent_prediction_hits << '/'
                << metrics.adjacent_prediction_targets << '/'
                << metrics.adjacent_prediction_sets << "\n"
                << "set-policy learning updates (coactivation/adjacent): "
                << metrics.coactivation_updates << '/'
                << metrics.adjacent_updates << "\n";
        }
        if (adjacent_rank1_hint_enabled) {
            std::cout
                << "adjacent rank1 hint (route-hits/qualified/host-resident/threshold): "
                << adjacent_rank1_metrics.route_hits << '/'
                << adjacent_rank1_metrics.qualified << '/'
                << adjacent_rank1_metrics.host_hints << '/'
                << adjacent_rank1_metrics.threshold << "\n"
                << "adjacent host hint (installed/matched/protected-victims): "
                << host_hint_metrics.installed << '/'
                << host_hint_metrics.matched_requests << '/'
                << host_hint_metrics.protected_victims << "\n";
        }
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        runtime = {};
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek-V4 runtime error: " << error.what()
                  << " (active Vulkan allocations "
                  << static_cast<double>(active_vulkan_buffer_bytes) /
                         (1024.0 * 1024.0 * 1024.0)
                  << " GiB)\n";
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}

#ifndef OVLLM_DSV4_RUNTIME_ONLY
int main(int argc, char** argv) {
    return dsv4_cli_main(argc, argv);
}
#endif
