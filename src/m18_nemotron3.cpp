#ifndef OVLLM_DSV4_RUNTIME_INCLUDED
#define OVLLM_DSV4_RUNTIME_INCLUDED 1
#define OVLLM_DSV4_RUNTIME_ONLY
#include "m13_deepseek_v4.cpp"
#endif

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <unordered_map>

// Nemotron-3-Nano-30B-A3B text-only executor.  This is deliberately separate from
// the retained DeepSeek and Step-3.7 executors.  It reuses their finite Vulkan
// queues and Q4 expert-acquisition design, but all model math below follows the
// authoritative Qwen3.6 text architecture.
namespace nemotron3 {

constexpr uint32_t kDim = 2688;
constexpr uint32_t kMoeDim = 1856;
constexpr uint32_t kSharedDim = 3712;
constexpr uint32_t kLayers = 52;
constexpr uint32_t kMoeLayers = 23;
constexpr uint32_t kExperts = 128;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kVocabulary = 131072;
constexpr uint32_t kBaseVocabulary = 131072;
constexpr uint32_t kFullLayers = 6;
constexpr uint32_t kMambaLayers = 23;
constexpr uint32_t kAttentionHeads = 32;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kMambaHeads = 64;
constexpr uint32_t kMambaHeadDim = 64;
constexpr uint32_t kMambaGroups = 8;
constexpr uint32_t kMambaState = 128;
constexpr uint32_t kMambaIntermediate = 4096;
constexpr uint32_t kMambaConvChannels = 6144;
constexpr uint32_t kMambaInput = 10304;
constexpr uint32_t kConvWidth = 4;
#ifdef OVLLM_LONG_CONTEXT_FORK
constexpr uint32_t kDefaultContext = 2048;
constexpr uint32_t kAttentionChunk = 1024;
constexpr uint32_t kAttentionPartialStride = kHeadDim + 2;
#else
constexpr uint32_t kMaximumContext = 2048;
#endif
constexpr char kPattern[] = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kUpScale = 0;
constexpr uint64_t kUpWeight = 623616;
constexpr uint64_t kDownScale = 3118080;
constexpr uint64_t kDownWeight = 3741696;
constexpr uint64_t kExpertRecordBytes = 6238208;
static_assert(kExpertRecordBytes % 4096 == 0);

constexpr uint32_t kBos = 1;
constexpr uint32_t kEndOfText = 2;
constexpr uint32_t kImStart = 10;
constexpr uint32_t kImEnd = 11;
constexpr uint32_t kThink = 12;
constexpr uint32_t kEndThink = 13;

static bool mamba_layer(uint32_t layer) { return kPattern[layer] == 'M'; }
static bool moe_layer(uint32_t layer) { return kPattern[layer] == 'E'; }
static bool attention_layer(uint32_t layer) { return kPattern[layer] == '*'; }
static uint32_t moe_index(uint32_t layer) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < layer; ++i) result += moe_layer(i);
    return result;
}
static uint32_t attention_index(uint32_t layer) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < layer; ++i) result += attention_layer(i);
    return result;
}
static uint32_t mamba_index(uint32_t layer) {
    uint32_t result = 0;
    for (uint32_t i = 0; i < layer; ++i) result += mamba_layer(i);
    return result;
}

using dsv4::ExpertHeader;
using dsv4::GroupEntry;
using dsv4::MergeEntry;
using dsv4::SharedHeader;
using dsv4::TensorEntry;
using dsv4::TensorFormat;
using dsv4::TokenEntry;
using dsv4::TokenizerHeader;

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

struct TensorDevice {
    DescriptorRange data{}, auxiliary{};
    TensorFormat format = TensorFormat::f32;
    uint32_t rank = 0;
    std::array<uint64_t, 8> shape{};
};

static void read_at(std::ifstream& input, uint64_t offset, void* output,
                    size_t bytes, const char* label) {
    input.seekg(static_cast<std::streamoff>(offset));
    input.read(static_cast<char*>(output), static_cast<std::streamsize>(bytes));
    if (!input)
        throw std::runtime_error(std::string("Qwen shared read failed: ") + label);
}

class SharedIndex {
public:
    explicit SharedIndex(const std::filesystem::path& path) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open Qwen shared container");
        read_at(input, 0, &header_, sizeof(header_), "header");
        if (std::memcmp(header_.magic, "ON3NSHR\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.tensor_entry_bytes != sizeof(TensorEntry) ||
            header_.dimension != kDim || header_.moe_dimension != kMoeDim ||
            header_.layers != kLayers || header_.heads != kAttentionHeads ||
            header_.kv_heads != kKvHeads || header_.head_dimension != kHeadDim ||
            header_.vocabulary != kVocabulary || header_.experts != kExperts ||
            header_.top_k != kTopK ||
            header_.expert_record_bytes != kExpertRecordBytes)
            throw std::runtime_error("Unsupported Nemotron shared container");
        const uint64_t actual = std::filesystem::file_size(path);
        if (header_.file_bytes != actual || header_.group_count != kLayers + 1 ||
            header_.tensor_count == 0 || header_.tensor_count > 4096)
            throw std::runtime_error("Invalid Qwen shared container bounds");
        groups_.resize(static_cast<size_t>(header_.group_count));
        entries_.resize(static_cast<size_t>(header_.tensor_count));
        read_at(input, header_.group_table_offset, groups_.data(),
                groups_.size() * sizeof(GroupEntry), "groups");
        read_at(input, header_.tensor_table_offset, entries_.data(),
                entries_.size() * sizeof(TensorEntry), "tensors");
        for (uint32_t group = 0; group < groups_.size(); ++group) {
            const GroupEntry& g = groups_[group];
            if (g.data_begin < kHeaderBytes || g.data_end <= g.data_begin ||
                g.data_end > actual || (g.data_begin & 4095u) ||
                (g.data_end & 4095u) ||
                uint64_t(g.first_tensor) + g.tensor_count > entries_.size())
                throw std::runtime_error("Invalid Qwen shared group");
            for (uint32_t i = 0; i < g.tensor_count; ++i) {
                const TensorEntry& entry = entries_[g.first_tensor + i];
                size_t length = 0;
                while (length < sizeof(entry.name) && entry.name[length]) ++length;
                if (length == sizeof(entry.name))
                    throw std::runtime_error("Unterminated Qwen tensor name");
                Entry value{};
                value.entry = entry;
                value.group = group;
                if (!by_name_.emplace(std::string(entry.name, length), value).second)
                    throw std::runtime_error("Duplicate Qwen tensor");
            }
        }
    }

    struct Entry { TensorEntry entry{}; uint32_t group = 0; };
    const Entry& require(const std::string& name) const {
        const auto found = by_name_.find(name);
        if (found == by_name_.end())
            throw std::runtime_error("Missing Qwen tensor: " + name);
        return found->second;
    }
    const SharedHeader& header() const { return header_; }
    const std::vector<GroupEntry>& groups() const { return groups_; }
    const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
    SharedHeader header_{};
    std::vector<GroupEntry> groups_;
    std::vector<TensorEntry> entries_;
    std::unordered_map<std::string, Entry> by_name_;
};

static HANDLE open_unbuffered(const std::filesystem::path& path) {
    HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING,
                              FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("Could not open unbuffered Qwen weight file");
    return file;
}

static void read_unbuffered(HANDLE file, uint64_t offset, void* destination,
                            uint64_t bytes) {
    if ((offset & 4095u) || (bytes & 4095u) ||
        (reinterpret_cast<uintptr_t>(destination) & 4095u))
        throw std::runtime_error("Unaligned Qwen unbuffered read");
    LARGE_INTEGER where{};
    where.QuadPart = offset;
    if (!SetFilePointerEx(file, where, nullptr, FILE_BEGIN))
        throw std::runtime_error("Qwen seek failed");
    uint8_t* output = static_cast<uint8_t*>(destination);
    while (bytes) {
        const DWORD part = static_cast<DWORD>(
            std::min<uint64_t>(bytes, 64ull * 1024 * 1024));
        DWORD transferred = 0;
        if (!ReadFile(file, output, part, &transferred, nullptr) ||
            transferred != part)
            throw std::runtime_error("Qwen unbuffered read failed");
        output += part;
        bytes -= part;
    }
}

class DeviceWeights {
public:
    DeviceWeights(const Runtime& runtime, const SharedIndex& index)
        : runtime_(runtime), index_(index), transfer_(runtime, runtime.queue) {
        HANDLE file = open_unbuffered(index.path());
        Buffer staging = dsv4::create_host_buffer_uninitialized(
            runtime, 64ull * 1024 * 1024);
        try {
            groups_.resize(index.groups().size());
            for (uint32_t gi = 0; gi < groups_.size(); ++gi) {
                const GroupEntry& group = index.groups()[gi];
                const uint64_t bytes = group.data_end - group.data_begin;
                groups_[gi] = create_device_buffer(runtime, bytes);
                device_bytes_ += groups_[gi].allocation_size;
                for (uint64_t done = 0; done < bytes; done += staging.size) {
                    const uint64_t part = std::min<uint64_t>(staging.size, bytes - done);
                    read_unbuffered(file, group.data_begin + done, staging.mapped, part);
                    dsv4::flush_buffer_range(runtime, staging, 0, part);
                    const uint64_t signal = transfer_.submit([&](VkCommandBuffer command) {
                        VkBufferCopy copy{0, done, part};
                        vkfn::CmdCopyBuffer(command, staging.handle,
                                            groups_[gi].handle, 1, &copy);
                        dsv4::transfer_barrier(command, groups_[gi]);
                    });
                    transfer_.wait(signal);
                }
            }
        } catch (...) {
            CloseHandle(file);
            destroy_buffer(runtime, staging);
            throw;
        }
        CloseHandle(file);
        destroy_buffer(runtime, staging);
    }

    ~DeviceWeights() {
        for (Buffer& group : groups_) destroy_buffer(runtime_, group);
    }

    TensorDevice tensor(const std::string& name) const {
        const SharedIndex::Entry& found = index_.require(name);
        const GroupEntry& group = index_.groups()[found.group];
        const TensorEntry& entry = found.entry;
        const auto range = [&](uint64_t offset, uint64_t bytes) {
            if (!bytes) return whole(groups_[found.group]);
            if (offset < group.data_begin || offset + bytes > group.data_end)
                throw std::runtime_error("Qwen tensor outside shared group");
            return arena_range(groups_[found.group], offset - group.data_begin, bytes);
        };
        TensorDevice result;
        result.data = range(entry.data_offset, entry.data_bytes);
        result.auxiliary = range(entry.auxiliary_offset, entry.auxiliary_bytes);
        result.format = static_cast<TensorFormat>(entry.dtype);
        result.rank = entry.rank;
        std::copy(std::begin(entry.shape), std::end(entry.shape),
                  result.shape.begin());
        return result;
    }

    uint64_t device_bytes() const { return device_bytes_; }

private:
    const Runtime& runtime_;
    const SharedIndex& index_;
    dsv4::FiniteQueue transfer_;
    std::vector<Buffer> groups_;
    uint64_t device_bytes_ = 0;
};

class Tokenizer {
public:
    explicit Tokenizer(const dsv4::ReadOnlyMapping& file) : file_(file) {
        if (file.size() < sizeof(TokenizerHeader))
            throw std::runtime_error("Truncated Qwen tokenizer.ovb");
        std::memcpy(&header_, file.data(), sizeof(header_));
        if (std::memcmp(header_.magic, "OVBPE2\0\0", 8) != 0 ||
            header_.version != 2 || header_.header_bytes != sizeof(TokenizerHeader) ||
            header_.vocabulary != kVocabulary ||
            header_.base_vocabulary != kBaseVocabulary ||
            header_.merge_count != 269443 || header_.bos != kBos ||
            header_.eos != kImEnd || header_.configured_pad != kEndOfText ||
            header_.pad_piece != 0 || header_.user != kImStart ||
            header_.assistant != kImEnd || header_.think != kThink ||
            header_.end_think != kEndThink ||
            header_.token_entry_bytes != sizeof(TokenEntry) ||
            header_.merge_entry_bytes != sizeof(MergeEntry) ||
            header_.file_bytes != file.size())
            throw std::runtime_error("Unsupported Qwen tokenizer container");
        validate(header_.token_table_offset,
                 uint64_t(kVocabulary) * sizeof(TokenEntry), "tokens");
        validate(header_.merge_table_offset,
                 uint64_t(header_.merge_count) * sizeof(MergeEntry), "merges");
        validate(header_.pieces_offset, header_.pieces_bytes, "pieces");
        entries_ = reinterpret_cast<const TokenEntry*>(
            file.data() + header_.token_table_offset);
        const auto* merges = reinterpret_cast<const MergeEntry*>(
            file.data() + header_.merge_table_offset);
        pieces_.resize(kVocabulary);
        special_.resize(kVocabulary);
        byte_token_.fill(UINT32_MAX);
        for (uint32_t token = 0; token < kVocabulary; ++token) {
            const TokenEntry& entry = entries_[token];
            if (entry.piece_offset < header_.pieces_offset ||
                entry.piece_offset > header_.pieces_offset + header_.pieces_bytes ||
                entry.byte_length > header_.pieces_offset + header_.pieces_bytes -
                                        entry.piece_offset)
                throw std::runtime_error("Qwen tokenizer piece out of bounds");
            pieces_[token] = std::string(reinterpret_cast<const char*>(
                file.data() + entry.piece_offset), entry.byte_length);
            special_[token] = (entry.flags & 2u) != 0 || token == kEndOfText ||
                              token == kImStart || token == kImEnd ||
                              token == kThink || token == kEndThink;
            if (token < kBaseVocabulary && pieces_[token].size() == 1) {
                const uint8_t byte = static_cast<uint8_t>(pieces_[token][0]);
                if (byte_token_[byte] == UINT32_MAX) byte_token_[byte] = token;
            }
        }
        merges_.reserve(uint64_t(header_.merge_count) * 2);
        for (uint32_t i = 0; i < header_.merge_count; ++i) {
            const MergeEntry& merge = merges[i];
            if (merge.left >= kBaseVocabulary || merge.right >= kBaseVocabulary ||
                merge.result >= kBaseVocabulary || merge.rank >= header_.merge_count)
                throw std::runtime_error("Invalid Qwen BPE merge");
            if (!merges_.emplace(pair_key(merge.left, merge.right),
                                 Merge{merge.result, merge.rank}).second)
                throw std::runtime_error("Duplicate Qwen BPE merge");
        }
        if (std::find(byte_token_.begin(), byte_token_.end(), UINT32_MAX) !=
            byte_token_.end())
            throw std::runtime_error("Qwen tokenizer lacks a raw byte token");
    }

    std::vector<uint32_t> encode_text(const std::string& text) const {
        std::vector<uint32_t> output;
        for (const std::string& segment : split_ascii(text)) {
            std::vector<uint32_t> symbols;
            symbols.reserve(segment.size());
            for (uint8_t byte : std::vector<uint8_t>(segment.begin(), segment.end()))
                symbols.push_back(byte_token_[byte]);
            while (symbols.size() > 1) {
                uint32_t best_rank = UINT32_MAX, best_result = 0;
                size_t best = symbols.size();
                for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                    const auto found = merges_.find(pair_key(symbols[i], symbols[i + 1]));
                    if (found != merges_.end() && found->second.rank < best_rank) {
                        best_rank = found->second.rank;
                        best_result = found->second.result;
                        best = i;
                    }
                }
                if (best == symbols.size()) break;
                symbols[best] = best_result;
                symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best + 1));
            }
            output.insert(output.end(), symbols.begin(), symbols.end());
        }
        return output;
    }

    std::vector<uint32_t> chat_prompt(const std::string& user_text,
                                      bool thinking = true,
                                      const std::string& system_text = {}) const {
        std::vector<uint32_t> result;
        const auto text = [&](const std::string& value) {
            const std::vector<uint32_t> encoded = encode_text(value);
            result.insert(result.end(), encoded.begin(), encoded.end());
        };
        // The official Nemotron template emits an empty system turn when the
        // caller supplies no system message; omitting it changes the prompt.
        result.push_back(kImStart);
        text("system\n" + system_text);
        result.push_back(kImEnd);
        text("\n");
        result.push_back(kImStart);
        text("user\n" + user_text);
        result.push_back(kImEnd);
        text("\n");
        result.push_back(kImStart);
        text("assistant\n");
        result.push_back(kThink);
        if (thinking) {
            text("\n");
        } else {
            // Official disabled-thinking generation suffix is immediate.
            result.push_back(kEndThink);
        }
        return result;
    }

    std::string decode_piece(uint32_t token) const {
        if (token >= pieces_.size() || special_[token]) return {};
        return pieces_[token];
    }
    bool is_eos(uint32_t token) const {
        return token == kImEnd || token == kEndOfText;
    }

private:
    struct Merge { uint32_t result; uint32_t rank; };
    static uint64_t pair_key(uint32_t left, uint32_t right) {
        return uint64_t(left) << 32u | right;
    }
    void validate(uint64_t offset, uint64_t bytes, const char* label) const {
        if (offset < header_.header_bytes || offset > file_.size() ||
            bytes > file_.size() - offset)
            throw std::runtime_error(std::string("Qwen tokenizer ") + label +
                                     " out of bounds");
    }
    static bool letter(uint8_t value) {
        return (value >= 'A' && value <= 'Z') ||
               (value >= 'a' && value <= 'z');
    }
    static bool digit(uint8_t value) { return value >= '0' && value <= '9'; }
    static bool space(uint8_t value) {
        return value == ' ' || value == '\t' || value == '\r' ||
               value == '\n' || value == '\v' || value == '\f';
    }
    static bool punctuation(uint8_t value) {
        return !space(value) && !letter(value) && !digit(value) && value < 0x80;
    }
    static bool contraction(const std::string& text, size_t position,
                            size_t& length) {
        if (text[position] != '\'') return false;
        static constexpr const char* suffixes[] = {
            "'re", "'ve", "'ll", "'s", "'t", "'m", "'d"};
        for (const char* suffix : suffixes) {
            const size_t n = std::strlen(suffix);
            if (position + n > text.size()) continue;
            bool equal = true;
            for (size_t i = 0; i < n; ++i) {
                const unsigned char a = static_cast<unsigned char>(text[position + i]);
                const unsigned char b = static_cast<unsigned char>(suffix[i]);
                equal = equal && std::tolower(a) == std::tolower(b);
            }
            if (equal) { length = n; return true; }
        }
        return false;
    }
    static std::vector<std::string> split_ascii(const std::string& text) {
        std::vector<std::string> result;
        size_t position = 0;
        while (position < text.size()) {
            const size_t begin = position;
            const uint8_t first = static_cast<uint8_t>(text[position]);
            size_t contraction_length = 0;
            if (contraction(text, position, contraction_length)) {
                position += contraction_length;
            } else if (digit(first)) {
                ++position; // Qwen's regex isolates every Unicode number.
            } else if (letter(first)) {
                while (position < text.size() &&
                       letter(static_cast<uint8_t>(text[position]))) ++position;
            } else if (first != '\r' && first != '\n' && !digit(first) &&
                       position + 1 < text.size() &&
                       letter(static_cast<uint8_t>(text[position + 1]))) {
                ++position;
                while (position < text.size() &&
                       letter(static_cast<uint8_t>(text[position]))) ++position;
            } else if (punctuation(first) ||
                       (first == ' ' && position + 1 < text.size() &&
                        punctuation(static_cast<uint8_t>(text[position + 1])))) {
                if (first == ' ') ++position;
                while (position < text.size() &&
                       punctuation(static_cast<uint8_t>(text[position]))) ++position;
                while (position < text.size() &&
                       (text[position] == '\r' || text[position] == '\n')) ++position;
            } else if (space(first)) {
                size_t end = position;
                size_t last_newline = std::string::npos;
                while (end < text.size() && space(static_cast<uint8_t>(text[end]))) {
                    if (text[end] == '\r' || text[end] == '\n') last_newline = end;
                    ++end;
                }
                position = last_newline == std::string::npos ? end : last_newline + 1;
                while (position < end &&
                       (text[position] == '\r' || text[position] == '\n')) ++position;
            } else if (first >= 0x80) {
                // Lossless byte fallback.  The bounded benchmark prompt is ASCII;
                // a full Unicode property engine is intentionally out of scope.
                while (position < text.size() &&
                       static_cast<uint8_t>(text[position]) >= 0x80) ++position;
            } else {
                ++position;
            }
            if (position == begin) ++position;
            result.push_back(text.substr(begin, position - begin));
        }
        return result;
    }

    const dsv4::ReadOnlyMapping& file_;
    TokenizerHeader header_{};
    const TokenEntry* entries_ = nullptr;
    std::vector<std::string> pieces_;
    std::vector<bool> special_;
    std::array<uint32_t, 256> byte_token_{};
    std::unordered_map<uint64_t, Merge> merges_;
};

class ExpertFile {
public:
    struct AsyncBatch {
        std::array<OVERLAPPED, kTopK> operations{};
        std::array<HANDLE, kTopK> events{};
        std::array<uint32_t, kTopK> ranks{};
        std::array<bool, kTopK> pending{};
        uint32_t count = 0;
        uint32_t remaining = 0;
        bool active = false;
    };

    explicit ExpertFile(const std::filesystem::path& path) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open Qwen expert container");
        read_at(input, 0, &header_, sizeof(header_), "expert header");
        if (std::memcmp(header_.magic, "ON3NNV4\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.dimension != kDim || header_.moe_dimension != kMoeDim ||
            header_.layers != kMoeLayers || header_.experts != kExperts ||
            header_.reserved != 103u ||
            header_.record_bytes != kExpertRecordBytes ||
            header_.core_records != uint64_t(kMoeLayers) * kExperts ||
            header_.file_bytes != std::filesystem::file_size(path) ||
            header_.w1_scale_offset != kUpScale ||
            header_.w1_weight_offset != kUpWeight ||
            header_.w3_scale_offset != 0 ||
            header_.w3_weight_offset != 0 ||
            header_.w2_scale_offset != kDownScale ||
            header_.w2_weight_offset != kDownWeight)
            throw std::runtime_error("Unsupported Qwen expert container");
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING,
                            FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS |
                                FILE_FLAG_OVERLAPPED,
                            nullptr);
        if (file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Could not open overlapped Qwen experts");
    }

    ~ExpertFile() {
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }

    uint64_t offset(uint32_t layer, uint32_t expert) const {
        if (layer >= kMoeLayers || expert >= kExperts)
            throw std::runtime_error("Invalid Qwen expert key");
        return header_.core_offset +
               (uint64_t(layer) * kExperts + expert) * kExpertRecordBytes;
    }

    void read_batch(const std::vector<uint64_t>& offsets,
                    const std::vector<void*>& destinations) {
        if (offsets.size() != destinations.size() || offsets.size() > kTopK)
            throw std::runtime_error("Invalid Qwen expert I/O batch");
        if (offsets.empty()) return;
        std::array<OVERLAPPED, kTopK> operations{};
        std::array<HANDLE, kTopK> events{};
        events.fill(nullptr);
        try {
            for (uint32_t i = 0; i < offsets.size(); ++i) {
                events[i] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (!events[i])
                    throw std::runtime_error("Qwen I/O event creation failed");
                operations[i].Offset = DWORD(offsets[i]);
                operations[i].OffsetHigh = DWORD(offsets[i] >> 32);
                operations[i].hEvent = events[i];
                const BOOL started = ReadFile(file_, destinations[i],
                                              DWORD(kExpertRecordBytes), nullptr,
                                              &operations[i]);
                if (!started && GetLastError() != ERROR_IO_PENDING)
                    throw std::runtime_error("Qwen expert read submission failed");
            }
            const DWORD waited = WaitForMultipleObjects(
                DWORD(offsets.size()), events.data(), TRUE, 60000);
            if (waited != WAIT_OBJECT_0)
                throw std::runtime_error("Qwen expert read timed out");
            for (uint32_t i = 0; i < offsets.size(); ++i) {
                DWORD transferred = 0;
                if (!GetOverlappedResult(file_, &operations[i], &transferred, FALSE) ||
                    transferred != kExpertRecordBytes)
                    throw std::runtime_error("Qwen overlapped expert read failed");
            }
        } catch (...) {
            for (HANDLE event : events) if (event) CloseHandle(event);
            throw;
        }
        for (HANDLE event : events) if (event) CloseHandle(event);
    }

    void begin_batch(const std::vector<uint64_t>& offsets,
                     const std::vector<void*>& destinations,
                     const std::vector<uint32_t>& ranks,
                     AsyncBatch& batch) {
        if (batch.active || offsets.size() != destinations.size() ||
            offsets.size() != ranks.size() || offsets.size() > kTopK)
            throw std::runtime_error("Invalid async Qwen expert batch");
        batch = {};
        batch.count = static_cast<uint32_t>(offsets.size());
        batch.remaining = batch.count;
        batch.active = true;
        try {
            for (uint32_t i = 0; i < batch.count; ++i) {
                batch.events[i] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (!batch.events[i])
                    throw std::runtime_error(
                        "Qwen async I/O event creation failed");
                batch.operations[i].Offset = DWORD(offsets[i]);
                batch.operations[i].OffsetHigh = DWORD(offsets[i] >> 32);
                batch.operations[i].hEvent = batch.events[i];
                batch.ranks[i] = ranks[i];
                batch.pending[i] = true;
                const BOOL started = ReadFile(
                    file_, destinations[i], DWORD(kExpertRecordBytes), nullptr,
                    &batch.operations[i]);
                if (!started && GetLastError() != ERROR_IO_PENDING)
                    throw std::runtime_error(
                        "Qwen async expert read submission failed");
            }
        } catch (...) {
            for (uint32_t i = 0; i < batch.count; ++i) {
                if (batch.pending[i]) CancelIoEx(file_, &batch.operations[i]);
                if (batch.events[i]) CloseHandle(batch.events[i]);
            }
            batch = {};
            throw;
        }
    }

    uint32_t wait_any(AsyncBatch& batch) {
        if (!batch.active || batch.remaining == 0)
            throw std::runtime_error("No pending Qwen expert read");
        std::array<HANDLE, kTopK> active_events{};
        std::array<uint32_t, kTopK> active_indices{};
        uint32_t active_count = 0;
        for (uint32_t i = 0; i < batch.count; ++i) {
            if (!batch.pending[i]) continue;
            active_events[active_count] = batch.events[i];
            active_indices[active_count++] = i;
        }
        const DWORD waited = WaitForMultipleObjects(
            active_count, active_events.data(), FALSE, 10000);
        if (waited < WAIT_OBJECT_0 ||
            waited >= WAIT_OBJECT_0 + active_count)
            throw std::runtime_error("Qwen progressive expert read timed out");
        const uint32_t index = active_indices[waited - WAIT_OBJECT_0];
        DWORD transferred = 0;
        if (!GetOverlappedResult(file_, &batch.operations[index],
                                 &transferred, FALSE) ||
            transferred != kExpertRecordBytes)
            throw std::runtime_error("Qwen progressive expert read failed");
        batch.pending[index] = false;
        --batch.remaining;
        return batch.ranks[index];
    }

    void finish_batch(AsyncBatch& batch) {
        if (!batch.active || batch.remaining)
            throw std::runtime_error("Qwen progressive batch is incomplete");
        for (uint32_t i = 0; i < batch.count; ++i)
            if (batch.events[i]) CloseHandle(batch.events[i]);
        batch = {};
    }

private:
    std::filesystem::path path_;
    ExpertHeader header_{};
    HANDLE file_ = INVALID_HANDLE_VALUE;
};

class HostExpertCache {
public:
    struct Batch {
        std::array<const uint8_t*, kTopK> pointers{};
        std::array<bool, kTopK> direct{};
        uint32_t disk_reads = 0;
    };
    struct ProgressiveBatch {
        Batch sources{};
        ExpertFile::AsyncBatch reads{};
        std::array<bool, kTopK> disk_pending{};
        bool active = false;
    };

    HostExpertCache(ExpertFile& file, uint64_t budget_bytes) : file_(file) {
        tiny_lfu_ = std::getenv("NEMOTRON3_HOST_TINYLFU") != nullptr;
        const uint64_t reserve = budget_bytes > 512ull * 1024 * 1024
            ? budget_bytes - 512ull * 1024 * 1024 : 0;
        slots_ = static_cast<uint32_t>(std::min<uint64_t>(
            reserve / kExpertRecordBytes, uint64_t(kMoeLayers) * kExperts));
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen RAM budget is too small");
        bytes_ = uint64_t(slots_) * kExpertRecordBytes;
        base_ = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, bytes_, MEM_RESERVE, PAGE_READWRITE));
        if (!base_) throw std::runtime_error("Could not reserve Qwen RAM cache");
        entries_.resize(slots_);
        locations_.assign(uint64_t(kMoeLayers) * kExperts, -1);
        frequency_.assign(locations_.size(), 0);
    }

    ~HostExpertCache() {
        if (base_) VirtualFree(base_, 0, MEM_RELEASE);
    }

    Batch resolve_batch(uint32_t layer,
                        const std::array<uint32_t, kTopK>& experts,
                        const std::array<bool, kTopK>& needed,
                        const std::array<void*, kTopK>& direct_destinations) {
        return resolve_batch_impl(layer, experts, needed, direct_destinations,
                                  nullptr, nullptr);
    }

    void begin_progressive(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        const std::array<bool, kTopK>& needed,
        const std::array<void*, kTopK>& direct_destinations,
        ProgressiveBatch& batch) {
        if (batch.active)
            throw std::runtime_error("Nested Qwen progressive acquisition");
        batch = {};
        batch.sources = resolve_batch_impl(
            layer, experts, needed, direct_destinations, &batch.reads,
            &batch.disk_pending);
        batch.active = true;
    }

    uint32_t wait_next_disk(ProgressiveBatch& batch) {
        if (!batch.active || !batch.reads.remaining)
            throw std::runtime_error("No Qwen progressive disk rank pending");
        const uint32_t rank = file_.wait_any(batch.reads);
        if (rank >= kTopK || !batch.disk_pending[rank])
            throw std::runtime_error("Qwen progressive rank completion drift");
        batch.disk_pending[rank] = false;
        return rank;
    }

    void finish_progressive(ProgressiveBatch& batch) {
        if (!batch.active) return;
        file_.finish_batch(batch.reads);
        batch.active = false;
    }

    uint32_t fill_remaining_uniform(double& seconds) {
        const auto started = std::chrono::steady_clock::now();
        constexpr uint32_t kPermutationMultiplier = 53;
        constexpr uint32_t kLayerOffset = 29;
        const uint32_t total_keys = kMoeLayers * kExperts;
        uint32_t candidate = 0;
        uint32_t added = 0;
        std::vector<uint64_t> offsets;
        std::vector<void*> destinations;
        std::vector<uint32_t> pending_slots;
        std::vector<uint32_t> pending_keys;
        offsets.reserve(kTopK);
        destinations.reserve(kTopK);
        pending_slots.reserve(kTopK);
        pending_keys.reserve(kTopK);

        auto flush = [&]() {
            if (offsets.empty()) return;
            file_.read_batch(offsets, destinations);
            for (size_t i = 0; i < pending_slots.size(); ++i) {
                Entry& entry = entries_[pending_slots[i]];
                entry.key = static_cast<int32_t>(pending_keys[i]);
                entry.age = ++clock_;
                locations_[pending_keys[i]] =
                    static_cast<int32_t>(pending_slots[i]);
                ++added;
            }
            offsets.clear();
            destinations.clear();
            pending_slots.clear();
            pending_keys.clear();
        };

        for (uint32_t slot = 0; slot < slots_; ++slot) {
            if (entries_[slot].key >= 0) continue;
            uint32_t key = UINT32_MAX;
            while (candidate < total_keys) {
                const uint32_t layer = candidate % kMoeLayers;
                const uint32_t round = candidate / kMoeLayers;
                ++candidate;
                const uint32_t expert =
                    (round * kPermutationMultiplier + layer * kLayerOffset) %
                    kExperts;
                const uint32_t proposed = layer * kExperts + expert;
                if (locations_[proposed] < 0) {
                    key = proposed;
                    break;
                }
            }
            if (key == UINT32_MAX)
                throw std::runtime_error(
                    "Nemotron RAM-cache top-off exhausted keys");
            uint8_t* pointer = base_ + uint64_t(slot) * kExpertRecordBytes;
            if (!entries_[slot].committed) {
                if (!VirtualAlloc(pointer, kExpertRecordBytes, MEM_COMMIT,
                                  PAGE_READWRITE))
                    throw std::runtime_error(
                        "Nemotron RAM-cache top-off commit failed");
                entries_[slot].committed = true;
                ++committed_;
            }
            const uint32_t layer = key / kExperts;
            const uint32_t expert = key % kExperts;
            offsets.push_back(file_.offset(layer, expert));
            destinations.push_back(pointer);
            pending_slots.push_back(slot);
            pending_keys.push_back(key);
            if (offsets.size() == kTopK) flush();
        }
        flush();
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return added;
    }

    Batch resolve_batch_impl(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        const std::array<bool, kTopK>& needed,
        const std::array<void*, kTopK>& direct_destinations,
        ExpertFile::AsyncBatch* async,
        std::array<bool, kTopK>* disk_pending) {
        Batch result{};
        std::vector<bool> reserved(slots_);
        std::vector<uint64_t> offsets;
        std::vector<void*> destinations;
        std::vector<uint32_t> read_ranks;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (!needed[rank]) continue;
            const uint32_t expert = experts[rank];
            const uint32_t key = layer * kExperts + expert;
            ++frequency_[key];
            const int32_t location = locations_[key];
            if (location >= 0) {
                ++hits_;
                entries_[location].age = ++clock_;
                reserved[location] = true;
                result.pointers[rank] =
                    base_ + uint64_t(location) * kExpertRecordBytes;
                continue;
            }
            ++misses_;
            // First-touch records execute from the already budgeted rank
            // staging buffer but do not evict a prompt-established RAM entry.
            // A second authoritative request admits the record normally.
            if (tiny_lfu_ && committed_ == slots_ && frequency_[key] == 1u) {
                ++admission_bypasses_;
                result.pointers[rank] = nullptr;
                result.direct[rank] = true;
                offsets.push_back(file_.offset(layer, expert));
                destinations.push_back(direct_destinations[rank]);
                read_ranks.push_back(rank);
                continue;
            }
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (reserved[slot]) continue;
                if (victim == UINT32_MAX || entries_[slot].key < 0 ||
                    (entries_[victim].key >= 0 &&
                     (frequency_[entries_[slot].key] <
                          frequency_[entries_[victim].key] ||
                      (frequency_[entries_[slot].key] ==
                           frequency_[entries_[victim].key] &&
                       entries_[slot].age < entries_[victim].age)))) {
                    victim = slot;
                    if (entries_[slot].key < 0) break;
                }
            }
            if (victim == UINT32_MAX)
                throw std::runtime_error("No Qwen RAM-cache victim");
            reserved[victim] = true;
            Entry& entry = entries_[victim];
            if (entry.key >= 0) locations_[entry.key] = -1;
            uint8_t* pointer = base_ + uint64_t(victim) * kExpertRecordBytes;
            if (!entry.committed) {
                if (!VirtualAlloc(pointer, kExpertRecordBytes, MEM_COMMIT,
                                  PAGE_READWRITE))
                    throw std::runtime_error("Qwen RAM-cache commit failed");
                entry.committed = true;
                ++committed_;
            }
            entry.key = static_cast<int32_t>(key);
            entry.age = ++clock_;
            locations_[key] = static_cast<int32_t>(victim);
            result.pointers[rank] = pointer;
            result.direct[rank] = true;
            offsets.push_back(file_.offset(layer, expert));
            destinations.push_back(direct_destinations[rank]);
            read_ranks.push_back(rank);
        }
        if (async) {
            file_.begin_batch(offsets, destinations, read_ranks, *async);
            if (disk_pending)
                for (uint32_t rank : read_ranks) (*disk_pending)[rank] = true;
        } else {
            file_.read_batch(offsets, destinations);
        }
        result.disk_reads = static_cast<uint32_t>(offsets.size());
        disk_bytes_ += uint64_t(offsets.size()) * kExpertRecordBytes;
        return result;
    }

    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t admission_bypasses() const { return admission_bypasses_; }
    uint64_t disk_bytes() const { return disk_bytes_; }
    uint64_t committed_bytes() const {
        return uint64_t(committed_) * kExpertRecordBytes;
    }
    uint32_t capacity() const { return slots_; }
    void reset_metrics() {
        hits_ = misses_ = disk_bytes_ = admission_bypasses_ = 0;
    }

private:
    struct Entry { int32_t key = -1; uint64_t age = 0; bool committed = false; };
    ExpertFile& file_;
    uint8_t* base_ = nullptr;
    uint64_t bytes_ = 0;
    uint32_t slots_ = 0, committed_ = 0;
    std::vector<Entry> entries_;
    std::vector<int32_t> locations_;
    std::vector<uint32_t> frequency_;
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, disk_bytes_ = 0;
    uint64_t admission_bypasses_ = 0;
    bool tiny_lfu_ = false;
};

class DeviceExpertCache {
public:
    DeviceExpertCache(const Runtime& runtime, uint32_t slots)
        : runtime_(runtime), slots_(slots) {
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen device expert cache is too small");
        layers_.resize(kMoeLayers);
        for (Layer& layer : layers_) {
            layer.arena = create_device_buffer(
                runtime, uint64_t(slots_) * kExpertRecordBytes);
            layer.entries.resize(slots_);
            device_bytes_ += layer.arena.allocation_size;
        }
    }

    ~DeviceExpertCache() {
        for (Layer& layer : layers_) destroy_buffer(runtime_, layer.arena);
    }

    struct Selection {
        std::array<uint32_t, kTopK> slots{};
        std::array<bool, kTopK> misses{};
    };

    Selection resolve(uint32_t layer_index,
                      const std::array<uint32_t, kTopK>& experts) {
        Layer& layer = layers_.at(layer_index);
        Selection result{};
        std::vector<bool> reserved(slots_);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            ++layer.frequency[experts[rank]];
            result.slots[rank] = UINT32_MAX;
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (layer.entries[slot].expert == int32_t(experts[rank])) {
                    result.slots[rank] = slot;
                    reserved[slot] = true;
                    layer.entries[slot].age = ++clock_;
                    ++hits_;
                    break;
                }
            }
        }
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (result.slots[rank] != UINT32_MAX) continue;
            ++misses_;
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (reserved[slot]) continue;
                if (victim == UINT32_MAX || layer.entries[slot].expert < 0 ||
                    (layer.entries[victim].expert >= 0 &&
                     (layer.frequency[layer.entries[slot].expert] <
                          layer.frequency[layer.entries[victim].expert] ||
                      (layer.frequency[layer.entries[slot].expert] ==
                           layer.frequency[layer.entries[victim].expert] &&
                       layer.entries[slot].age < layer.entries[victim].age)))) {
                    victim = slot;
                    if (layer.entries[slot].expert < 0) break;
                }
            }
            if (victim == UINT32_MAX)
                throw std::runtime_error("No Qwen device-cache victim");
            layer.entries[victim].expert = int32_t(experts[rank]);
            layer.entries[victim].age = ++clock_;
            reserved[victim] = true;
            result.slots[rank] = victim;
            result.misses[rank] = true;
        }
        return result;
    }

    DescriptorRange record(uint32_t layer, uint32_t slot) const {
        return arena_range(layers_.at(layer).arena,
                           uint64_t(slot) * kExpertRecordBytes,
                           kExpertRecordBytes);
    }
    Buffer& arena(uint32_t layer) { return layers_.at(layer).arena; }
    uint32_t slots() const { return slots_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t device_bytes() const { return device_bytes_; }
    void reset_metrics() { hits_ = misses_ = 0; }

private:
    struct Entry { int32_t expert = -1; uint64_t age = 0; };
    struct Layer {
        Buffer arena{};
        std::vector<Entry> entries;
        std::array<uint32_t, kExperts> frequency{};
    };
    const Runtime& runtime_;
    uint32_t slots_;
    std::vector<Layer> layers_;
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, device_bytes_ = 0;
};

struct Pipelines {
    VkPipeline embedding{}, rms{}, quant{}, q4{}, q4_residual{}, q8{};
    VkPipeline nvfp4{}, nvfp4_residual{};
    VkPipeline relu2{}, router{}, expert_gate{}, expert_down{}, reduce{};
    VkPipeline expert_gate_batch{}, expert_down_batch{};
    VkPipeline store_value{}, attention{};
#ifdef OVLLM_LONG_CONTEXT_FORK
    VkPipeline attention_reduce{};
#endif
    VkPipeline conv{}, mamba_update{}, mamba_norm{}, argmax{};
};

class Kernels {
public:
    Kernels(const Runtime& runtime, const std::filesystem::path& directory)
        : runtime_(runtime), resources_(create_compute_resources(runtime, 20000)),
          dummy_(create_device_buffer(runtime, 4096)) {
        const auto load = [&](const char* name) {
            VkPipeline pipeline = dsv4::create_dsv4_pipeline(
                runtime_, resources_, directory / (std::string(name) + ".comp.spv"), 64);
            return pipeline;
        };
        pipelines_.embedding = load("dsv4_embedding");
        pipelines_.rms = load("nemotron3_rmsnorm");
        pipelines_.quant = load("dsv4_quantize_q8");
        pipelines_.q4 = load("dsv4_q4g64t_gemv");
        pipelines_.q4_residual = load("dsv4_q4g64t_gemv_residual");
        pipelines_.nvfp4 = load("nemotron3_nvfp4_gemv");
        pipelines_.nvfp4_residual = load("nemotron3_nvfp4_gemv_residual");
        pipelines_.q8 = load("dsv4_q8_gemv");
        pipelines_.relu2 = load("nemotron3_relu2");
        pipelines_.router = load("nemotron3_router_top6");
        pipelines_.expert_gate = load("nemotron3_expert_up_nvfp4");
        pipelines_.expert_down = load("nemotron3_expert_down_nvfp4");
        // Qwen3.6 first-pass benchmark uses the finite per-rank path.  Keep
        // aliases so the retained descriptor ABI remains intact without
        // requiring a second, architecture-specific BDA kernel variant.
        pipelines_.expert_gate_batch = pipelines_.expert_gate;
        pipelines_.expert_down_batch = pipelines_.expert_down;
        pipelines_.reduce = load("nemotron3_reduce");
#ifdef OVLLM_LONG_CONTEXT_FORK
        pipelines_.store_value = load("nemotron3_long_attention_store");
        pipelines_.attention = load("nemotron3_long_attention_partial");
        pipelines_.attention_reduce = load("nemotron3_long_attention_reduce");
#else
        pipelines_.store_value = load("nemotron3_attention_store");
        pipelines_.attention = load("nemotron3_attention");
#endif
        pipelines_.conv = load("nemotron3_mamba_conv");
        pipelines_.mamba_update = load("nemotron3_mamba_update");
        pipelines_.mamba_norm = load("nemotron3_mamba_gated_norm");
        pipelines_.argmax = load("qwen35_greedy_argmax");
    }

    ~Kernels() {
        for (VkPipeline pipeline : resources_.pipelines)
            vkfn::DestroyPipeline(runtime_.device, pipeline, nullptr);
        for (VkShaderModule module : resources_.shader_modules)
            vkfn::DestroyShaderModule(runtime_.device, module, nullptr);
        if (resources_.descriptor_pool)
            vkfn::DestroyDescriptorPool(runtime_.device, resources_.descriptor_pool,
                                        nullptr);
        if (resources_.pipeline_layout)
            vkfn::DestroyPipelineLayout(runtime_.device, resources_.pipeline_layout,
                                        nullptr);
        if (resources_.descriptor_layout)
            vkfn::DestroyDescriptorSetLayout(runtime_.device,
                                             resources_.descriptor_layout, nullptr);
        destroy_buffer(runtime_, dummy_);
    }

    VkDescriptorSet set(std::initializer_list<DescriptorRange> ranges) {
        std::array<DescriptorRange, 6> filled;
        filled.fill(whole(dummy_));
        uint32_t index = 0;
        for (const DescriptorRange& range : ranges) {
            if (index >= filled.size())
                throw std::runtime_error("Too many Qwen shader descriptors");
            filled[index++] = range;
        }
        return dsv4::create_dsv4_set(runtime_, resources_, filled);
    }

    void dispatch(VkCommandBuffer command, VkPipeline pipeline,
                  VkDescriptorSet set, const void* push, uint32_t x,
                  uint32_t y = 1) {
        dsv4::dispatch_dsv4(command, resources_, pipeline, set, push, x, y);
    }

    const Pipelines& p() const { return pipelines_; }
    DescriptorRange dummy() const { return whole(dummy_); }

private:
    const Runtime& runtime_;
    ComputeResources resources_{};
    Buffer dummy_{};
    Pipelines pipelines_{};
};

struct Push { uint32_t a, b, c, d; };

#ifdef OVLLM_LONG_CONTEXT_FORK
static uint32_t requested_context_tokens() {
    if (const char* exact = std::getenv("NEMOTRON3_CONTEXT_TOKENS")) {
        const uint64_t value = std::stoull(exact);
        if (value < 128 || value > UINT32_MAX)
            throw std::runtime_error("NEMOTRON3_CONTEXT_TOKENS must be 128..2^32-1");
        return static_cast<uint32_t>(value);
    }
    if (const char* text = std::getenv("NEMOTRON3_CONTEXT_GIB")) {
        const double gib = std::stod(text);
        if (gib < 0.05 || gib > 16.0)
            throw std::runtime_error("NEMOTRON3_CONTEXT_GIB must be 0.05..16");
        const uint64_t bytes = static_cast<uint64_t>(gib * double(1ull << 30));
        const uint64_t per_token = uint64_t(kFullLayers) * 2u * kKvHeads *
            kHeadDim * sizeof(uint16_t);
        return static_cast<uint32_t>(bytes / per_token);
    }
    return kDefaultContext;
}
#endif

class QwenEngine {
public:
    QwenEngine(const Runtime& runtime, const SharedIndex& index,
               const std::filesystem::path& expert_path,
               const std::filesystem::path& shader_directory,
               uint64_t ram_budget, uint32_t device_slots)
        : runtime_(runtime), weights_(runtime, index), expert_file_(expert_path),
          host_cache_(expert_file_, ram_budget),
          device_cache_(runtime, device_slots),
          kernels_(runtime, shader_directory), compute_(runtime, runtime.queue),
          transfer_(runtime, runtime.secondary_queue) {
#ifdef OVLLM_LONG_CONTEXT_FORK
        max_context_ = requested_context_tokens();
        attention_chunks_ = (max_context_ + kAttentionChunk - 1u) /
            kAttentionChunk;
#endif
        batch_experts_ = false;
        progressive_experts_ =
            std::getenv("NEMOTRON3_NO_PROGRESSIVE") == nullptr;
        if (progressive_experts_)
            progressive_compute_ = std::make_unique<
                dsv4::experiment::FiniteQueueRing<12>>(
                    runtime_.device, runtime_.queue, runtime_.queue_family,
                    dsv4::finite_queue_ring_api());
        allocate_buffers();
        initialize_persistent_buffers();
        make_rope();
        build_sets();
        staging_.resize(kTopK);
        for (Buffer& buffer : staging_)
            buffer = dsv4::create_host_buffer_uninitialized(runtime_,
                                                             kExpertRecordBytes);
    }

    ~QwenEngine() {
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        destroy_all();
    }

    std::vector<uint32_t> generate(const Tokenizer& tokenizer,
                                   const std::vector<uint32_t>& prompt,
                                   uint32_t count) {
        if (prompt.empty()) throw std::runtime_error("Qwen prompt is empty");
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (uint64_t(prompt.size()) + count > max_context_)
            throw std::runtime_error("Nemotron prompt plus generation exceeds configured context");
#endif
        uint32_t position = 0;
        uint32_t next = 0;
        for (uint32_t token : prompt) next = run_token(token, position++);
        if (std::getenv("NEMOTRON3_FILL_RAM_CACHE")) {
            double fill_seconds = 0.0;
            const uint32_t filled =
                host_cache_.fill_remaining_uniform(fill_seconds);
            std::cout << "RAM cache top-off: " << filled << " records, "
                      << double(host_cache_.committed_bytes()) /
                             double(1ull << 30)
                      << " GiB cache, " << fill_seconds << " s\n";
        }
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (std::getenv("NEMOTRON3_LONG_CONTEXT_STRESS")) {
            if (max_context_ <= count + 2u)
                throw std::runtime_error("Nemotron context is too small for stress decode");
            const uint32_t target = max_context_ - count;
            expand_context_for_stress(prompt.size() - 1u, target);
            position = target;
            std::cout << "Nemotron long-context stress positions: " << target
                      << ".." << (max_context_ - 1u) << "\n";
        }
#endif
        reset_decode_metrics();
        const auto started = std::chrono::steady_clock::now();
        std::vector<uint32_t> output;
        for (uint32_t i = 0; i < count; ++i) {
            output.push_back(next);
            std::cout << tokenizer.decode_piece(next) << std::flush;
            if (tokenizer.is_eos(next) || i + 1 == count) break;
            next = run_token(next, position++);
        }
        decode_seconds_ = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return output;
    }

    uint64_t device_hits() const { return device_cache_.hits(); }
    uint64_t device_misses() const { return device_cache_.misses(); }
    uint64_t ram_hits() const { return host_cache_.hits(); }
    uint64_t ram_misses() const { return host_cache_.misses(); }
    uint64_t ram_admission_bypasses() const {
        return host_cache_.admission_bypasses();
    }
    uint64_t disk_bytes() const { return host_cache_.disk_bytes(); }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
    uint64_t host_copy_bytes() const { return host_copy_bytes_; }
    uint64_t cold_stalled_layers() const { return cold_stalled_layers_; }
    uint64_t progressive_layers() const { return progressive_layers_; }
    uint64_t progressive_resident_ranks() const {
        return progressive_resident_ranks_;
    }
    uint64_t progressive_ram_ranks() const { return progressive_ram_ranks_; }
    uint64_t progressive_disk_ranks() const { return progressive_disk_ranks_; }
    uint64_t ram_bytes() const {
        return host_cache_.committed_bytes() +
               uint64_t(staging_.size()) * kExpertRecordBytes
#ifdef OVLLM_LONG_CONTEXT_FORK
               + kv_cache_.allocation_size
#endif
               ;
    }
    uint64_t vram_bytes() const {
        return weights_.device_bytes() + device_cache_.device_bytes() +
               activation_device_bytes_;
    }
    uint32_t host_slots() const { return host_cache_.capacity(); }
    uint32_t device_slots() const { return device_cache_.slots(); }
    double decode_seconds() const { return decode_seconds_; }
    double pre_seconds() const { return pre_seconds_; }
    double acquisition_seconds() const { return acquisition_seconds_; }
    double expert_seconds() const { return expert_seconds_; }

private:

    void trace_hidden(uint32_t layer) {
        if (!diagnostic_.handle) return;
        const VkDeviceSize bytes = uint64_t(kDim) * sizeof(float);
        const uint64_t copied = compute_.submit([&](VkCommandBuffer command) {
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = hidden_.handle;
            barrier.size = bytes;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &barrier, 0, nullptr);
            VkBufferCopy copy{0, 0, bytes};
            vkfn::CmdCopyBuffer(command, hidden_.handle, diagnostic_.handle, 1,
                                &copy);
        });
        compute_.wait(copied);
        invalidate_buffer(runtime_, diagnostic_);
        const float* values = static_cast<const float*>(diagnostic_.mapped);
        double squares = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < kDim; ++i) {
            const float value = values[i];
            if (!std::isfinite(value)) { ++nonfinite; continue; }
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            squares += double(value) * value;
        }
        std::cerr << "hidden layer " << layer << " rms="
                  << std::sqrt(squares / double(kDim)) << " min=" << minimum
                  << " max=" << maximum << " nonfinite=" << nonfinite << '\n';
        if (mamba_layer(layer)) {
            trace_values(z_, kMambaIntermediate, "mamba-y", layer);
            trace_values(mixed_qkv_, kMambaIntermediate, "mamba-gate", layer);
            trace_values(context_, kMambaIntermediate, "mamba-context", layer);
        }
    }

    void trace_values(const Buffer& source, uint32_t count, const char* label,
                      uint32_t layer) {
        if (!diagnostic_.handle) return;
        const VkDeviceSize bytes = uint64_t(count) * sizeof(float);
        const uint64_t copied = compute_.submit([&](VkCommandBuffer command) {
            VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = source.handle;
            barrier.size = bytes;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &barrier, 0, nullptr);
            VkBufferCopy copy{0, 0, bytes};
            vkfn::CmdCopyBuffer(command, source.handle, diagnostic_.handle, 1,
                                &copy);
        });
        compute_.wait(copied);
        invalidate_buffer(runtime_, diagnostic_);
        const float* values = static_cast<const float*>(diagnostic_.mapped);
        double squares = 0.0;
        float minimum = std::numeric_limits<float>::infinity();
        float maximum = -std::numeric_limits<float>::infinity();
        uint32_t nonfinite = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const float value = values[i];
            if (!std::isfinite(value)) { ++nonfinite; continue; }
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            squares += double(value) * value;
        }
        std::cerr << label << " layer " << layer << " rms="
                  << std::sqrt(squares / double(count)) << " min=" << minimum
                  << " max=" << maximum << " nonfinite=" << nonfinite << '\n';
    }

    Buffer device(uint64_t bytes) {
        Buffer buffer = create_device_buffer(runtime_, bytes);
        activation_device_bytes_ += buffer.allocation_size;
        return buffer;
    }

    void allocate_buffers() {
        token_ = create_buffer(runtime_, sizeof(uint32_t));
        // The router owns words 0..15.  Words 16..31 are an optional compact
        // table of eight device addresses for the batched Q4 expert kernels.
        routing_ = create_buffer(runtime_, 32u * sizeof(uint32_t));
        hidden_ = device(uint64_t(kDim) * 4);
        normalized_ = device(uint64_t(kDim) * 4);
        qgate_ = device(uint64_t(kAttentionHeads) * kHeadDim * 4);
        key_ = device(uint64_t(kKvHeads) * kHeadDim * 4);
        value_ = device(uint64_t(kKvHeads) * kHeadDim * 4);
        context_ = device(uint64_t(kMambaIntermediate) * 4);
        mixed_qkv_ = device(uint64_t(kMambaInput) * 4);
        convolved_qkv_ = device(uint64_t(kMambaConvChannels) * 4);
        z_ = device(uint64_t(kMambaIntermediate) * 4);
        ab_ = device(4);
        // The largest dense activation is the 4096-wide Mamba/attention
        // context: 1024 packed Q8 words plus 32 K128 scale words.
        quant_ = device(uint64_t(kMambaIntermediate / 4 +
                                 kMambaIntermediate / 128) * 4);
        shared_gate_values_ = device(4);
        shared_up_values_ = device(uint64_t(kSharedDim) * 4);
        shared_intermediate_ = device(uint64_t(kSharedDim) * 4);
        shared_output_ = device(kDim * 4ull);
        shared_expert_gate_ = device(4);
        router_logits_ = device(kExperts * 4ull);
        expert_intermediate_ = device(uint64_t(kTopK) * kMoeDim * 4);
        expert_quant_ = device(uint64_t(kTopK) * (kMoeDim / 4u + (kMoeDim + 127u) / 128u) * 4);
        expert_outputs_ = device(uint64_t(kTopK) * kDim * 4);
        logits_ = device(uint64_t(kVocabulary) * 4);
        argmax_workspace_ = device(512ull * 4);
        conv_state_ = device(uint64_t(kMambaLayers) * kMambaConvChannels *
                             kConvWidth * 4);
        recurrent_state_ = device(uint64_t(kMambaLayers) *
                                  kMambaHeads * kMambaHeadDim * kMambaState * 4);
#ifdef OVLLM_LONG_CONTEXT_FORK
        const uint64_t kv_bytes = uint64_t(kFullLayers) * 2u * max_context_ *
            kKvHeads * kHeadDim * sizeof(uint16_t);
        kv_cache_ = dsv4::create_host_buffer_uninitialized(runtime_, kv_bytes);
        attention_partial_ = device(uint64_t(attention_chunks_) *
            kAttentionHeads * kAttentionPartialStride * sizeof(float));
#else
        kv_cache_ = device(uint64_t(kFullLayers) * 2 * kMaximumContext *
                           kKvHeads * kHeadDim * 4);
#endif
        rope_ = device(4096);
    }

    void initialize_persistent_buffers() {
        constexpr uint64_t chunk_bytes = 64ull * 1024 * 1024;
        Buffer zeros = dsv4::create_host_buffer_uninitialized(runtime_, chunk_bytes);
        std::memset(zeros.mapped, 0, static_cast<size_t>(chunk_bytes));
        dsv4::flush_buffer_range(runtime_, zeros, 0, chunk_bytes);
        for (Buffer* buffer : {&conv_state_, &recurrent_state_
#ifndef OVLLM_LONG_CONTEXT_FORK
                               , &kv_cache_
#endif
                               }) {
            for (uint64_t offset = 0; offset < buffer->size;
                 offset += chunk_bytes) {
                const uint64_t bytes =
                    std::min<uint64_t>(chunk_bytes, buffer->size - offset);
                const uint64_t signal = compute_.submit(
                    [&](VkCommandBuffer command) {
                        VkBufferCopy copy{0, offset, bytes};
                        vkfn::CmdCopyBuffer(command, zeros.handle, buffer->handle,
                                            1, &copy);
                        dsv4::transfer_barrier(command, *buffer);
                    });
                compute_.wait(signal);
            }
        }
        destroy_buffer(runtime_, zeros);
    }

    void make_rope() {}

    struct LayerSets {
        VkDescriptorSet norm{}, hidden_quant{};
        VkDescriptorSet qgate{}, key{}, value{}, store_value{};
        VkDescriptorSet attention{}, context_quant{}, attention_out{};
#ifdef OVLLM_LONG_CONTEXT_FORK
        VkDescriptorSet attention_reduce{};
#endif
        VkDescriptorSet mamba_in{}, conv{}, mamba_update{}, mamba_norm{}, mamba_out{};
        VkDescriptorSet router_gemv{}, router{};
        VkDescriptorSet shared_up{}, shared_relu2{};
        VkDescriptorSet shared_quant{}, shared_down{};
        std::vector<VkDescriptorSet> expert_gate, expert_down;
        bool q_native=false, k_native=false, v_native=false, attention_out_native=false;
        bool mamba_in_native=false, mamba_out_native=false;
        bool shared_up_native=false, shared_down_native=false;
    };

    TensorDevice tensor(const std::string& name, TensorFormat format,
                        uint64_t first, uint64_t second = 0) const {
        TensorDevice value = weights_.tensor(name);
        const uint32_t expected_rank = second ? 2u : 1u;
        if (value.format != format || value.rank != expected_rank ||
            value.shape[0] != first || (second && value.shape[1] != second))
            throw std::runtime_error("Unexpected Qwen tensor ABI: " + name);
        return value;
    }

    TensorDevice matrix_tensor(const std::string& name, uint64_t rows,
                               uint64_t columns) const {
        TensorDevice value = weights_.tensor(name);
        if ((value.format != TensorFormat::q4g64t &&
             value.format != TensorFormat::nvfp4_bf16) ||
            value.rank != 2 || value.shape[0] != rows ||
            value.shape[1] != columns)
            throw std::runtime_error("Unexpected Nemotron matrix ABI: " + name);
        return value;
    }

    VkDescriptorSet q4_set(DescriptorRange activation, const TensorDevice& weight,
                           DescriptorRange output,
                           DescriptorRange residual = {}) {
        return kernels_.set({activation, weight.data, weight.auxiliary, output,
                             residual.buffer ? residual : kernels_.dummy()});
    }

    void build_sets() {
        const TensorDevice embedding =
            tensor("embed", TensorFormat::q8_row, kVocabulary, kDim);
        const TensorDevice final_norm =
            tensor("final_norm", TensorFormat::f32, kDim);
        const TensorDevice lm_head =
            tensor("lm_head", TensorFormat::q8_row, kVocabulary, kDim);
        embedding_set_ = kernels_.set(
            {embedding.data, embedding.auxiliary, whole(token_), whole(hidden_)});
        final_norm_set_ = kernels_.set(
            {whole(hidden_), final_norm.data, whole(normalized_)});
        final_quant_set_ = kernels_.set({whole(normalized_), whole(quant_)});
        lm_head_set_ = kernels_.set(
            {whole(quant_), lm_head.data, lm_head.auxiliary, whole(logits_)});
        argmax_set_ = kernels_.set(
            {whole(logits_), whole(token_), whole(argmax_workspace_)});
        expert_quant_set_ = kernels_.set({whole(expert_intermediate_), whole(expert_quant_)});
        const uint64_t packed_words = kMoeDim / 4u;
        const uint64_t scale_words = (kMoeDim + 127u) / 128u;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint64_t input_offset =
                uint64_t(rank) * kMoeDim * sizeof(float);
            const uint64_t output_offset = uint64_t(rank) * packed_words * 4;
            expert_rank_quant_sets_[rank] = kernels_.set({
                arena_range(expert_intermediate_, input_offset,
                            uint64_t(kMoeDim) * sizeof(float)),
                arena_range(expert_quant_, output_offset,
                            expert_quant_.size - output_offset)});
        }
        reduce_set_ = kernels_.set({whole(expert_outputs_), whole(shared_output_), whole(hidden_)});

        layers_.resize(kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            LayerSets& sets = layers_[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            const TensorDevice norm = tensor(prefix + "norm", TensorFormat::f32, kDim);
            sets.norm = kernels_.set({whole(hidden_), norm.data, whole(normalized_)});
            sets.hidden_quant = kernels_.set({whole(normalized_), whole(quant_)});

            if (attention_layer(layer)) {
                const TensorDevice query = matrix_tensor(prefix + "q_proj",
                    kAttentionHeads * kHeadDim, kDim);
                const TensorDevice key = matrix_tensor(prefix + "k_proj",
                    kKvHeads * kHeadDim, kDim);
                const TensorDevice value = matrix_tensor(prefix + "v_proj",
                    kKvHeads * kHeadDim, kDim);
                const TensorDevice output = matrix_tensor(prefix + "o_proj",
                    kDim, kAttentionHeads * kHeadDim);
                sets.q_native = query.format == TensorFormat::nvfp4_bf16;
                sets.k_native = key.format == TensorFormat::nvfp4_bf16;
                sets.v_native = value.format == TensorFormat::nvfp4_bf16;
                sets.attention_out_native = output.format == TensorFormat::nvfp4_bf16;
                sets.qgate = q4_set(whole(quant_), query, whole(qgate_));
                sets.key = q4_set(whole(quant_), key, whole(key_));
                sets.value = q4_set(whole(quant_), value, whole(value_));
#ifdef OVLLM_LONG_CONTEXT_FORK
                const uint64_t layer_cache_bytes = uint64_t(max_context_) * 2u *
                    kKvHeads * kHeadDim * sizeof(uint16_t);
                const DescriptorRange layer_cache = arena_range(kv_cache_,
                    uint64_t(attention_index(layer)) * layer_cache_bytes,
                    layer_cache_bytes);
                sets.store_value = kernels_.set(
                    {whole(qgate_), whole(key_), whole(value_), layer_cache});
                sets.attention = kernels_.set(
                    {whole(qgate_), layer_cache, whole(attention_partial_)});
                sets.attention_reduce = kernels_.set(
                    {whole(attention_partial_), whole(context_)});
#else
                sets.store_value = kernels_.set({whole(qgate_), whole(key_), whole(value_), whole(kv_cache_)});
                sets.attention = kernels_.set({whole(qgate_), whole(kv_cache_), whole(context_)});
#endif
                sets.context_quant = kernels_.set({whole(context_), whole(quant_)});
                sets.attention_out = q4_set(whole(quant_), output, whole(hidden_),
                                            whole(hidden_));
            } else if (mamba_layer(layer)) {
                const TensorDevice input = matrix_tensor(prefix + "mamba_in",
                                                         kMambaInput, kDim);
                const TensorDevice output = matrix_tensor(prefix + "mamba_out",
                                                          kDim, kMambaIntermediate);
                sets.mamba_in_native = input.format == TensorFormat::nvfp4_bf16;
                sets.mamba_out_native = output.format == TensorFormat::nvfp4_bf16;
                const TensorDevice conv = tensor(prefix + "mamba_conv_weight", TensorFormat::f32,
                                                 kMambaConvChannels, kConvWidth);
                const TensorDevice bias = tensor(prefix + "mamba_conv_bias", TensorFormat::f32,
                                                 kMambaConvChannels);
                const TensorDevice params = tensor(prefix + "mamba_params", TensorFormat::f32,
                                                   kMambaHeads * 3 + kMambaIntermediate);
                const uint64_t linear = mamba_index(layer);
                const DescriptorRange conv_state = arena_range(
                    conv_state_, linear * uint64_t(kMambaConvChannels) * kConvWidth * 4,
                    uint64_t(kMambaConvChannels) * kConvWidth * 4);
                const DescriptorRange recurrent_state = arena_range(
                    recurrent_state_, linear * uint64_t(kMambaHeads) * kMambaHeadDim * kMambaState * 4,
                    uint64_t(kMambaHeads) * kMambaHeadDim * kMambaState * 4);
                sets.mamba_in = q4_set(whole(quant_), input, whole(mixed_qkv_));
                sets.conv = kernels_.set({whole(mixed_qkv_), conv.data, bias.data,
                                          conv_state, whole(convolved_qkv_)});
                sets.mamba_update = kernels_.set({whole(mixed_qkv_), whole(convolved_qkv_),
                                                  recurrent_state, params.data, whole(z_)});
                const DescriptorRange gamma{params.data.buffer,
                    params.data.offset + uint64_t(kMambaHeads * 3) * 4,
                    uint64_t(kMambaIntermediate) * 4};
                sets.mamba_norm = kernels_.set({whole(z_), whole(mixed_qkv_), gamma, whole(context_)});
                sets.context_quant = kernels_.set({whole(context_), whole(quant_)});
                sets.mamba_out = q4_set(whole(quant_), output, whole(hidden_), whole(hidden_));
            } else {
                const TensorDevice router = tensor(prefix + "router", TensorFormat::q8_row,
                                                   kExperts, kDim);
                const TensorDevice correction = tensor(prefix + "correction", TensorFormat::f32,
                                                       kExperts);
                sets.router_gemv = kernels_.set({whole(quant_), router.data, router.auxiliary,
                                                 whole(router_logits_)});
                sets.router = kernels_.set({whole(router_logits_), correction.data, whole(routing_)});
                const TensorDevice shared_up = matrix_tensor(prefix + "shared_up",
                                                             kSharedDim, kDim);
                const TensorDevice shared_down = matrix_tensor(prefix + "shared_down",
                                                               kDim, kSharedDim);
                sets.shared_up_native = shared_up.format == TensorFormat::nvfp4_bf16;
                sets.shared_down_native = shared_down.format == TensorFormat::nvfp4_bf16;
                sets.shared_up = q4_set(whole(quant_), shared_up, whole(shared_up_values_));
                sets.shared_relu2 = kernels_.set({whole(shared_up_values_), whole(shared_intermediate_)});
                sets.shared_quant = kernels_.set({whole(shared_intermediate_), whole(expert_quant_)});
                sets.shared_down = q4_set(whole(expert_quant_), shared_down, whole(shared_output_));
                const uint32_t local = moe_index(layer);
                sets.expert_gate.resize(device_cache_.slots());
                sets.expert_down.resize(device_cache_.slots());
                for (uint32_t slot = 0; slot < device_cache_.slots(); ++slot) {
                    const DescriptorRange record = device_cache_.record(local, slot);
                    sets.expert_gate[slot] = kernels_.set({whole(quant_), record, whole(routing_),
                                                          whole(expert_intermediate_)});
                    sets.expert_down[slot] = kernels_.set({whole(expert_quant_), record,
                                                          whole(routing_), whole(expert_outputs_)});
                }
            }
        }
    }

    void reset_decode_metrics() {
        host_cache_.reset_metrics();
        device_cache_.reset_metrics();
        transfer_bytes_ = 0;
        host_copy_bytes_ = 0;
        cold_stalled_layers_ = 0;
        progressive_layers_ = progressive_resident_ranks_ = 0;
        progressive_ram_ranks_ = progressive_disk_ranks_ = 0;
        pre_seconds_ = acquisition_seconds_ = expert_seconds_ = 0;
    }

#ifdef OVLLM_LONG_CONTEXT_FORK
    void expand_context_for_stress(uint32_t source_position,
                                   uint32_t target_position) {
        if (target_position <= source_position + 1u) return;
        invalidate_buffer(runtime_, kv_cache_);
        const uint64_t token_bytes = uint64_t(kKvHeads) * kHeadDim *
            sizeof(uint16_t);
        const uint64_t plane_bytes = uint64_t(max_context_) * token_bytes;
        const uint64_t layer_bytes = 2u * plane_bytes;
        auto* base = static_cast<uint8_t*>(kv_cache_.mapped);
        for (uint32_t layer = 0; layer < kFullLayers; ++layer) {
            for (uint32_t plane = 0; plane < 2u; ++plane) {
                uint8_t* values = base + uint64_t(layer) * layer_bytes +
                    uint64_t(plane) * plane_bytes;
                const uint32_t begin = source_position + 1u;
                std::memcpy(values + uint64_t(begin) * token_bytes,
                            values + uint64_t(source_position) * token_bytes,
                            token_bytes);
                uint32_t filled = 1u;
                while (begin + filled < target_position) {
                    const uint32_t count = std::min<uint32_t>(
                        filled, target_position - begin - filled);
                    std::memcpy(values + uint64_t(begin + filled) * token_bytes,
                                values + uint64_t(begin) * token_bytes,
                                uint64_t(count) * token_bytes);
                    filled += count;
                }
            }
        }
        dsv4::flush_buffer_range(runtime_, kv_cache_, 0, kv_cache_.size);
    }
#endif

    uint32_t run_token(uint32_t token, uint32_t position) {
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (position >= max_context_)
#else
        if (position >= kMaximumContext)
#endif
            throw std::runtime_error("Nemotron runtime context cap reached");
        *static_cast<uint32_t*>(token_.mapped) = token;
        flush_buffer(runtime_, token_);
        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            Push push{kVocabulary, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().embedding, embedding_set_,
                              &push, (kDim + 63) / 64);
            compute_barrier(command);
        });
        compute_.wait(signal);

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            auto started = std::chrono::steady_clock::now();
            if (!moe_layer(layer)) {
                signal = compute_.submit([&](VkCommandBuffer command) {
                    record_mixer(command, layer, position);
                });
                compute_.wait(signal);
                pre_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                continue;
            }
            signal = compute_.submit([&](VkCommandBuffer command) {
                record_moe_pre(command, layer);
            });
            compute_.wait(signal);
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();

            invalidate_buffer(runtime_, routing_);
            std::array<uint32_t, kTopK> experts{};
            const uint32_t* routes = static_cast<const uint32_t*>(routing_.mapped);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                experts[rank] = routes[rank];
                if (experts[rank] >= kExperts)
                    throw std::runtime_error("Qwen router returned invalid expert");
            }

            const auto expert_started = std::chrono::steady_clock::now();
            compute_.submit([&](VkCommandBuffer command) {
                record_shared(command, layer);
            });
            started = std::chrono::steady_clock::now();
            const uint32_t local = moe_index(layer);
            const DeviceExpertCache::Selection selection =
                device_cache_.resolve(local, experts);
            selected_slots_ = selection.slots;
            std::array<void*, kTopK> direct_destinations{};
            for (uint32_t rank = 0; rank < kTopK; ++rank)
                direct_destinations[rank] = staging_[rank].mapped;
            const bool has_missing = std::any_of(
                selection.misses.begin(), selection.misses.end(),
                [](bool missing) { return missing; });
            if (progressive_experts_ && has_missing) {
                HostExpertCache::ProgressiveBatch batch{};
                host_cache_.begin_progressive(
                    local, experts, selection.misses, direct_destinations,
                    batch);
                if (batch.sources.disk_reads) ++cold_stalled_layers_;

                // Device hits can run immediately behind the already queued
                // shared expert while RAM and NVMe records are prepared.
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (selection.misses[rank]) continue;
                    progressive_compute_->submit([&](VkCommandBuffer command) {
                        record_expert_rank(command, layer, rank);
                    });
                    ++progressive_resident_ranks_;
                }

                std::vector<uint32_t> ram_ranks;
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (!selection.misses[rank] ||
                        batch.sources.direct[rank]) continue;
                    std::memcpy(staging_[rank].mapped,
                                batch.sources.pointers[rank],
                                kExpertRecordBytes);
                    host_copy_bytes_ += kExpertRecordBytes;
                    dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                             kExpertRecordBytes);
                    ram_ranks.push_back(rank);
                }
                uint64_t last_ready = 0;
                if (!ram_ranks.empty()) {
                    last_ready = transfer_.submit([&](VkCommandBuffer command) {
                        for (uint32_t rank : ram_ranks) {
                            const VkBufferCopy copy{
                                0, uint64_t(selected_slots_[rank]) *
                                       kExpertRecordBytes,
                                kExpertRecordBytes};
                            vkfn::CmdCopyBuffer(
                                command, staging_[rank].handle,
                                device_cache_.arena(local).handle, 1, &copy);
                        }
                        dsv4::transfer_barrier(
                            command, device_cache_.arena(local));
                    });
                    transfer_bytes_ +=
                        uint64_t(ram_ranks.size()) * kExpertRecordBytes;
                    for (uint32_t rank : ram_ranks) {
                        progressive_compute_->submit(
                            [&](VkCommandBuffer command) {
                                record_expert_rank(command, layer, rank);
                            }, transfer_.semaphore(), last_ready);
                        ++progressive_ram_ranks_;
                    }
                }

                while (batch.reads.remaining) {
                    const uint32_t rank =
                        host_cache_.wait_next_disk(batch);
                    dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                             kExpertRecordBytes);
                    last_ready = transfer_.submit(
                        [&](VkCommandBuffer command) {
                            const VkBufferCopy copy{
                                0, uint64_t(selected_slots_[rank]) *
                                       kExpertRecordBytes,
                                kExpertRecordBytes};
                            vkfn::CmdCopyBuffer(
                                command, staging_[rank].handle,
                                device_cache_.arena(local).handle, 1, &copy);
                            dsv4::transfer_barrier(
                                command, device_cache_.arena(local));
                        });
                    transfer_bytes_ += kExpertRecordBytes;
                    progressive_compute_->submit(
                        [&](VkCommandBuffer command) {
                            record_expert_rank(command, layer, rank);
                        }, transfer_.semaphore(), last_ready);
                    ++progressive_disk_ranks_;
                }
                host_cache_.finish_progressive(batch);
                acquisition_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();

                const uint64_t finished = progressive_compute_->submit(
                    [&](VkCommandBuffer command) {
                        record_expert_finish(command);
                    });
                // Once the last H2D is complete, populate admitted RAM slots
                // while the finite compute queue finishes late ranks/reduce.
                if (last_ready) transfer_.wait(last_ready);
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (!selection.misses[rank] ||
                        !batch.sources.direct[rank] ||
                        !batch.sources.pointers[rank]) continue;
                    std::memcpy(
                        const_cast<uint8_t*>(batch.sources.pointers[rank]),
                        staging_[rank].mapped, kExpertRecordBytes);
                    host_copy_bytes_ += kExpertRecordBytes;
                }
                progressive_compute_->wait(finished);
                ++progressive_layers_;
                expert_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - expert_started).count();
                continue;
            }
            const HostExpertCache::Batch sources = host_cache_.resolve_batch(
                local, experts, selection.misses, direct_destinations);
            if (sources.disk_reads) ++cold_stalled_layers_;

            std::vector<uint32_t> copied;
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                if (!selection.misses[rank]) continue;
                if (!sources.direct[rank]) {
                    std::memcpy(staging_[rank].mapped, sources.pointers[rank],
                                kExpertRecordBytes);
                    host_copy_bytes_ += kExpertRecordBytes;
                }
                dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                         kExpertRecordBytes);
                copied.push_back(rank);
            }
            uint64_t ready = 0;
            if (!copied.empty()) {
                ready = transfer_.submit([&](VkCommandBuffer command) {
                    for (uint32_t rank : copied) {
                        VkBufferCopy copy{
                            0, uint64_t(selected_slots_[rank]) * kExpertRecordBytes,
                            kExpertRecordBytes};
                        vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                                            device_cache_.arena(local).handle,
                                            1, &copy);
                    }
                    dsv4::transfer_barrier(command, device_cache_.arena(local));
                });
                transfer_bytes_ += uint64_t(copied.size()) * kExpertRecordBytes;
            }
            acquisition_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();

            signal = compute_.submit(
                [&](VkCommandBuffer command) { record_experts(command, layer); },
                ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            bool fill_ram = false;
            for (uint32_t rank : copied)
                fill_ram = fill_ram || sources.direct[rank];
            if (fill_ram) {
                transfer_.wait(ready);
                for (uint32_t rank : copied) {
                    if (!sources.direct[rank] || !sources.pointers[rank]) continue;
                    std::memcpy(const_cast<uint8_t*>(sources.pointers[rank]),
                                staging_[rank].mapped, kExpertRecordBytes);
                    host_copy_bytes_ += kExpertRecordBytes;
                }
            }
            compute_.wait(signal);
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
        }

        signal = compute_.submit([&](VkCommandBuffer command) {
            Push push{1, kDim, float_bits(1e-5f), 0};
            kernels_.dispatch(command, kernels_.p().rms, final_norm_set_, &push, 1);
            compute_barrier(command);
            push = {kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, final_quant_set_, &push,
                              (kDim + 127) / 128);
            compute_barrier(command);
            push = {kVocabulary, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q8, lm_head_set_, &push,
                              (kVocabulary + 7) / 8);
            compute_barrier(command);
            push = {kVocabulary, 256, 0, 0};
            kernels_.dispatch(command, kernels_.p().argmax, argmax_set_, &push, 256);
            compute_barrier(command);
            push = {kVocabulary, 256, 1, 0};
            kernels_.dispatch(command, kernels_.p().argmax, argmax_set_, &push, 1);
            compute_barrier(command);
        });
        compute_.wait(signal);
        invalidate_buffer(runtime_, token_);
        return *static_cast<const uint32_t*>(token_.mapped);
    }

    void record_mixer(VkCommandBuffer command, uint32_t layer,
                      uint32_t position) {
        LayerSets& sets = layers_[layer];
        Push push{1, kDim, float_bits(1e-5f), 0};
        kernels_.dispatch(command, kernels_.p().rms, sets.norm, &push, 1);
        compute_barrier(command);
        push = {kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.hidden_quant, &push,
                          (kDim + 127) / 128);
        compute_barrier(command);

        if (attention_layer(layer)) {
            push = {kAttentionHeads * kHeadDim, kDim, kDim / 4, 0};
            kernels_.dispatch(command, sets.q_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.qgate, &push,
                              (kAttentionHeads * kHeadDim) / 8);
            push = {kKvHeads * kHeadDim, kDim, kDim / 4, 0};
            kernels_.dispatch(command, sets.k_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.key, &push,
                              (kKvHeads * kHeadDim) / 8);
            kernels_.dispatch(command, sets.v_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.value, &push,
                              (kKvHeads * kHeadDim) / 8);
            compute_barrier(command);
#ifdef OVLLM_LONG_CONTEXT_FORK
            push = {position, max_context_, kAttentionHeads, 0};
#else
            push = {attention_index(layer), position, kAttentionHeads, 0};
#endif
            kernels_.dispatch(command, kernels_.p().store_value,
                              sets.store_value, &push,
                              (kKvHeads * kHeadDim + 63) / 64);
            compute_barrier(command);
#ifdef OVLLM_LONG_CONTEXT_FORK
            const uint32_t active_chunks = (position + kAttentionChunk) /
                kAttentionChunk;
            push = {position, max_context_, kAttentionChunk, active_chunks};
            kernels_.dispatch(command, kernels_.p().attention, sets.attention,
                              &push, active_chunks * kAttentionHeads);
            compute_barrier(command);
            push = {active_chunks, kAttentionHeads, 0, 0};
            kernels_.dispatch(command, kernels_.p().attention_reduce,
                              sets.attention_reduce, &push, kAttentionHeads);
#else
            push = {attention_index(layer), position, kAttentionHeads, 0};
            kernels_.dispatch(command, kernels_.p().attention, sets.attention,
                              &push, kAttentionHeads);
#endif
            compute_barrier(command);
            push = {kAttentionHeads * kHeadDim, 128,
                    kAttentionHeads * kHeadDim / 4,
                    kAttentionHeads * kHeadDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, sets.context_quant,
                              &push, kAttentionHeads * kHeadDim / 128);
            compute_barrier(command);
            push = {kDim, kAttentionHeads * kHeadDim,
                    kAttentionHeads * kHeadDim / 4, 0};
            kernels_.dispatch(command, sets.attention_out_native ? kernels_.p().nvfp4_residual : kernels_.p().q4_residual,
                              sets.attention_out, &push, kDim / 8);
        } else {
            push = {kMambaInput, kDim, kDim / 4, 0};
            kernels_.dispatch(command, sets.mamba_in_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.mamba_in, &push,
                              (kMambaInput + 7) / 8);
            compute_barrier(command);
            push = {kMambaConvChannels, kConvWidth, 0, 0};
            kernels_.dispatch(command, kernels_.p().conv, sets.conv, &push,
                              kMambaConvChannels / 64);
            compute_barrier(command);
            push = {kMambaHeads, kMambaHeadDim, kMambaGroups, kMambaState};
            kernels_.dispatch(command, kernels_.p().mamba_update,
                              sets.mamba_update, &push, kMambaHeads);
            compute_barrier(command);
            push = {kMambaGroups, kMambaIntermediate / kMambaGroups, 0, 0};
            kernels_.dispatch(command, kernels_.p().mamba_norm,
                              sets.mamba_norm, &push, kMambaGroups);
            compute_barrier(command);
            push = {kMambaIntermediate, 128, kMambaIntermediate / 4,
                    kMambaIntermediate / 4};
            kernels_.dispatch(command, kernels_.p().quant, sets.context_quant,
                              &push, kMambaIntermediate / 128);
            compute_barrier(command);
            push = {kDim, kMambaIntermediate, kMambaIntermediate / 4, 0};
            kernels_.dispatch(command, sets.mamba_out_native ? kernels_.p().nvfp4_residual : kernels_.p().q4_residual, sets.mamba_out,
                              &push, kDim / 8);
        }
        compute_barrier(command);
    }

    void record_moe_pre(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        Push push{1, kDim, float_bits(1e-5f), 0};
        kernels_.dispatch(command, kernels_.p().rms, sets.norm, &push, 1);
        compute_barrier(command);
        push = {kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.hidden_quant, &push,
                          (kDim + 127) / 128);
        compute_barrier(command);
        push = {kExperts, kDim, kDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().q8, sets.router_gemv, &push,
                          kExperts / 8);
        compute_barrier(command);
        push = {kExperts, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().router, sets.router, &push, 1);
        compute_barrier(command);
    }

    void record_shared(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        Push push{kSharedDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, sets.shared_up_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.shared_up, &push,
                          kSharedDim / 8);
        compute_barrier(command);
        push = {kSharedDim, 0, 0, 0};
        kernels_.dispatch(command, kernels_.p().relu2, sets.shared_relu2,
                          &push, (kSharedDim + 63) / 64);
        compute_barrier(command);
        push = {kSharedDim, 128, kSharedDim / 4, kSharedDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.shared_quant, &push,
                          (kSharedDim + 127) / 128);
        compute_barrier(command);
        push = {kDim, kSharedDim, kSharedDim / 4, 0};
        kernels_.dispatch(command, sets.shared_down_native ? kernels_.p().nvfp4 : kernels_.p().q4, sets.shared_down, &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_experts(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            Push push{rank, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate,
                              sets.expert_gate[selected_slots_[rank]], &push,
                              (kMoeDim + 7) / 8);
        }
        compute_barrier(command);
        const uint32_t packed = kMoeDim / 4u;
        const uint32_t scales = (kMoeDim + 127u) / 128u;
        const uint32_t all_packed = kTopK * packed;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint32_t relative = all_packed + rank * scales - rank * packed;
            Push push{kMoeDim, 128, packed, relative};
            kernels_.dispatch(command, kernels_.p().quant,
                              expert_rank_quant_sets_[rank], &push, scales);
        }
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            Push push{rank, all_packed, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_down,
                              sets.expert_down[selected_slots_[rank]], &push,
                              kDim / 8);
        }
        compute_barrier(command);
        Push push{kDim, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
    }

    void record_expert_rank(VkCommandBuffer command, uint32_t layer,
                            uint32_t rank) {
        LayerSets& sets = layers_[layer];
        Push push{rank, kDim / 4, 0, 0};
        kernels_.dispatch(command, kernels_.p().expert_gate,
                          sets.expert_gate[selected_slots_[rank]], &push,
                          (kMoeDim + 7) / 8);
        compute_barrier(command);
        // The descriptor begins at this rank's packed output.  Scale indices
        // are rebased so the resulting bytes are identical to the retained
        // all-rank quantizer layout.
        const uint32_t scales = (kMoeDim + 127u) / 128u;
        const uint32_t scale_relative =
            kTopK * kMoeDim / 4u + rank * scales - rank * (kMoeDim / 4u);
        push = {kMoeDim, 128u, kMoeDim / 4u, scale_relative};
        kernels_.dispatch(command, kernels_.p().quant,
                          expert_rank_quant_sets_[rank], &push,
                          scales);
        compute_barrier(command);
        push = {rank, kTopK * kMoeDim / 4u, 0, 0};
        kernels_.dispatch(command, kernels_.p().expert_down,
                          sets.expert_down[selected_slots_[rank]], &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_expert_finish(VkCommandBuffer command) {
        const Push push{kDim, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
    }

    void destroy_all() {
        for (Buffer* buffer : {&rope_, &kv_cache_,
#ifdef OVLLM_LONG_CONTEXT_FORK
                               &attention_partial_,
#endif
                               &recurrent_state_,
                               &conv_state_, &argmax_workspace_, &logits_,
                               &expert_outputs_, &expert_quant_,
                               &expert_intermediate_, &router_logits_,
                               &shared_expert_gate_, &shared_output_,
                               &shared_intermediate_, &shared_up_values_,
                               &shared_gate_values_, &quant_, &ab_, &z_,
                               &convolved_qkv_, &mixed_qkv_, &context_,
                               &value_, &key_, &qgate_, &normalized_, &hidden_,
                               &routing_, &token_})
            destroy_buffer(runtime_, *buffer);
    }

    const Runtime& runtime_;
    DeviceWeights weights_;
    ExpertFile expert_file_;
    HostExpertCache host_cache_;
    DeviceExpertCache device_cache_;
    Kernels kernels_;
    dsv4::FiniteQueue compute_, transfer_;
    Buffer token_{}, routing_{}, hidden_{}, normalized_{}, diagnostic_{};
    Buffer qgate_{}, key_{}, value_{}, context_{};
    Buffer mixed_qkv_{}, convolved_qkv_{}, z_{}, ab_{}, quant_{};
    Buffer shared_gate_values_{}, shared_up_values_{}, shared_intermediate_{};
    Buffer shared_output_{}, shared_expert_gate_{}, router_logits_{};
    Buffer expert_intermediate_{}, expert_quant_{}, expert_outputs_{};
    Buffer logits_{}, argmax_workspace_{}, conv_state_{}, recurrent_state_{};
    Buffer kv_cache_{}, rope_{};
#ifdef OVLLM_LONG_CONTEXT_FORK
    Buffer attention_partial_{};
    uint32_t max_context_ = kDefaultContext;
    uint32_t attention_chunks_ = 2;
#endif
    std::vector<Buffer> staging_;
    std::vector<LayerSets> layers_;
    std::array<uint32_t, kTopK> selected_slots_{};
    VkDescriptorSet embedding_set_{}, final_norm_set_{}, final_quant_set_{};
    VkDescriptorSet lm_head_set_{}, argmax_set_{}, expert_quant_set_{};
    std::array<VkDescriptorSet, kTopK> expert_rank_quant_sets_{};
    VkDescriptorSet reduce_set_{}, expert_gate_batch_set_{}, expert_down_batch_set_{};
    bool batch_experts_ = false;
    bool progressive_experts_ = false;
    std::unique_ptr<dsv4::experiment::FiniteQueueRing<12>> progressive_compute_;
    uint64_t activation_device_bytes_ = 0;
    uint64_t transfer_bytes_ = 0, host_copy_bytes_ = 0;
    uint64_t cold_stalled_layers_ = 0;
    uint64_t progressive_layers_ = 0;
    uint64_t progressive_resident_ranks_ = 0;
    uint64_t progressive_ram_ranks_ = 0;
    uint64_t progressive_disk_ranks_ = 0;
    double decode_seconds_ = 0, pre_seconds_ = 0;
    double acquisition_seconds_ = 0, expert_seconds_ = 0;
};

static uint64_t ram_budget() {
    const char* text = std::getenv("NEMOTRON3_RAM_GIB");
    const double gib = text ? std::stod(text) : 16.0;
    if (gib < 2.0 || gib > 56.0)
        throw std::runtime_error("NEMOTRON3_RAM_GIB must be 2..56");
    return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

static uint32_t device_slots() {
    const char* text = std::getenv("NEMOTRON3_DEVICE_SLOTS_PER_LAYER");
    const uint32_t slots = text ? static_cast<uint32_t>(std::stoul(text)) : 60;
    if (slots < kTopK || slots > 128)
        throw std::runtime_error("NEMOTRON3_DEVICE_SLOTS_PER_LAYER must be 6..128");
    return slots;
}

} // namespace nemotron3

int nemotron3_cli_main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: amd_nemotron3.exe <runtime-dir> "
                         "<prompt|--inspect|--tokenize> [new-tokens]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file((directory / "tokenizer.ovb").string());
        nemotron3::Tokenizer tokenizer(tokenizer_file);
        if (std::strcmp(argv[2], "--tokenize") == 0) {
            if (argc < 4) throw std::runtime_error("tokenize text required");
            const bool thinking = std::getenv("NEMOTRON3_NO_THINK") == nullptr;
            const std::vector<uint32_t> tokens =
                tokenizer.chat_prompt(argv[3], thinking);
            std::cout << "tokens:";
            for (uint32_t token : tokens) std::cout << ' ' << token;
            std::cout << '\n';
            return 0;
        }
        nemotron3::SharedIndex index(directory / "model-nvfp4.ovs");
        nemotron3::ExpertFile inspect_experts(directory / "experts-nvfp4.ovx");
        if (std::strcmp(argv[2], "--inspect") == 0) {
            std::cout << "Nemotron-3-Nano-30B-A3B runtime containers validated\n";
            return 0;
        }

        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << '\n';
        const uint32_t count = argc >= 4
            ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint64_t budget = nemotron3::ram_budget();
        const uint32_t slots = nemotron3::device_slots();
        const bool thinking = std::getenv("NEMOTRON3_NO_THINK") == nullptr;
        std::vector<uint32_t> result;
        double decode = 0, pre = 0, acquisition = 0, expert = 0;
        uint64_t device_hits = 0, device_misses = 0;
        uint64_t ram_hits = 0, ram_misses = 0, disk = 0, transfer = 0;
        uint64_t ram_bypasses = 0;
        uint64_t host_copy = 0, cold_layers = 0, ram = 0, vram = 0;
        uint64_t progressive_layers = 0, progressive_resident = 0;
        uint64_t progressive_ram = 0, progressive_disk = 0;
        uint32_t host_slots = 0;
        {
            nemotron3::QwenEngine engine(
                runtime, index, directory / "experts-nvfp4.ovx",
                std::filesystem::absolute(argv[0]).parent_path(), budget, slots);
            std::cout << "precision: preserved E2M1 NVFP4/BF16-K16 experts/dense, "
                         "Q8 embedding/head/router\n"
                      << "RAM budget: "
                      << double(budget) / double(1ull << 30) << " GiB\n"
                      << "expert slots device/RAM: " << slots
                      << " per layer / " << engine.host_slots() << " global\n";
            result = engine.generate(tokenizer,
                tokenizer.chat_prompt(argv[2], thinking), count);
            decode = engine.decode_seconds();
            pre = engine.pre_seconds();
            acquisition = engine.acquisition_seconds();
            expert = engine.expert_seconds();
            device_hits = engine.device_hits();
            device_misses = engine.device_misses();
            ram_hits = engine.ram_hits();
            ram_misses = engine.ram_misses();
            ram_bypasses = engine.ram_admission_bypasses();
            disk = engine.disk_bytes();
            transfer = engine.transfer_bytes();
            host_copy = engine.host_copy_bytes();
            cold_layers = engine.cold_stalled_layers();
            progressive_layers = engine.progressive_layers();
            progressive_resident = engine.progressive_resident_ranks();
            progressive_ram = engine.progressive_ram_ranks();
            progressive_disk = engine.progressive_disk_ranks();
            ram = engine.ram_bytes();
            vram = engine.vram_bytes();
            host_slots = engine.host_slots();
        }
        const uint64_t timed_tokens = result.size() > 1 ? result.size() - 1 : 0;
        const double divisor = timed_tokens ? double(timed_tokens) : 1.0;
        std::cout << "\ntoken ids:";
        for (uint32_t token : result) std::cout << ' ' << token;
        std::cout << "\ndecode throughput: "
                  << (decode > 0 ? timed_tokens / decode : 0.0) << " tok/s ("
                  << timed_tokens << " timed token transitions)\n"
                  << "device hits/misses: " << device_hits << '/'
                  << device_misses << "\nRAM hits/misses: " << ram_hits << '/'
                  << ram_misses << "\nRAM admission bypasses: "
                  << ram_bypasses << '\n'
                  << "expert SSD / host-copy / H2D bytes per output: "
                  << disk / divisor << " / " << host_copy / divisor << " / "
                  << transfer / divisor << '\n'
                  << "acquisition ms per output: "
                  << acquisition * 1000.0 / divisor << '\n'
                  << "cold-expert stalled layers per output: "
                  << cold_layers / divisor << '\n'
                  << "progressive layers / resident / RAM / disk ranks: "
                  << progressive_layers << " / " << progressive_resident
                  << " / " << progressive_ram << " / " << progressive_disk
                  << '\n'
                  << "attention+router / acquisition CPU-visible / "
                     "shared+expert wall s: "
                  << pre << " / " << acquisition << " / " << expert << '\n'
                  << "peak explicit host model RAM GiB: "
                  << double(ram) / double(1ull << 30) << " (capacity "
                  << host_slots << " records)\n"
                  << "peak device allocation estimate GiB: "
                  << double(vram) / double(1ull << 30) << '\n';
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Nemotron runtime error: " << error.what() << '\n';
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}

#ifndef OVLLM_NEMOTRON3_RUNTIME_ONLY
int main(int argc, char** argv) {
    return nemotron3_cli_main(argc, argv);
}
#endif
