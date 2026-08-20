#ifndef OVLLM_DSV4_RUNTIME_INCLUDED
#define OVLLM_DSV4_RUNTIME_INCLUDED 1
#define OVLLM_DSV4_RUNTIME_ONLY
#include "m13_deepseek_v4.cpp"
#endif

#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include "expert_acquisition_trace.hpp"

// Qwen3.5-122B-A10B text-only executor.  This is deliberately separate from
// the retained DeepSeek and Step-3.7 executors.  It reuses their finite Vulkan
// queues and Q4 expert-acquisition design, but all model math below follows the
// authoritative Qwen3.5 text architecture.
namespace qwen35 {

constexpr uint32_t kDim = 3072;
constexpr uint32_t kMoeDim = 1024;
constexpr uint32_t kLayers = 48;
constexpr uint32_t kExperts = 256;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kBaseVocabulary = 248044;
constexpr uint32_t kFullLayers = 12;
constexpr uint32_t kLinearLayers = 36;
constexpr uint32_t kAttentionHeads = 32;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kLinearKeyHeads = 16;
constexpr uint32_t kLinearValueHeads = 64;
constexpr uint32_t kLinearHeadDim = 128;
constexpr uint32_t kLinearQkv = 12288;
constexpr uint32_t kLinearValue = 8192;
constexpr uint32_t kConvWidth = 4;
constexpr uint32_t kMaximumContext = 2048;

constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kGateScale = 0;
constexpr uint64_t kGateWeight = 98304;
constexpr uint64_t kUpScale = 1671168;
constexpr uint64_t kUpWeight = 1769472;
constexpr uint64_t kDownScale = 3342336;
constexpr uint64_t kDownWeight = 3440640;
constexpr uint64_t kExpertRecordBytes = 5013504;
static_assert(kExpertRecordBytes % 4096 == 0);

constexpr uint32_t kEndOfText = 248044;
constexpr uint32_t kImStart = 248045;
constexpr uint32_t kImEnd = 248046;
constexpr uint32_t kThink = 248068;
constexpr uint32_t kEndThink = 248069;

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
    explicit SharedIndex(const std::filesystem::path& path,
                         bool mtp_container = false) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open Qwen shared container");
        read_at(input, 0, &header_, sizeof(header_), "header");
        const char* expected_magic = mtp_container ? "OQ35MTP\0" : "OQ35SHR\0";
        const uint32_t expected_layers = mtp_container ? 1u : kLayers;
        if (std::memcmp(header_.magic, expected_magic, 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.tensor_entry_bytes != sizeof(TensorEntry) ||
            header_.dimension != kDim || header_.moe_dimension != kMoeDim ||
            header_.layers != expected_layers || header_.heads != kAttentionHeads ||
            header_.kv_heads != kKvHeads || header_.head_dimension != kHeadDim ||
            header_.vocabulary != kVocabulary || header_.experts != kExperts ||
            header_.top_k != kTopK ||
            header_.expert_record_bytes != kExpertRecordBytes)
            throw std::runtime_error("Unsupported Qwen3.5 shared container");
        const uint64_t actual = std::filesystem::file_size(path);
        if (header_.file_bytes != actual ||
            header_.group_count != uint64_t(expected_layers) + 1u ||
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
            header_.merge_count != 247587 || header_.bos != UINT32_MAX ||
            header_.eos != kImEnd || header_.configured_pad != kEndOfText ||
            header_.pad_piece != kEndOfText || header_.user != kImStart ||
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
        if (!system_text.empty()) {
            result.push_back(kImStart);
            text("system\n" + system_text);
            result.push_back(kImEnd);
            text("\n");
        }
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
            text("\n\n");
            result.push_back(kEndThink);
            text("\n\n");
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
    struct SplitBatch {
        std::array<OVERLAPPED, 2 * kTopK> operations{};
        std::array<HANDLE, 2 * kTopK> events{};
        std::array<uint32_t, 2 * kTopK> ranks{};
        std::array<uint8_t, 2 * kTopK> parts{};
        std::array<uint32_t, 2 * kTopK> bytes{};
        std::array<bool, 2 * kTopK> pending{};
        uint32_t count = 0, remaining = 0;
        bool active = false;
    };

    explicit ExpertFile(const std::filesystem::path& path) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open Qwen expert container");
        read_at(input, 0, &header_, sizeof(header_), "expert header");
        if (std::memcmp(header_.magic, "OQ35EXP\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.dimension != kDim || header_.moe_dimension != kMoeDim ||
            header_.layers != kLayers || header_.experts != kExperts ||
            header_.record_bytes != kExpertRecordBytes ||
            header_.core_records != uint64_t(kLayers) * kExperts ||
            header_.file_bytes != std::filesystem::file_size(path) ||
            header_.w1_scale_offset != kGateScale ||
            header_.w1_weight_offset != kGateWeight ||
            header_.w3_scale_offset != kUpScale ||
            header_.w3_weight_offset != kUpWeight ||
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
        if (layer >= kLayers || expert >= kExperts)
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

    void begin_split(const std::vector<uint64_t>& offsets,
                     const std::vector<void*>& destinations,
                     const std::vector<uint32_t>& ranks,
                     SplitBatch& batch) {
        if (batch.active || offsets.size() != destinations.size() ||
            offsets.size() != ranks.size() || offsets.size() > kTopK)
            throw std::runtime_error("Invalid split Qwen expert batch");
        batch = {};
        batch.count = static_cast<uint32_t>(offsets.size()) * 2u;
        batch.remaining = batch.count;
        batch.active = true;
        try {
            for (uint32_t item = 0; item < offsets.size(); ++item) {
                for (uint32_t part = 0; part < 2; ++part) {
                    const uint32_t index = item * 2u + part;
                    const uint64_t relative = part ? kDownScale : 0;
                    const uint32_t bytes = static_cast<uint32_t>(
                        part ? kExpertRecordBytes - kDownScale : kDownScale);
                    batch.events[index] =
                        CreateEventW(nullptr, TRUE, FALSE, nullptr);
                    if (!batch.events[index])
                        throw std::runtime_error("Qwen split event creation failed");
                    const uint64_t offset = offsets[item] + relative;
                    batch.operations[index].Offset = DWORD(offset);
                    batch.operations[index].OffsetHigh = DWORD(offset >> 32);
                    batch.operations[index].hEvent = batch.events[index];
                    batch.ranks[index] = ranks[item];
                    batch.parts[index] = static_cast<uint8_t>(part);
                    batch.bytes[index] = bytes;
                    batch.pending[index] = true;
                    const BOOL started = ReadFile(file_,
                        static_cast<uint8_t*>(destinations[item]) + relative,
                        bytes, nullptr, &batch.operations[index]);
                    if (!started && GetLastError() != ERROR_IO_PENDING)
                        throw std::runtime_error("Qwen split read submission failed");
                }
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

    std::pair<uint32_t, uint32_t> wait_any_split(SplitBatch& batch) {
        if (!batch.active || !batch.remaining)
            throw std::runtime_error("No Qwen split read pending");
        std::array<HANDLE, 2 * kTopK> events{};
        std::array<uint32_t, 2 * kTopK> indices{};
        uint32_t count = 0;
        for (uint32_t i = 0; i < batch.count; ++i) if (batch.pending[i]) {
            events[count] = batch.events[i];
            indices[count++] = i;
        }
        const DWORD waited = WaitForMultipleObjects(count, events.data(), FALSE,
                                                     10000);
        if (waited < WAIT_OBJECT_0 || waited >= WAIT_OBJECT_0 + count)
            throw std::runtime_error("Qwen split expert read timed out");
        const uint32_t index = indices[waited - WAIT_OBJECT_0];
        DWORD transferred = 0;
        if (!GetOverlappedResult(file_, &batch.operations[index], &transferred,
                                 FALSE) || transferred != batch.bytes[index])
            throw std::runtime_error("Qwen split expert read failed");
        batch.pending[index] = false;
        --batch.remaining;
        return {batch.ranks[index], batch.parts[index]};
    }

    void finish_split(SplitBatch& batch) {
        if (!batch.active || batch.remaining)
            throw std::runtime_error("Qwen split batch incomplete");
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
    struct SplitProgressiveBatch {
        Batch sources{};
        ExpertFile::SplitBatch reads{};
        bool active = false;
    };
    struct Many {
        std::array<const uint8_t*, 2 * kTopK> pointers{};
        uint32_t count = 0;
        uint32_t disk_reads = 0;
    };

    HostExpertCache(ExpertFile& file, uint64_t budget_bytes) : file_(file) {
        tiny_lfu_ = std::getenv("QWEN_HOST_TINYLFU") != nullptr;
        lru_ = std::getenv("QWEN_HOST_LRU") != nullptr;
        const uint64_t reserve = budget_bytes > 512ull * 1024 * 1024
            ? budget_bytes - 512ull * 1024 * 1024 : 0;
        slots_ = static_cast<uint32_t>(std::min<uint64_t>(
            reserve / kExpertRecordBytes, uint64_t(kLayers) * kExperts));
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen RAM budget is too small");
        bytes_ = uint64_t(slots_) * kExpertRecordBytes;
        base_ = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, bytes_, MEM_RESERVE, PAGE_READWRITE));
        if (!base_) throw std::runtime_error("Could not reserve Qwen RAM cache");
        entries_.resize(slots_);
        locations_.assign(uint64_t(kLayers) * kExperts, -1);
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

    void begin_progressive_split(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        const std::array<bool, kTopK>& needed,
        const std::array<void*, kTopK>& direct_destinations,
        SplitProgressiveBatch& batch) {
        if (batch.active)
            throw std::runtime_error("Nested Qwen split acquisition");
        batch = {};
        batch.sources = resolve_batch_impl(layer, experts, needed,
            direct_destinations, nullptr, nullptr, &batch.reads);
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

    std::pair<uint32_t, uint32_t> wait_next_split(
        SplitProgressiveBatch& batch) {
        if (!batch.active) throw std::runtime_error("No Qwen split batch");
        return file_.wait_any_split(batch.reads);
    }

    void finish_progressive_split(SplitProgressiveBatch& batch) {
        if (!batch.active) return;
        file_.finish_split(batch.reads);
        batch.active = false;
    }

    uint32_t fill_remaining_uniform(double& seconds) {
        const auto started = std::chrono::steady_clock::now();
        constexpr uint32_t kPermutationMultiplier = 73;
        constexpr uint32_t kLayerOffset = 29;
        const uint32_t total_keys = kLayers * kExperts;
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
                const uint32_t layer = candidate % kLayers;
                const uint32_t round = candidate / kLayers;
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
                throw std::runtime_error("Qwen RAM-cache top-off exhausted keys");
            uint8_t* pointer = base_ + uint64_t(slot) * kExpertRecordBytes;
            if (!entries_[slot].committed) {
                if (!VirtualAlloc(pointer, kExpertRecordBytes, MEM_COMMIT,
                                  PAGE_READWRITE))
                    throw std::runtime_error(
                        "Qwen RAM-cache top-off commit failed");
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

    Many resolve_many(uint32_t layer,
                      const std::array<uint32_t, 2 * kTopK>& experts,
                      uint32_t count) {
        if (count > 2 * kTopK) throw std::runtime_error("Too many Qwen experts");
        Many result{};
        result.count = count;
        std::vector<bool> reserved(slots_);
        std::vector<uint64_t> offsets;
        std::vector<void*> destinations;
        const auto flush_reads = [&]() {
            if (offsets.empty()) return;
            file_.read_batch(offsets, destinations);
            result.disk_reads += static_cast<uint32_t>(offsets.size());
            disk_bytes_ += uint64_t(offsets.size()) * kExpertRecordBytes;
            offsets.clear();
            destinations.clear();
        };
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t expert = experts[index];
            const uint32_t key = layer * kExperts + expert;
            ++frequency_[key];
            const int32_t location = locations_[key];
            if (location >= 0) {
                ++hits_;
                entries_[location].age = ++clock_;
                reserved[location] = true;
                result.pointers[index] =
                    base_ + uint64_t(location) * kExpertRecordBytes;
                continue;
            }
            ++misses_;
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (reserved[slot]) continue;
                bool replace = victim == UINT32_MAX || entries_[slot].key < 0;
                if (!replace && entries_[victim].key >= 0) {
                    replace = lru_ ? entries_[slot].age < entries_[victim].age :
                        (frequency_[entries_[slot].key] <
                             frequency_[entries_[victim].key] ||
                         (frequency_[entries_[slot].key] ==
                              frequency_[entries_[victim].key] &&
                          entries_[slot].age < entries_[victim].age));
                }
                if (replace) { victim = slot; if (entries_[slot].key < 0) break; }
            }
            if (victim == UINT32_MAX)
                throw std::runtime_error("No Qwen many-cache victim");
            Entry& entry = entries_[victim];
            if (entry.key >= 0) locations_[entry.key] = -1;
            uint8_t* pointer = base_ + uint64_t(victim) * kExpertRecordBytes;
            if (!entry.committed) {
                if (!VirtualAlloc(pointer, kExpertRecordBytes, MEM_COMMIT,
                                  PAGE_READWRITE))
                    throw std::runtime_error("Qwen many-cache commit failed");
                entry.committed = true;
                ++committed_;
            }
            entry.key = static_cast<int32_t>(key);
            entry.age = ++clock_;
            locations_[key] = static_cast<int32_t>(victim);
            reserved[victim] = true;
            result.pointers[index] = pointer;
            offsets.push_back(file_.offset(layer, expert));
            destinations.push_back(pointer);
            if (offsets.size() == kTopK) flush_reads();
        }
        flush_reads();
        return result;
    }

    Batch resolve_batch_impl(
        uint32_t layer, const std::array<uint32_t, kTopK>& experts,
        const std::array<bool, kTopK>& needed,
        const std::array<void*, kTopK>& direct_destinations,
        ExpertFile::AsyncBatch* async,
        std::array<bool, kTopK>* disk_pending,
        ExpertFile::SplitBatch* split = nullptr) {
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
                bool replace = victim == UINT32_MAX || entries_[slot].key < 0;
                if (!replace && entries_[victim].key >= 0) {
                    replace = lru_ ?
                        entries_[slot].age < entries_[victim].age :
                        (frequency_[entries_[slot].key] <
                             frequency_[entries_[victim].key] ||
                         (frequency_[entries_[slot].key] ==
                              frequency_[entries_[victim].key] &&
                          entries_[slot].age < entries_[victim].age));
                }
                if (replace) {
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
        if (split) {
            file_.begin_split(offsets, destinations, read_ranks, *split);
        } else if (async) {
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
    bool lru_ = false;
};

class DeviceExpertCache {
public:
    DeviceExpertCache(const Runtime& runtime, uint32_t slots)
        : runtime_(runtime), slots_(slots) {
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen device expert cache is too small");
        capacities_.assign(kLayers, slots_);
        if (const char* profile = std::getenv("QWEN_DEVICE_SLOT_PROFILE")) {
            profiled_ = true;
            std::string text(profile);
            size_t begin = 0;
            uint64_t sum = 0;
            for (uint32_t layer = 0; layer < kLayers; ++layer) {
                const size_t end = text.find(',', begin);
                const std::string item = text.substr(begin, end - begin);
                if (item.empty())
                    throw std::runtime_error("Empty Qwen device profile entry");
                const uint32_t value = static_cast<uint32_t>(std::stoul(item));
                if (value < kTopK || value > 32u)
                    throw std::runtime_error("Qwen device profile entry is outside 8..32");
                capacities_[layer] = value;
                sum += value;
                if (layer + 1u < kLayers) {
                    if (end == std::string::npos)
                        throw std::runtime_error("Short Qwen device profile");
                    begin = end + 1u;
                } else if (end != std::string::npos) {
                    throw std::runtime_error("Long Qwen device profile");
                }
            }
            if (sum != uint64_t(slots_) * kLayers)
                throw std::runtime_error("Qwen device profile changes the VRAM slot budget");
        }
        layers_.resize(kLayers);
        for (uint32_t index = 0; index < kLayers; ++index) {
            Layer& layer = layers_[index];
            layer.arena = create_device_buffer(
                runtime, uint64_t(capacities_[index]) * kExpertRecordBytes);
            layer.entries.resize(capacities_[index]);
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
    struct Batch2 {
        std::array<std::array<uint32_t, kTopK>, 2> slots{};
        std::array<uint32_t, 2 * kTopK> unique_experts{};
        std::array<uint32_t, 2 * kTopK> unique_slots{};
        std::array<bool, 2 * kTopK> unique_misses{};
        uint32_t unique_count = 0;
        uint32_t reused_occurrences = 0;
    };

    Selection resolve(uint32_t layer_index,
                      const std::array<uint32_t, kTopK>& experts) {
        Layer& layer = layers_.at(layer_index);
        const uint32_t capacity = capacities_.at(layer_index);
        Selection result{};
        std::vector<bool> reserved(capacity);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            ++layer.frequency[experts[rank]];
            result.slots[rank] = UINT32_MAX;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
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
            for (uint32_t slot = 0; slot < capacity; ++slot) {
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

    Batch2 resolve_two(
        uint32_t layer_index,
        const std::array<std::array<uint32_t, kTopK>, 2>& experts) {
        Layer& layer = layers_.at(layer_index);
        const uint32_t capacity = capacities_.at(layer_index);
        Batch2 result{};
        if (capacity < 2 * kTopK)
            throw std::runtime_error("Qwen verify2 requires at least 16 slots/layer");
        std::array<uint32_t, 2 * kTopK> occurrence_unique{};
        for (uint32_t row = 0; row < 2; ++row) {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const uint32_t expert = experts[row][rank];
                ++layer.frequency[expert];
                uint32_t unique = 0;
                for (; unique < result.unique_count; ++unique)
                    if (result.unique_experts[unique] == expert) break;
                if (unique == result.unique_count) {
                    result.unique_experts[result.unique_count++] = expert;
                } else {
                    ++result.reused_occurrences;
                }
                occurrence_unique[row * kTopK + rank] = unique;
            }
        }
        std::vector<bool> reserved(capacity);
        for (uint32_t unique = 0; unique < result.unique_count; ++unique) {
            const uint32_t expert = result.unique_experts[unique];
            result.unique_slots[unique] = UINT32_MAX;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
                if (layer.entries[slot].expert != int32_t(expert)) continue;
                result.unique_slots[unique] = slot;
                reserved[slot] = true;
                layer.entries[slot].age = ++clock_;
                ++hits_;
                break;
            }
        }
        for (uint32_t unique = 0; unique < result.unique_count; ++unique) {
            if (result.unique_slots[unique] != UINT32_MAX) continue;
            ++misses_;
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < capacity; ++slot) {
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
                throw std::runtime_error("No Qwen verify2 device victim");
            layer.entries[victim].expert =
                int32_t(result.unique_experts[unique]);
            layer.entries[victim].age = ++clock_;
            reserved[victim] = true;
            result.unique_slots[unique] = victim;
            result.unique_misses[unique] = true;
        }
        for (uint32_t row = 0; row < 2; ++row)
            for (uint32_t rank = 0; rank < kTopK; ++rank)
                result.slots[row][rank] =
                    result.unique_slots[occurrence_unique[row * kTopK + rank]];
        return result;
    }

    DescriptorRange record(uint32_t layer, uint32_t slot) const {
        return arena_range(layers_.at(layer).arena,
                           uint64_t(slot) * kExpertRecordBytes,
                           kExpertRecordBytes);
    }
    Buffer& arena(uint32_t layer) { return layers_.at(layer).arena; }
    uint32_t slots() const { return slots_; }
    uint32_t slots(uint32_t layer) const { return capacities_.at(layer); }
    uint32_t total_slots() const {
        return static_cast<uint32_t>(std::accumulate(
            capacities_.begin(), capacities_.end(), uint64_t(0)));
    }
    bool profiled() const { return profiled_; }
    uint32_t minimum_slots() const {
        return *std::min_element(capacities_.begin(), capacities_.end());
    }
    uint32_t maximum_slots() const {
        return *std::max_element(capacities_.begin(), capacities_.end());
    }
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
    std::vector<uint32_t> capacities_;
    bool profiled_ = false;
    std::vector<Layer> layers_;
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, device_bytes_ = 0;
};

struct Pipelines {
    VkPipeline embedding{}, rms{}, quant{}, q4{}, q4_residual{}, q8{};
    VkPipeline swiglu{}, router{}, expert_gate{}, expert_down{}, reduce{};
    VkPipeline expert_gate_batch{}, expert_down_batch{};
    VkPipeline qk{}, store_value{}, attention{}, head_gate{};
    VkPipeline conv{}, delta{}, argmax{}, mtp_fuse{};
};

class Kernels {
public:
    Kernels(const Runtime& runtime, const std::filesystem::path& directory)
        : runtime_(runtime), resources_(create_compute_resources(runtime, 8192)),
          dummy_(create_device_buffer(runtime, 4096)) {
        const auto load = [&](const char* name) {
            return dsv4::create_dsv4_pipeline(
                runtime_, resources_, directory / (std::string(name) + ".comp.spv"), 64);
        };
        pipelines_.embedding = load("dsv4_embedding");
        pipelines_.rms = load("step37_rmsnorm");
        pipelines_.quant = load("dsv4_quantize_q8");
        pipelines_.q4 = load("dsv4_q4g64t_gemv");
        pipelines_.q4_residual = load("dsv4_q4g64t_gemv_residual");
        pipelines_.q8 = load("dsv4_q8_gemv");
        pipelines_.swiglu = load("step37_swiglu");
        pipelines_.router = load("qwen35_router_top8");
        pipelines_.expert_gate = load("qwen35_expert_gate_up_q4");
        pipelines_.expert_down = load("qwen35_expert_down_q4");
        pipelines_.expert_gate_batch =
            load("qwen35_expert_gate_up_q4_bda_batch");
        pipelines_.expert_down_batch =
            load("qwen35_expert_down_q4_bda_batch");
        pipelines_.reduce = load("qwen35_reduce_shared_gate");
        pipelines_.qk = load("qwen35_qk_rope_cache");
        pipelines_.store_value = load("qwen35_store_value");
        pipelines_.attention = load("qwen35_attention");
        pipelines_.head_gate = load("qwen35_head_gate");
        pipelines_.conv = load("qwen35_conv_update");
        pipelines_.delta = load("qwen35_delta_recurrent_norm");
        pipelines_.argmax = load("qwen35_greedy_argmax");
        pipelines_.mtp_fuse = load("step37_mtp_fuse");
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
        batch_experts_ = std::getenv("QWEN_EXPERT_BATCH_BDA") != nullptr;
        progressive_experts_ =
            std::getenv("QWEN_PROGRESSIVE_EXPERTS") != nullptr;
        verify2_enabled_ = std::getenv("QWEN_VERIFY2_EXPERIMENT") != nullptr;
        tensor_split_ = std::getenv("QWEN_TENSOR_SPLIT_EXPERIMENT") != nullptr;
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
        if (verify2_enabled_) {
            verify_staging_.resize(2 * kTopK);
            for (Buffer& buffer : verify_staging_)
                buffer = dsv4::create_host_buffer_uninitialized(
                    runtime_, kExpertRecordBytes);
        }
        expert_trace_.open(1u, kLayers, kExperts, kTopK,
                           static_cast<uint32_t>(kExpertRecordBytes),
                           device_cache_.total_slots(),
                           host_cache_.capacity());
    }

    ~QwenEngine() {
        for (Buffer& buffer : verify_staging_) destroy_buffer(runtime_, buffer);
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        destroy_all();
    }

    std::vector<uint32_t> generate(const Tokenizer& tokenizer,
                                   const std::vector<uint32_t>& prompt,
                                   uint32_t count) {
        if (prompt.empty()) throw std::runtime_error("Qwen prompt is empty");
        uint32_t position = 0;
        uint32_t next = 0;
        for (; position < prompt.size(); ++position)
            next = run_token(prompt[position], position);
        if (std::getenv("QWEN_FILL_RAM_CACHE")) {
            double fill_seconds = 0.0;
            const uint32_t filled =
                host_cache_.fill_remaining_uniform(fill_seconds);
            std::cout << "RAM cache top-off: " << filled << " records, "
                      << double(host_cache_.committed_bytes()) /
                             double(1ull << 30)
                      << " GiB cache, " << fill_seconds << " s\n";
        }
        reset_decode_metrics();
        expert_trace_.set_decode(true);
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
               uint64_t(staging_.size()) * kExpertRecordBytes;
    }
    uint64_t vram_bytes() const {
        return weights_.device_bytes() + device_cache_.device_bytes() +
               activation_device_bytes_;
    }
    uint32_t host_slots() const { return host_cache_.capacity(); }
    uint32_t device_slots() const { return device_cache_.slots(); }
    uint32_t device_total_slots() const { return device_cache_.total_slots(); }
    uint32_t device_minimum_slots() const { return device_cache_.minimum_slots(); }
    uint32_t device_maximum_slots() const { return device_cache_.maximum_slots(); }
    bool device_profiled() const { return device_cache_.profiled(); }
    double decode_seconds() const { return decode_seconds_; }
    double pre_seconds() const { return pre_seconds_; }
    double acquisition_seconds() const { return acquisition_seconds_; }
    double expert_seconds() const { return expert_seconds_; }
    // Experimental MTP accessors.  They expose the already-retained main-model
    // state without changing the ordinary generation path.
    uint32_t process_experiment_token(uint32_t token, uint32_t position) {
        return run_token(token, position);
    }
    DescriptorRange hidden_experiment_range() const { return whole(hidden_); }
    TensorDevice embedding_experiment_tensor() const {
        return weights_.tensor("embed");
    }
    TensorDevice lm_head_experiment_tensor() const {
        return weights_.tensor("lm_head");
    }
    void reset_experiment_metrics() { reset_decode_metrics(); }
    uint32_t project_experiment_hidden(DescriptorRange source) {
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{source.offset, 0, uint64_t(kDim) * sizeof(float)};
            vkfn::CmdCopyBuffer(command, source.buffer, hidden_.handle, 1, &copy);
            dsv4::transfer_barrier(command, hidden_);
            Push push{1, kDim, float_bits(1e-6f), 0};
            kernels_.dispatch(command, kernels_.p().rms, final_norm_set_, &push, 1);
            compute_barrier(command);
            push = {kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, final_quant_set_, &push,
                              kDim / 128);
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
    uint32_t project_experiment_normalized(DescriptorRange source) {
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{source.offset, 0, uint64_t(kDim) * sizeof(float)};
            vkfn::CmdCopyBuffer(command, source.buffer, normalized_.handle, 1,
                                &copy);
            dsv4::transfer_barrier(command, normalized_);
            Push push{kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, final_quant_set_, &push,
                              kDim / 128);
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
    std::array<uint32_t, 2> verify2_experiment(
        const std::array<uint32_t, 2>& tokens, uint32_t position) {
        if (!verify2_enabled_)
            throw std::runtime_error("Qwen verify2 experiment is disabled");
        if (position + 1 >= kMaximumContext)
            throw std::runtime_error("Qwen verify2 context cap reached");
        auto* token_words = static_cast<uint32_t*>(verify_tokens_.mapped);
        token_words[0] = tokens[0];
        token_words[1] = tokens[1];
        flush_buffer(runtime_, verify_tokens_);
        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            for (uint32_t row = 0; row < 2; ++row) {
                Push push{kVocabulary, kDim, kDim / 4, 0};
                kernels_.dispatch(command, kernels_.p().embedding,
                                  verify_embedding_sets_[row], &push, kDim / 64);
            }
            compute_barrier(command);
        });
        compute_.wait(signal);

        const auto to_transfer = [](VkCommandBuffer command) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr,
                0, nullptr);
        };
        const auto to_compute = [](VkCommandBuffer command) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                nullptr, 0, nullptr);
        };

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            const auto pre_started = std::chrono::steady_clock::now();
            signal = compute_.submit([&](VkCommandBuffer command) {
                for (uint32_t row = 0; row < 2; ++row) {
                    to_transfer(command);
                    VkBufferCopy hidden_copy{0, 0, uint64_t(kDim) * 4};
                    vkfn::CmdCopyBuffer(command, verify_hidden_[row].handle,
                                        hidden_.handle, 1, &hidden_copy);
                    to_compute(command);
                    record_attention(command, layer, position + row);
                    record_router(command, layer);
                    to_transfer(command);
                    vkfn::CmdCopyBuffer(command, hidden_.handle,
                                        verify_hidden_[row].handle, 1,
                                        &hidden_copy);
                    const VkBufferCopy quant_copy{0, 0, quant_.size};
                    vkfn::CmdCopyBuffer(command, quant_.handle,
                                        verify_quant_[row].handle, 1,
                                        &quant_copy);
                    const VkBufferCopy route_copy{0,
                        uint64_t(row) * 16u * sizeof(uint32_t),
                        16u * sizeof(uint32_t)};
                    vkfn::CmdCopyBuffer(command, routing_.handle,
                                        verify_routing_.handle, 1, &route_copy);
                    if (row == 0 && !full_attention(layer)) {
                        const uint64_t linear = linear_index(layer);
                        const VkBufferCopy conv_copy{
                            linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                            linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                            uint64_t(kLinearQkv) * kConvWidth * 4};
                        vkfn::CmdCopyBuffer(command, conv_state_.handle,
                            verify_conv_snapshot_.handle, 1, &conv_copy);
                        const VkBufferCopy recurrent_copy{
                            linear * uint64_t(kLinearValueHeads) *
                                kLinearHeadDim * kLinearHeadDim * 4,
                            linear * uint64_t(kLinearValueHeads) *
                                kLinearHeadDim * kLinearHeadDim * 4,
                            uint64_t(kLinearValueHeads) * kLinearHeadDim *
                                kLinearHeadDim * 4};
                        vkfn::CmdCopyBuffer(command, recurrent_state_.handle,
                            verify_recurrent_snapshot_.handle, 1,
                            &recurrent_copy);
                    }
                    to_compute(command);
                }
            });
            compute_.wait(signal);
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pre_started).count();
            invalidate_buffer(runtime_, verify_routing_);
            std::array<std::array<uint32_t, kTopK>, 2> experts{};
            const uint32_t* routes =
                static_cast<const uint32_t*>(verify_routing_.mapped);
            for (uint32_t row = 0; row < 2; ++row)
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    experts[row][rank] = routes[row * 16u + rank];
                    if (experts[row][rank] >= kExperts)
                        throw std::runtime_error(
                            "Qwen verify2 router returned invalid expert");
                }

            const auto acquire_started = std::chrono::steady_clock::now();
            const DeviceExpertCache::Batch2 selection =
                device_cache_.resolve_two(layer, experts);
            verify_unique_experts_ += selection.unique_count;
            verify_occurrences_ += 2 * kTopK;
            verify_reused_occurrences_ += selection.reused_occurrences;
            std::array<uint32_t, 2 * kTopK> missing_experts{};
            std::array<uint32_t, 2 * kTopK> missing_unique{};
            uint32_t missing_count = 0;
            for (uint32_t unique = 0; unique < selection.unique_count; ++unique) {
                if (!selection.unique_misses[unique]) continue;
                missing_experts[missing_count] = selection.unique_experts[unique];
                missing_unique[missing_count++] = unique;
            }
            const HostExpertCache::Many sources = host_cache_.resolve_many(
                layer, missing_experts, missing_count);
            if (sources.disk_reads) ++cold_stalled_layers_;
            for (uint32_t index = 0; index < missing_count; ++index) {
                std::memcpy(verify_staging_[index].mapped,
                            sources.pointers[index], kExpertRecordBytes);
                host_copy_bytes_ += kExpertRecordBytes;
                dsv4::flush_buffer_range(runtime_, verify_staging_[index], 0,
                                         kExpertRecordBytes);
            }
            uint64_t ready = 0;
            if (missing_count) {
                ready = transfer_.submit([&](VkCommandBuffer command) {
                    for (uint32_t index = 0; index < missing_count; ++index) {
                        const uint32_t unique = missing_unique[index];
                        const VkBufferCopy copy{0,
                            uint64_t(selection.unique_slots[unique]) *
                                kExpertRecordBytes,
                            kExpertRecordBytes};
                        vkfn::CmdCopyBuffer(command,
                            verify_staging_[index].handle,
                            device_cache_.arena(layer).handle, 1, &copy);
                    }
                    dsv4::transfer_barrier(command, device_cache_.arena(layer));
                });
                transfer_bytes_ += uint64_t(missing_count) * kExpertRecordBytes;
            }
            acquisition_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - acquire_started).count();

            const auto expert_started = std::chrono::steady_clock::now();
            signal = compute_.submit([&](VkCommandBuffer command) {
                for (uint32_t row = 0; row < 2; ++row) {
                    to_transfer(command);
                    const VkBufferCopy hidden_copy{0, 0, uint64_t(kDim) * 4};
                    vkfn::CmdCopyBuffer(command, verify_hidden_[row].handle,
                                        hidden_.handle, 1, &hidden_copy);
                    const VkBufferCopy quant_copy{0, 0, quant_.size};
                    vkfn::CmdCopyBuffer(command, verify_quant_[row].handle,
                                        quant_.handle, 1, &quant_copy);
                    const VkBufferCopy route_copy{
                        uint64_t(row) * 16u * sizeof(uint32_t), 0,
                        16u * sizeof(uint32_t)};
                    vkfn::CmdCopyBuffer(command, verify_routing_.handle,
                                        routing_.handle, 1, &route_copy);
                    to_compute(command);
                    selected_slots_ = selection.slots[row];
                    record_shared(command, layer);
                    record_experts(command, layer);
                    to_transfer(command);
                    vkfn::CmdCopyBuffer(command, hidden_.handle,
                                        verify_hidden_[row].handle, 1,
                                        &hidden_copy);
                    to_compute(command);
                }
            }, ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            compute_.wait(signal);
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
        }
        std::array<uint32_t, 2> result{};
        result[0] = project_experiment_hidden(whole(verify_hidden_[0]));
        result[1] = project_experiment_hidden(whole(verify_hidden_[1]));
        return result;
    }

    void accept_verify2_experiment(bool second_row) {
        if (!verify2_enabled_)
            throw std::runtime_error("Qwen verify2 experiment is disabled");
        if (second_row) return;
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy hidden_copy{0, 0, uint64_t(kDim) * 4};
            vkfn::CmdCopyBuffer(command, verify_hidden_[0].handle,
                                hidden_.handle, 1, &hidden_copy);
            const VkBufferCopy conv_copy{0, 0, conv_state_.size};
            vkfn::CmdCopyBuffer(command, verify_conv_snapshot_.handle,
                                conv_state_.handle, 1, &conv_copy);
            const VkBufferCopy recurrent_copy{0, 0, recurrent_state_.size};
            vkfn::CmdCopyBuffer(command, verify_recurrent_snapshot_.handle,
                                recurrent_state_.handle, 1, &recurrent_copy);
            dsv4::transfer_barrier(command, hidden_);
            dsv4::transfer_barrier(command, conv_state_);
            dsv4::transfer_barrier(command, recurrent_state_);
        });
        compute_.wait(signal);
    }
    uint64_t verify_unique_experts() const { return verify_unique_experts_; }
    uint64_t verify_occurrences() const { return verify_occurrences_; }
    uint64_t verify_reused_occurrences() const {
        return verify_reused_occurrences_;
    }
private:
    static bool full_attention(uint32_t layer) { return layer % 4u == 3u; }
    static uint32_t full_index(uint32_t layer) { return layer / 4u; }
    static uint32_t linear_index(uint32_t layer) { return layer - layer / 4u; }

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
        qgate_ = device(16384ull * 4);
        key_ = device(512ull * 4);
        value_ = device(512ull * 4);
        context_ = device(kLinearValue * 4ull);
        mixed_qkv_ = device(kLinearQkv * 4ull);
        convolved_qkv_ = device(kLinearQkv * 4ull);
        z_ = device(kLinearValue * 4ull);
        ab_ = device(128ull * 4);
        quant_ = device(4224ull * 4);
        shared_gate_values_ = device(kMoeDim * 4ull);
        shared_up_values_ = device(kMoeDim * 4ull);
        shared_intermediate_ = device(kMoeDim * 4ull);
        shared_output_ = device(kDim * 4ull);
        shared_expert_gate_ = device(4);
        router_logits_ = device(kExperts * 4ull);
        expert_intermediate_ = device(uint64_t(kTopK) * kMoeDim * 4);
        expert_quant_ = device(2112ull * 4);
        expert_outputs_ = device(uint64_t(kTopK) * kDim * 4);
        logits_ = device(uint64_t(kVocabulary) * 4);
        argmax_workspace_ = device(512ull * 4);
        conv_state_ = device(uint64_t(kLinearLayers) * kLinearQkv *
                             kConvWidth * 4);
        recurrent_state_ = device(uint64_t(kLinearLayers) *
                                  kLinearValueHeads * kLinearHeadDim *
                                  kLinearHeadDim * 4);
        kv_cache_ = device(uint64_t(kFullLayers) * 2 * kMaximumContext *
                           kKvHeads * kHeadDim * 4);
        rope_ = device(uint64_t(kMaximumContext) * kRopeDim * 4);
        if (verify2_enabled_) {
            verify_tokens_ = create_buffer(runtime_, 2u * sizeof(uint32_t));
            verify_routing_ = create_buffer(runtime_,
                                             2u * 16u * sizeof(uint32_t));
            for (Buffer& buffer : verify_hidden_)
                buffer = device(uint64_t(kDim) * sizeof(float));
            for (Buffer& buffer : verify_quant_)
                buffer = device(4224ull * sizeof(uint32_t));
            verify_conv_snapshot_ = device(conv_state_.size);
            verify_recurrent_snapshot_ = device(recurrent_state_.size);
        }
    }

    void initialize_persistent_buffers() {
        constexpr uint64_t chunk_bytes = 64ull * 1024 * 1024;
        Buffer zeros = dsv4::create_host_buffer_uninitialized(runtime_, chunk_bytes);
        std::memset(zeros.mapped, 0, static_cast<size_t>(chunk_bytes));
        dsv4::flush_buffer_range(runtime_, zeros, 0, chunk_bytes);
        for (Buffer* buffer : {&conv_state_, &recurrent_state_, &kv_cache_}) {
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

    void make_rope() {
        std::vector<float> table(uint64_t(kMaximumContext) * kRopeDim);
        for (uint32_t position = 0; position < kMaximumContext; ++position) {
            for (uint32_t pair = 0; pair < kRopeDim / 2; ++pair) {
                const double inverse = std::pow(10000000.0,
                                                -2.0 * double(pair) /
                                                    double(kRopeDim));
                const double angle = double(position) * inverse;
                table[uint64_t(position) * kRopeDim + pair] =
                    static_cast<float>(std::cos(angle));
                table[uint64_t(position) * kRopeDim + kRopeDim / 2 + pair] =
                    static_cast<float>(std::sin(angle));
            }
        }
        Buffer staging = dsv4::create_host_buffer_uninitialized(
            runtime_, table.size() * sizeof(float));
        std::memcpy(staging.mapped, table.data(), table.size() * sizeof(float));
        dsv4::flush_buffer_range(runtime_, staging, 0,
                                 table.size() * sizeof(float));
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{0, 0, table.size() * sizeof(float)};
            vkfn::CmdCopyBuffer(command, staging.handle, rope_.handle, 1, &copy);
            dsv4::transfer_barrier(command, rope_);
        });
        compute_.wait(signal);
        destroy_buffer(runtime_, staging);
    }

    struct LayerSets {
        VkDescriptorSet input_norm{}, hidden_quant{};
        VkDescriptorSet qgate{}, key{}, value{}, qk{}, store_value{};
        VkDescriptorSet attention{}, head_gate{}, context_quant{}, attention_out{};
        VkDescriptorSet gdn_qkv{}, gdn_z{}, ab{}, conv{}, delta{}, gdn_out{};
        VkDescriptorSet post_norm{}, router_gemv{}, router{};
        VkDescriptorSet shared_gate{}, shared_up{}, shared_swiglu{};
        VkDescriptorSet shared_quant{}, shared_down{}, shared_expert_gate{};
        std::vector<VkDescriptorSet> expert_gate, expert_down;
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
        if (verify2_enabled_) {
            for (uint32_t row = 0; row < 2; ++row) {
                verify_embedding_sets_[row] = kernels_.set({
                    embedding.data, embedding.auxiliary,
                    arena_range(verify_tokens_, uint64_t(row) * sizeof(uint32_t),
                                sizeof(uint32_t)),
                    whole(verify_hidden_[row])});
            }
        }
        final_norm_set_ = kernels_.set(
            {whole(hidden_), final_norm.data, whole(normalized_)});
        final_quant_set_ = kernels_.set({whole(normalized_), whole(quant_)});
        lm_head_set_ = kernels_.set(
            {whole(quant_), lm_head.data, lm_head.auxiliary, whole(logits_)});
        argmax_set_ = kernels_.set(
            {whole(logits_), whole(token_), whole(argmax_workspace_)});
        expert_quant_set_ = kernels_.set(
            {whole(expert_intermediate_), whole(expert_quant_)});
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint64_t input_offset =
                uint64_t(rank) * kMoeDim * sizeof(float);
            const uint64_t output_offset =
                uint64_t(rank) * (kMoeDim / 4u) * sizeof(uint32_t);
            expert_rank_quant_sets_[rank] = kernels_.set({
                arena_range(expert_intermediate_, input_offset,
                            uint64_t(kMoeDim) * sizeof(float)),
                arena_range(expert_quant_, output_offset,
                            expert_quant_.size - output_offset)});
        }
        reduce_set_ = kernels_.set({whole(expert_outputs_), whole(shared_output_),
                                    whole(shared_expert_gate_), whole(hidden_)});
        const DescriptorRange expert_addresses = arena_range(
            routing_, 16u * sizeof(uint32_t), kTopK * sizeof(uint64_t));
        expert_gate_batch_set_ = kernels_.set(
            {whole(quant_), expert_addresses, whole(routing_),
             whole(expert_intermediate_)});
        expert_down_batch_set_ = kernels_.set(
            {whole(expert_quant_), expert_addresses, whole(routing_),
             whole(expert_outputs_)});

        layers_.resize(kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            LayerSets& sets = layers_[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            const TensorDevice input_norm =
                tensor(prefix + "input_norm", TensorFormat::f32, kDim);
            const TensorDevice post_norm =
                tensor(prefix + "post_norm", TensorFormat::f32, kDim);
            sets.input_norm = kernels_.set(
                {whole(hidden_), input_norm.data, whole(normalized_)});
            sets.hidden_quant = kernels_.set({whole(normalized_), whole(quant_)});

            if (full_attention(layer)) {
                const TensorDevice query = tensor(prefix + "q_proj",
                    TensorFormat::q4g64t, 16384, kDim);
                const TensorDevice key = tensor(prefix + "k_proj",
                    TensorFormat::q4g64t, 512, kDim);
                const TensorDevice value = tensor(prefix + "v_proj",
                    TensorFormat::q4g64t, 512, kDim);
                const TensorDevice output = tensor(prefix + "o_proj",
                    TensorFormat::q4g64t, kDim, kLinearValue);
                const TensorDevice qnorm = tensor(prefix + "q_norm",
                    TensorFormat::f32, kHeadDim);
                const TensorDevice knorm = tensor(prefix + "k_norm",
                    TensorFormat::f32, kHeadDim);
                sets.qgate = q4_set(whole(quant_), query, whole(qgate_));
                sets.key = q4_set(whole(quant_), key, whole(key_));
                sets.value = q4_set(whole(quant_), value, whole(value_));
                sets.qk = kernels_.set({whole(qgate_), whole(key_), qnorm.data,
                                        knorm.data, whole(kv_cache_), whole(rope_)});
                sets.store_value = kernels_.set({whole(value_), whole(kv_cache_)});
                sets.attention = kernels_.set(
                    {whole(qgate_), whole(kv_cache_), whole(context_)});
                sets.head_gate = kernels_.set({whole(context_), whole(qgate_)});
                sets.context_quant = kernels_.set({whole(context_), whole(quant_)});
                sets.attention_out = q4_set(whole(quant_), output, whole(hidden_),
                                            whole(hidden_));
            } else {
                const TensorDevice qkv = tensor(prefix + "gdn_qkv",
                    TensorFormat::q4g64t, kLinearQkv, kDim);
                const TensorDevice z = tensor(prefix + "gdn_z",
                    TensorFormat::q4g64t, kLinearValue, kDim);
                const TensorDevice ab = tensor(prefix + "ab_proj",
                    TensorFormat::q4g64t, 128, kDim);
                const TensorDevice output = tensor(prefix + "gdn_out",
                    TensorFormat::q4g64t, kDim, kLinearValue);
                const TensorDevice conv = tensor(prefix + "conv",
                    TensorFormat::f32, kLinearQkv, kConvWidth);
                const TensorDevice params = tensor(prefix + "delta_params",
                    TensorFormat::f32, 256);
                const uint64_t linear = linear_index(layer);
                const DescriptorRange conv_state = arena_range(
                    conv_state_, linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                    uint64_t(kLinearQkv) * kConvWidth * 4);
                const DescriptorRange recurrent_state = arena_range(
                    recurrent_state_, linear * uint64_t(kLinearValueHeads) *
                        kLinearHeadDim * kLinearHeadDim * 4,
                    uint64_t(kLinearValueHeads) * kLinearHeadDim *
                        kLinearHeadDim * 4);
                sets.gdn_qkv = q4_set(whole(quant_), qkv, whole(mixed_qkv_));
                sets.gdn_z = q4_set(whole(quant_), z, whole(z_));
                sets.ab = q4_set(whole(quant_), ab, whole(ab_));
                sets.conv = kernels_.set({whole(mixed_qkv_), conv.data,
                                          conv_state, whole(convolved_qkv_)});
                sets.delta = kernels_.set({whole(convolved_qkv_), whole(z_),
                                           whole(ab_), recurrent_state,
                                           params.data, whole(context_)});
                sets.context_quant = kernels_.set({whole(context_), whole(quant_)});
                sets.gdn_out = q4_set(whole(quant_), output, whole(hidden_),
                                      whole(hidden_));
            }

            sets.post_norm = kernels_.set(
                {whole(hidden_), post_norm.data, whole(normalized_)});
            const TensorDevice router = tensor(prefix + "router",
                TensorFormat::q8_row, kExperts, kDim);
            sets.router_gemv = kernels_.set({whole(quant_), router.data,
                                             router.auxiliary,
                                             whole(router_logits_)});
            sets.router = kernels_.set({whole(router_logits_), whole(routing_)});

            const TensorDevice shared_gate = tensor(prefix + "shared_gate_proj",
                TensorFormat::q4g64t, kMoeDim, kDim);
            const TensorDevice shared_up = tensor(prefix + "shared_up_proj",
                TensorFormat::q4g64t, kMoeDim, kDim);
            const TensorDevice shared_down = tensor(prefix + "shared_down_proj",
                TensorFormat::q4g64t, kDim, kMoeDim);
            const TensorDevice expert_gate = tensor(prefix + "shared_expert_gate",
                TensorFormat::q4g64t, 1, kDim);
            sets.shared_gate = q4_set(whole(quant_), shared_gate,
                                      whole(shared_gate_values_));
            sets.shared_up = q4_set(whole(quant_), shared_up,
                                    whole(shared_up_values_));
            sets.shared_swiglu = kernels_.set({whole(shared_gate_values_),
                                               whole(shared_up_values_),
                                               kernels_.dummy(),
                                               whole(shared_intermediate_)});
            sets.shared_quant = kernels_.set(
                {whole(shared_intermediate_), whole(expert_quant_)});
            sets.shared_down = q4_set(whole(expert_quant_), shared_down,
                                      whole(shared_output_));
            sets.shared_expert_gate = q4_set(whole(quant_), expert_gate,
                                             whole(shared_expert_gate_));

            sets.expert_gate.resize(device_cache_.slots(layer));
            sets.expert_down.resize(device_cache_.slots(layer));
            for (uint32_t slot = 0; slot < device_cache_.slots(layer); ++slot) {
                const DescriptorRange record = device_cache_.record(layer, slot);
                sets.expert_gate[slot] = kernels_.set(
                    {whole(quant_), record, whole(routing_),
                     whole(expert_intermediate_)});
                sets.expert_down[slot] = kernels_.set(
                    {whole(expert_quant_), record, whole(routing_),
                     whole(expert_outputs_)});
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
        verify_unique_experts_ = verify_occurrences_ = 0;
        verify_reused_occurrences_ = 0;
        pre_seconds_ = acquisition_seconds_ = expert_seconds_ = 0;
    }

    uint32_t run_token(uint32_t token, uint32_t position) {
        if (position >= kMaximumContext)
            throw std::runtime_error("Qwen runtime context cap reached");
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
            signal = compute_.submit([&](VkCommandBuffer command) {
                record_attention(command, layer, position);
                record_router(command, layer);
            });
            compute_.wait(signal);
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();

            invalidate_buffer(runtime_, routing_);
            std::array<uint32_t, kTopK> experts{};
            std::array<float, kTopK> route_weights{};
            const uint32_t* routes = static_cast<const uint32_t*>(routing_.mapped);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                experts[rank] = routes[rank];
                std::memcpy(&route_weights[rank], routes + kTopK + rank,
                            sizeof(float));
                if (experts[rank] >= kExperts)
                    throw std::runtime_error("Qwen router returned invalid expert");
            }

            const auto expert_started = std::chrono::steady_clock::now();
            compute_.submit([&](VkCommandBuffer command) {
                record_shared(command, layer);
            });
            started = std::chrono::steady_clock::now();
            const DeviceExpertCache::Selection selection =
                device_cache_.resolve(layer, experts);
            selected_slots_ = selection.slots;
            std::array<void*, kTopK> direct_destinations{};
            for (uint32_t rank = 0; rank < kTopK; ++rank)
                direct_destinations[rank] = staging_[rank].mapped;
            const bool has_missing = std::any_of(
                selection.misses.begin(), selection.misses.end(),
                [](bool missing) { return missing; });
            if (tensor_split_ && progressive_experts_ && has_missing) {
                HostExpertCache::SplitProgressiveBatch batch{};
                host_cache_.begin_progressive_split(
                    layer, experts, selection.misses, direct_destinations,
                    batch);
                if (batch.sources.disk_reads) ++cold_stalled_layers_;

                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (selection.misses[rank]) continue;
                    progressive_compute_->submit([&](VkCommandBuffer command) {
                        record_expert_rank(command, layer, rank);
                    });
                    ++progressive_resident_ranks_;
                }

                uint64_t last_ready = 0;
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (!selection.misses[rank] ||
                        batch.sources.direct[rank]) continue;
                    std::memcpy(staging_[rank].mapped,
                                batch.sources.pointers[rank],
                                kExpertRecordBytes);
                    host_copy_bytes_ += kExpertRecordBytes;
                    dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                             kExpertRecordBytes);
                    last_ready = transfer_.submit([&](VkCommandBuffer command) {
                        const VkBufferCopy copy{0,
                            uint64_t(selected_slots_[rank]) * kExpertRecordBytes,
                            kExpertRecordBytes};
                        vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                            device_cache_.arena(layer).handle, 1, &copy);
                        dsv4::transfer_barrier(command,
                                               device_cache_.arena(layer));
                    });
                    transfer_bytes_ += kExpertRecordBytes;
                    progressive_compute_->submit(
                        [&](VkCommandBuffer command) {
                            record_expert_rank(command, layer, rank);
                        }, transfer_.semaphore(), last_ready);
                    ++progressive_ram_ranks_;
                }

                std::array<bool, kTopK> prefix_ready{}, suffix_ready{};
                std::array<bool, kTopK> gate_submitted{}, down_submitted{};
                const auto submit_gate = [&](uint32_t rank) {
                    dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                             kDownScale);
                    last_ready = transfer_.submit([&](VkCommandBuffer command) {
                        const VkBufferCopy copy{0,
                            uint64_t(selected_slots_[rank]) * kExpertRecordBytes,
                            kDownScale};
                        vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                            device_cache_.arena(layer).handle, 1, &copy);
                        dsv4::transfer_barrier(command,
                                               device_cache_.arena(layer));
                    });
                    progressive_compute_->submit(
                        [&](VkCommandBuffer command) {
                            record_expert_rank_gate(command, layer, rank);
                        }, transfer_.semaphore(), last_ready);
                    transfer_bytes_ += kDownScale;
                    gate_submitted[rank] = true;
                };
                const auto submit_down = [&](uint32_t rank) {
                    dsv4::flush_buffer_range(runtime_, staging_[rank],
                        kDownScale, kExpertRecordBytes - kDownScale);
                    last_ready = transfer_.submit([&](VkCommandBuffer command) {
                        const VkBufferCopy copy{kDownScale,
                            uint64_t(selected_slots_[rank]) * kExpertRecordBytes +
                                kDownScale,
                            kExpertRecordBytes - kDownScale};
                        vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                            device_cache_.arena(layer).handle, 1, &copy);
                        dsv4::transfer_barrier(command,
                                               device_cache_.arena(layer));
                    });
                    progressive_compute_->submit(
                        [&](VkCommandBuffer command) {
                            record_expert_rank_down(command, layer, rank);
                        }, transfer_.semaphore(), last_ready);
                    transfer_bytes_ += kExpertRecordBytes - kDownScale;
                    down_submitted[rank] = true;
                    ++progressive_disk_ranks_;
                };
                while (batch.reads.remaining) {
                    const auto [rank, part] =
                        host_cache_.wait_next_split(batch);
                    if (part == 0) prefix_ready[rank] = true;
                    else suffix_ready[rank] = true;
                    if (prefix_ready[rank] && !gate_submitted[rank])
                        submit_gate(rank);
                    if (suffix_ready[rank] && gate_submitted[rank] &&
                        !down_submitted[rank])
                        submit_down(rank);
                }
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    if (!selection.misses[rank] ||
                        !batch.sources.direct[rank]) continue;
                    if (!gate_submitted[rank]) submit_gate(rank);
                    if (!down_submitted[rank]) submit_down(rank);
                }
                host_cache_.finish_progressive_split(batch);
                acquisition_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - started).count();
                const uint64_t finished = progressive_compute_->submit(
                    [&](VkCommandBuffer command) { record_expert_finish(command); });
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
                write_expert_trace(position, layer, experts, route_weights,
                                   selection, batch.sources, started,
                                   std::chrono::steady_clock::now());
                ++progressive_layers_;
                expert_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - expert_started).count();
                continue;
            }
            if (progressive_experts_ && has_missing) {
                HostExpertCache::ProgressiveBatch batch{};
                host_cache_.begin_progressive(
                    layer, experts, selection.misses, direct_destinations,
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
                                device_cache_.arena(layer).handle, 1, &copy);
                        }
                        dsv4::transfer_barrier(
                            command, device_cache_.arena(layer));
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
                                device_cache_.arena(layer).handle, 1, &copy);
                            dsv4::transfer_barrier(
                                command, device_cache_.arena(layer));
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
                write_expert_trace(position, layer, experts, route_weights,
                                   selection, batch.sources, started,
                                   std::chrono::steady_clock::now());
                ++progressive_layers_;
                expert_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - expert_started).count();
                continue;
            }
            const HostExpertCache::Batch sources = host_cache_.resolve_batch(
                layer, experts, selection.misses, direct_destinations);
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
                                            device_cache_.arena(layer).handle,
                                            1, &copy);
                    }
                    dsv4::transfer_barrier(command, device_cache_.arena(layer));
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
            write_expert_trace(position, layer, experts, route_weights,
                               selection, sources, started,
                               std::chrono::steady_clock::now());
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
        }

        signal = compute_.submit([&](VkCommandBuffer command) {
            Push push{1, kDim, float_bits(1e-6f), 0};
            kernels_.dispatch(command, kernels_.p().rms, final_norm_set_, &push, 1);
            compute_barrier(command);
            push = {kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, final_quant_set_, &push,
                              kDim / 128);
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

    void record_attention(VkCommandBuffer command, uint32_t layer,
                          uint32_t position) {
        LayerSets& sets = layers_[layer];
        Push push{1, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().rms, sets.input_norm, &push, 1);
        compute_barrier(command);
        push = {kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.hidden_quant, &push,
                          kDim / 128);
        compute_barrier(command);

        if (full_attention(layer)) {
            push = {16384, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, sets.qgate, &push, 2048);
            push = {512, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, sets.key, &push, 64);
            kernels_.dispatch(command, kernels_.p().q4, sets.value, &push, 64);
            compute_barrier(command);
            push = {full_index(layer), position, kAttentionHeads, kRopeDim / 2};
            kernels_.dispatch(command, kernels_.p().qk, sets.qk, &push,
                              kAttentionHeads);
            push = {full_index(layer), position, 0, 0};
            kernels_.dispatch(command, kernels_.p().store_value,
                              sets.store_value, &push,
                              (kKvHeads * kHeadDim + 63) / 64);
            compute_barrier(command);
            push = {full_index(layer), position, kAttentionHeads, 0};
            kernels_.dispatch(command, kernels_.p().attention, sets.attention,
                              &push, kAttentionHeads);
            compute_barrier(command);
            push = {kLinearValue, kAttentionHeads, 0, 0};
            kernels_.dispatch(command, kernels_.p().head_gate, sets.head_gate,
                              &push, kLinearValue / 64);
            compute_barrier(command);
            push = {kLinearValue, 128, kLinearValue / 4, kLinearValue / 4};
            kernels_.dispatch(command, kernels_.p().quant, sets.context_quant,
                              &push, kLinearValue / 128);
            compute_barrier(command);
            push = {kDim, kLinearValue, kLinearValue / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4_residual,
                              sets.attention_out, &push, kDim / 8);
        } else {
            push = {kLinearQkv, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, sets.gdn_qkv, &push,
                              kLinearQkv / 8);
            push = {kLinearValue, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, sets.gdn_z, &push,
                              kLinearValue / 8);
            push = {128, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, sets.ab, &push, 16);
            compute_barrier(command);
            push = {kLinearQkv, kConvWidth, 0, 0};
            kernels_.dispatch(command, kernels_.p().conv, sets.conv, &push,
                              kLinearQkv / 64);
            compute_barrier(command);
            push = {kLinearValueHeads, kLinearHeadDim, 0, 0};
            kernels_.dispatch(command, kernels_.p().delta, sets.delta, &push,
                              kLinearValueHeads);
            compute_barrier(command);
            push = {kLinearValue, 128, kLinearValue / 4, kLinearValue / 4};
            kernels_.dispatch(command, kernels_.p().quant, sets.context_quant,
                              &push, kLinearValue / 128);
            compute_barrier(command);
            push = {kDim, kLinearValue, kLinearValue / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4_residual, sets.gdn_out,
                              &push, kDim / 8);
        }
        compute_barrier(command);
    }

    void write_expert_trace(
        uint32_t position, uint32_t layer,
        const std::array<uint32_t, kTopK>& experts,
        const std::array<float, kTopK>& weights,
        const DeviceExpertCache::Selection& selection,
        const HostExpertCache::Batch& sources,
        std::chrono::steady_clock::time_point started,
        std::chrono::steady_clock::time_point completed) {
        if (!expert_trace_.enabled()) return;
        uint8_t device_mask = 0, ram_mask = 0, disk_mask = 0;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint8_t bit = static_cast<uint8_t>(1u << rank);
            if (!selection.misses[rank]) device_mask |= bit;
            else if (sources.direct[rank]) disk_mask |= bit;
            else ram_mask |= bit;
        }
        expert_trace_.event(position, layer, experts, weights, device_mask,
                            ram_mask, disk_mask, selection.slots,
                            started, completed);
    }

    void record_router(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        Push push{1, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().rms, sets.post_norm, &push, 1);
        compute_barrier(command);
        push = {kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.hidden_quant, &push,
                          kDim / 128);
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
        Push push{kMoeDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().q4, sets.shared_gate, &push,
                          kMoeDim / 8);
        kernels_.dispatch(command, kernels_.p().q4, sets.shared_up, &push,
                          kMoeDim / 8);
        push = {1, kDim, kDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().q4, sets.shared_expert_gate,
                          &push, 1);
        compute_barrier(command);
        push = {kMoeDim, float_bits(3.402823466e+38f), 0, 0};
        kernels_.dispatch(command, kernels_.p().swiglu, sets.shared_swiglu,
                          &push, kMoeDim / 64);
        compute_barrier(command);
        push = {kMoeDim, 128, kMoeDim / 4, kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.shared_quant, &push,
                          kMoeDim / 128);
        compute_barrier(command);
        push = {kDim, kMoeDim, kMoeDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().q4, sets.shared_down, &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_experts(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        if (batch_experts_) {
            auto* words = static_cast<uint32_t*>(routing_.mapped);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                const DescriptorRange record =
                    device_cache_.record(layer, selected_slots_[rank]);
                VkBufferDeviceAddressInfo info{
                    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
                info.buffer = record.buffer;
                const uint64_t address =
                    vkfn::GetBufferDeviceAddress(runtime_.device, &info) +
                    record.offset;
                if (!address)
                    throw std::runtime_error(
                        "Qwen selected expert has no device address");
                std::memcpy(words + 16u + rank * 2u, &address, sizeof(address));
            }
            flush_buffer(runtime_, routing_);
            Push push{0, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate_batch,
                              expert_gate_batch_set_, &push,
                              kMoeDim / 8, kTopK);
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                Push push{rank, kDim / 4, 0, 0};
                kernels_.dispatch(command, kernels_.p().expert_gate,
                                  sets.expert_gate[selected_slots_[rank]], &push,
                                  kMoeDim / 8);
            }
        }
        compute_barrier(command);
        Push push{kTopK * kMoeDim, 128, kTopK * kMoeDim / 4,
                  kTopK * kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, expert_quant_set_, &push,
                          kTopK * kMoeDim / 128);
        compute_barrier(command);
        if (batch_experts_) {
            push = {0, kTopK * kMoeDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_down_batch,
                              expert_down_batch_set_, &push,
                              kDim / 8, kTopK);
        } else {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                push = {rank, kTopK * kMoeDim / 4, 0, 0};
                kernels_.dispatch(command, kernels_.p().expert_down,
                                  sets.expert_down[selected_slots_[rank]], &push,
                                  kDim / 8);
            }
        }
        compute_barrier(command);
        push = {kDim, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
    }

    void record_expert_rank(VkCommandBuffer command, uint32_t layer,
                            uint32_t rank) {
        record_expert_rank_gate(command, layer, rank);
        record_expert_rank_down(command, layer, rank);
    }

    void record_expert_rank_gate(VkCommandBuffer command, uint32_t layer,
                                 uint32_t rank) {
        LayerSets& sets = layers_[layer];
        Push push{rank, kDim / 4, 0, 0};
        kernels_.dispatch(command, kernels_.p().expert_gate,
                          sets.expert_gate[selected_slots_[rank]], &push,
                          kMoeDim / 8);
        compute_barrier(command);
        // The descriptor begins at this rank's packed output.  Scale indices
        // are rebased so the resulting bytes are identical to the retained
        // all-rank quantizer layout.
        const uint32_t scale_relative =
            kTopK * kMoeDim / 4u + rank * (kMoeDim / 128u) -
            rank * (kMoeDim / 4u);
        push = {kMoeDim, 128u, kMoeDim / 4u, scale_relative};
        kernels_.dispatch(command, kernels_.p().quant,
                          expert_rank_quant_sets_[rank], &push,
                          kMoeDim / 128u);
        compute_barrier(command);
    }

    void record_expert_rank_down(VkCommandBuffer command, uint32_t layer,
                                 uint32_t rank) {
        LayerSets& sets = layers_[layer];
        Push push{rank, kTopK * kMoeDim / 4u, 0, 0};
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
        if (verify2_enabled_) {
            destroy_buffer(runtime_, verify_recurrent_snapshot_);
            destroy_buffer(runtime_, verify_conv_snapshot_);
            for (Buffer& buffer : verify_quant_) destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_hidden_) destroy_buffer(runtime_, buffer);
            destroy_buffer(runtime_, verify_routing_);
            destroy_buffer(runtime_, verify_tokens_);
        }
        for (Buffer* buffer : {&rope_, &kv_cache_, &recurrent_state_,
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
    Buffer token_{}, routing_{}, hidden_{}, normalized_{};
    Buffer qgate_{}, key_{}, value_{}, context_{};
    Buffer mixed_qkv_{}, convolved_qkv_{}, z_{}, ab_{}, quant_{};
    Buffer shared_gate_values_{}, shared_up_values_{}, shared_intermediate_{};
    Buffer shared_output_{}, shared_expert_gate_{}, router_logits_{};
    Buffer expert_intermediate_{}, expert_quant_{}, expert_outputs_{};
    Buffer logits_{}, argmax_workspace_{}, conv_state_{}, recurrent_state_{};
    Buffer kv_cache_{}, rope_{};
    std::vector<Buffer> staging_;
    std::vector<Buffer> verify_staging_;
    Buffer verify_tokens_{}, verify_routing_{};
    std::array<Buffer, 2> verify_hidden_{}, verify_quant_{};
    Buffer verify_conv_snapshot_{}, verify_recurrent_snapshot_{};
    std::vector<LayerSets> layers_;
    std::array<uint32_t, kTopK> selected_slots_{};
    VkDescriptorSet embedding_set_{}, final_norm_set_{}, final_quant_set_{};
    VkDescriptorSet lm_head_set_{}, argmax_set_{}, expert_quant_set_{};
    std::array<VkDescriptorSet, kTopK> expert_rank_quant_sets_{};
    VkDescriptorSet reduce_set_{}, expert_gate_batch_set_{}, expert_down_batch_set_{};
    std::array<VkDescriptorSet, 2> verify_embedding_sets_{};
    bool batch_experts_ = false;
    bool progressive_experts_ = false;
    bool verify2_enabled_ = false;
    bool tensor_split_ = false;
    std::unique_ptr<dsv4::experiment::FiniteQueueRing<12>> progressive_compute_;
    ovllm_trace::Writer expert_trace_;
    uint64_t activation_device_bytes_ = 0;
    uint64_t transfer_bytes_ = 0, host_copy_bytes_ = 0;
    uint64_t cold_stalled_layers_ = 0;
    uint64_t progressive_layers_ = 0;
    uint64_t progressive_resident_ranks_ = 0;
    uint64_t progressive_ram_ranks_ = 0;
    uint64_t progressive_disk_ranks_ = 0;
    uint64_t verify_unique_experts_ = 0;
    uint64_t verify_occurrences_ = 0;
    uint64_t verify_reused_occurrences_ = 0;
    double decode_seconds_ = 0, pre_seconds_ = 0;
    double acquisition_seconds_ = 0, expert_seconds_ = 0;
};

static uint64_t ram_budget() {
    const char* text = std::getenv("QWEN_RAM_GIB");
    const double gib = text ? std::stod(text) : 24.0;
    if (gib < 2.0 || gib > 56.0)
        throw std::runtime_error("QWEN_RAM_GIB must be 2..56");
    return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

static uint32_t device_slots() {
    const char* text = std::getenv("QWEN_DEVICE_SLOTS_PER_LAYER");
    const uint32_t slots = text ? static_cast<uint32_t>(std::stoul(text)) : 17;
    if (slots < kTopK || slots > 32)
        throw std::runtime_error("QWEN_DEVICE_SLOTS_PER_LAYER must be 8..32");
    return slots;
}

} // namespace qwen35

int qwen35_cli_main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: amd_qwen35.exe <runtime-dir> "
                         "<prompt|--inspect|--tokenize> [new-tokens]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file((directory / "tokenizer.ovb").string());
        qwen35::Tokenizer tokenizer(tokenizer_file);
        if (std::strcmp(argv[2], "--tokenize") == 0) {
            if (argc < 4) throw std::runtime_error("tokenize text required");
            const bool thinking = std::getenv("QWEN_NO_THINK") == nullptr;
            const std::vector<uint32_t> tokens =
                tokenizer.chat_prompt(argv[3], thinking);
            std::cout << "tokens:";
            for (uint32_t token : tokens) std::cout << ' ' << token;
            std::cout << '\n';
            return 0;
        }
        qwen35::SharedIndex index(directory / "model-q4g64.ovs");
        qwen35::ExpertFile inspect_experts(directory / "experts-q4g64.ovx");
        if (std::strcmp(argv[2], "--inspect") == 0) {
            std::cout << "Qwen3.5-122B-A10B runtime containers validated\n";
            return 0;
        }

        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << '\n';
        const uint32_t count = argc >= 4
            ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint64_t budget = qwen35::ram_budget();
        const uint32_t slots = qwen35::device_slots();
        const bool thinking = std::getenv("QWEN_NO_THINK") == nullptr;
        const std::vector<uint32_t> prompt =
            tokenizer.chat_prompt(argv[2], thinking);
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
            qwen35::QwenEngine engine(
                runtime, index, directory / "experts-q4g64.ovx",
                std::filesystem::absolute(argv[0]).parent_path(), budget, slots);
            std::cout << "precision: Q4G64T experts/shared/dense, "
                         "Q8 embedding/head/router\n"
                      << "RAM budget: "
                      << double(budget) / double(1ull << 30) << " GiB\n"
                      << "expert slots device/RAM: ";
            if (engine.device_profiled())
                std::cout << engine.device_total_slots() << " total ("
                          << engine.device_minimum_slots() << ".."
                          << engine.device_maximum_slots() << " per layer, profiled)";
            else
                std::cout << slots << " per layer";
            std::cout << " / " << engine.host_slots() << " global\n"
                      << "host expert eviction: "
                      << (std::getenv("QWEN_HOST_LRU") ? "LRU" : "LFU") << '\n';
            result = engine.generate(tokenizer, prompt, count);
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
        std::cerr << "Qwen3.5 runtime error: " << error.what() << '\n';
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}

#ifndef OVLLM_QWEN35_RUNTIME_ONLY
int main(int argc, char** argv) {
    return qwen35_cli_main(argc, argv);
}
#endif
