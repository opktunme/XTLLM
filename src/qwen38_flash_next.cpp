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
#include "xtllm_chat.hpp"

// Qwen3.8-Flash-Next text-only executor.  This is deliberately separate from
// the retained DeepSeek and Step-3.7 executors.  It reuses their finite Vulkan
// queues and Q4 expert-acquisition design, but all model math below follows the
// authoritative Qwen4-Exp text architecture.  It supports the exact
// short-context (<= QSA budget) path, where sparse attention selects every
// causal token, plus four-stream gated residuals and the official FP8 PLE.
namespace qwen38 {

constexpr uint32_t kDim = 2560;
constexpr uint32_t kHcCount = 4;
constexpr uint32_t kHcDim = kDim * kHcCount;
constexpr uint32_t kHcLowrank = 320;
constexpr uint32_t kMoeDim = 640;
constexpr uint32_t kLayers = 48;
constexpr uint32_t kExperts = 512;
constexpr uint32_t kTopK = 10;
constexpr uint32_t kVocabulary = 248320;
constexpr uint32_t kBaseVocabulary = 248044;
constexpr uint32_t kFullLayers = 12;
constexpr uint32_t kLinearLayers = 36;
constexpr uint32_t kAttentionHeads = 24;
constexpr uint32_t kKvHeads = 2;
constexpr uint32_t kHeadDim = 256;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kLinearKeyHeads = 16;
constexpr uint32_t kLinearValueHeads = 48;
constexpr uint32_t kLinearHeadDim = 128;
constexpr uint32_t kLinearQkv = 10240;
constexpr uint32_t kLinearValue = 6144;
constexpr uint32_t kConvWidth = 4;
#ifdef OVLLM_LONG_CONTEXT_FORK
constexpr uint32_t kDefaultContext = 2048;
constexpr uint32_t kAttentionChunk = 1024;
constexpr uint32_t kAttentionPartialStride = kHeadDim + 2;
#else
constexpr uint32_t kMaximumContext = 2048;
#endif

constexpr uint64_t kHeaderBytes = 4096;
constexpr uint64_t kGateScale = 0;
constexpr uint64_t kGateWeight = 51200;
#ifdef OVLLM_QWEN38_Q3_EXPERTS
constexpr uint64_t kUpScale = 665600;
constexpr uint64_t kUpWeight = 716800;
constexpr uint64_t kDownScale = 1331200;
constexpr uint64_t kDownWeight = 1382400;
constexpr uint64_t kExpertRecordBytes = 1998848;
#else
constexpr uint64_t kUpScale = 870400;
constexpr uint64_t kUpWeight = 921600;
constexpr uint64_t kDownScale = 1740800;
constexpr uint64_t kDownWeight = 1792000;
constexpr uint64_t kExpertRecordBytes = 2613248;
#endif
static_assert(kExpertRecordBytes % 4096 == 0);

constexpr uint32_t kPleParts = 128;
constexpr uint32_t kPleRowsPerPart = 2500012;
constexpr uint32_t kPleRowBytes = 160;
constexpr uint64_t kPleRows = uint64_t(kPleParts) * kPleRowsPerPart;
constexpr uint32_t kPleHeads = 16;
constexpr uint32_t kPleHistory = 10;

constexpr uint32_t kVerifyBatch = 4;
constexpr uint32_t kVerifyDimQuantU32 = kDim / 4 + kDim / 128;
constexpr uint32_t kVerifyHcQuantU32 = kHcDim / 4 + kHcDim / 128;
constexpr uint32_t kVerifyLowQuantU32 = kHcLowrank / 4 +
    (kHcLowrank + 127) / 128;
constexpr uint32_t kVerifyContextQuantU32 = kLinearValue / 4 +
    kLinearValue / 128;
constexpr uint32_t kVerifyMoeQuantU32 = kMoeDim / 4 + kMoeDim / 128;

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
        const char* expected_magic = mtp_container ? "OQ38MTP\0" : "OQ38SHR\0";
        const uint32_t expected_layers = mtp_container ? 1u : kLayers;
        if (std::memcmp(header_.magic, expected_magic, 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.tensor_entry_bytes != sizeof(TensorEntry) ||
            header_.dimension != kDim || header_.moe_dimension != kMoeDim ||
            header_.layers != expected_layers || header_.heads != kAttentionHeads ||
            header_.kv_heads != kKvHeads || header_.head_dimension != kHeadDim ||
            header_.vocabulary != kVocabulary || header_.experts != kExperts ||
            header_.top_k != kTopK ||
#ifdef OVLLM_QWEN38_Q3_EXPERTS
            // The shared tensor payload is unchanged for the routed-expert
            // Q3 variant.  Its retained header describes the companion Q4
            // expert stream; ExpertFile independently validates the Q3 ABI.
            header_.expert_record_bytes != 2613248)
#else
            header_.expert_record_bytes != kExpertRecordBytes)
#endif
            throw std::runtime_error("Unsupported Qwen3.8 shared container");
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

#pragma pack(push, 1)
struct PleHeader {
    char magic[8];
    uint32_t version, header_bytes, parts, rows_per_part;
    uint32_t row_bytes, total_rows, scale_bits, reserved;
    uint64_t data_offset, data_bytes, file_bytes, revision;
    uint64_t reserved_qword[4];
};
#pragma pack(pop)
static_assert(sizeof(PleHeader) == 104);

class PleLookup {
public:
    struct State {
        std::array<uint32_t, 2> history{};
        uint32_t history_size = 0;
    };
    explicit PleLookup(const std::filesystem::path& path) {
        std::ifstream metadata(path, std::ios::binary);
        if (!metadata) throw std::runtime_error("Could not open Qwen3.8 PLE table");
        metadata.read(reinterpret_cast<char*>(&header_), sizeof(header_));
        if (!metadata || std::memcmp(header_.magic, "OQ38PLE\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kHeaderBytes ||
            header_.parts != kPleParts ||
            header_.rows_per_part != kPleRowsPerPart ||
            header_.row_bytes != kPleRowBytes ||
            header_.total_rows != kPleRows ||
            header_.data_offset != kHeaderBytes ||
            header_.file_bytes != std::filesystem::file_size(path))
            throw std::runtime_error("Unsupported Qwen3.8 PLE table");
        std::memcpy(&scale_, &header_.scale_bits, sizeof(scale_));
        file_ = open_unbuffered(path);
        aligned_ = static_cast<uint8_t*>(VirtualAlloc(
            nullptr, 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        if (!aligned_) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::runtime_error("Could not allocate aligned PLE read buffer");
        }
        make_fp8_table();
    }

    ~PleLookup() {
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        if (aligned_) VirtualFree(aligned_, 0, MEM_RELEASE);
    }

    void lookup(uint32_t token, float* output) {
        const uint32_t previous1 = history_size_ >= 1 ? history_[history_size_ - 1] : kEndOfText;
        const uint32_t previous2 = history_size_ >= 2 ? history_[history_size_ - 2] : kEndOfText;
        const std::array<uint32_t, 3> tokens{token, previous1, previous2};
        for (uint32_t head = 0; head < kPleHeads; ++head) {
            const uint32_t order = head < 8 ? 2u : 3u;
            uint64_t mixed = 0;
            for (uint32_t position = 0; position < order; ++position)
                mixed ^= uint64_t(tokens[position]) * kMultipliers[position];
            int64_t signed_mixed = 0;
            std::memcpy(&signed_mixed, &mixed, sizeof(mixed));
            int64_t remainder = signed_mixed % int64_t(kVocabSizes[head]);
            if (remainder < 0) remainder += kVocabSizes[head];
            const uint64_t row = kOffsets[head] + uint64_t(remainder);
            read_row(row, output + head * kPleRowBytes);
        }
        if (token == kEndOfText) {
            history_size_ = 0;
        } else if (history_size_ < history_.size()) {
            history_[history_size_++] = token;
        } else {
            history_[0] = history_[1];
            history_[1] = token;
        }
    }

    uint64_t bytes_read() const { return bytes_read_; }
    void reset_metrics() { bytes_read_ = 0; }
    State state() const { return {history_, history_size_}; }
    void restore(const State& state) {
        history_ = state.history;
        history_size_ = state.history_size;
    }

private:
    void make_fp8_table() {
        for (uint32_t encoded = 0; encoded < 256; ++encoded) {
            const float sign = (encoded & 0x80u) ? -1.0f : 1.0f;
            const uint32_t exponent = (encoded >> 3u) & 15u;
            const uint32_t mantissa = encoded & 7u;
            float value = 0;
            if (exponent == 0)
                value = (float(mantissa) / 8.0f) * std::ldexp(1.0f, -6);
            else if (exponent == 15 && mantissa == 7)
                value = 0; // Checkpoint contains finite weights; keep NaN defensive.
            else
                value = (1.0f + float(mantissa) / 8.0f) *
                    std::ldexp(1.0f, int(exponent) - 7);
            fp8_[encoded] = sign * value * scale_;
        }
    }

    void read_row(uint64_t row, float* output) {
        if (row >= kPleRows) throw std::runtime_error("PLE row out of range");
        const uint64_t offset = header_.data_offset + row * kPleRowBytes;
        const uint64_t aligned_offset = offset & ~uint64_t(4095);
        const uint32_t within = static_cast<uint32_t>(offset - aligned_offset);
        const uint64_t bytes = (uint64_t(within) + kPleRowBytes + 4095) & ~uint64_t(4095);
        read_unbuffered(file_, aligned_offset, aligned_, bytes);
        bytes_read_ += bytes;
        for (uint32_t i = 0; i < kPleRowBytes; ++i)
            output[i] = fp8_[aligned_[within + i]];
    }

    inline static constexpr std::array<uint64_t, 3> kMultipliers{
        23703573157769ull, 20109073645365ull, 8052911324071ull};
    inline static constexpr std::array<uint32_t, 16> kVocabSizes{
        20000003u,20000023u,20000033u,20000047u,20000059u,20000063u,
        20000069u,20000077u,20000081u,20000093u,20000107u,20000147u,
        20000153u,20000159u,20000161u,20000171u};
    inline static constexpr std::array<uint64_t, 16> kOffsets{
        0ull,20000003ull,40000026ull,60000059ull,80000106ull,100000165ull,
        120000228ull,140000297ull,160000374ull,180000455ull,200000548ull,
        220000655ull,240000802ull,260000955ull,280001114ull,300001275ull};
    PleHeader header_{};
    HANDLE file_ = INVALID_HANDLE_VALUE;
    uint8_t* aligned_ = nullptr;
    float scale_ = 0;
    std::array<float, 256> fp8_{};
    std::array<uint32_t, 2> history_{};
    uint32_t history_size_ = 0;
    uint64_t bytes_read_ = 0;
};

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
        std::filesystem::path transcript_path;
        if (xtllm_chat::referenced_path(user_text, transcript_path))
            return xtllm_chat::render_im(
                xtllm_chat::read(transcript_path),
                [this](const std::string& value) { return encode_text(value); },
                kImStart, kImEnd, kThink, kEndThink, thinking,
                xtllm_chat::Suffix::qwen_think, false, system_text);
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

    explicit ExpertFile(const std::filesystem::path& path) : path_(path) {
        std::ifstream input(path, std::ios::binary);
        if (!input) throw std::runtime_error("Could not open Qwen expert container");
        read_at(input, 0, &header_, sizeof(header_), "expert header");
        if (std::memcmp(header_.magic, "OQ38EXP\0", 8) != 0 ||
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

private:
    std::filesystem::path path_;
    ExpertHeader header_{};
    HANDLE file_ = INVALID_HANDLE_VALUE;
};

class HostExpertCache {
public:
    struct Batch {
        std::array<const uint8_t*, kTopK> pointers{};
        std::array<VkBuffer, kTopK> imported_buffers{};
        std::array<VkDeviceSize, kTopK> imported_offsets{};
        std::array<bool, kTopK> direct{};
        uint32_t disk_reads = 0;
    };
    struct ProgressiveBatch {
        Batch sources{};
        ExpertFile::AsyncBatch reads{};
        std::array<bool, kTopK> disk_pending{};
        bool active = false;
    };
    struct Many {
        std::array<const uint8_t*, 4 * kTopK> pointers{};
        uint32_t disk_reads = 0;
    };

    HostExpertCache(const Runtime& runtime, ExpertFile& file,
                    uint64_t budget_bytes)
        : runtime_(runtime), file_(file) {
        tiny_lfu_ = std::getenv("QWEN38_HOST_TINYLFU") != nullptr;
        const uint64_t reserve = budget_bytes > 512ull * 1024 * 1024
            ? budget_bytes - 512ull * 1024 * 1024 : 0;
        slots_ = static_cast<uint32_t>(std::min<uint64_t>(
            reserve / kExpertRecordBytes, uint64_t(kLayers) * kExperts));
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen RAM budget is too small");
        bytes_ = uint64_t(slots_) * kExpertRecordBytes;
        import_requested_ =
            std::getenv("QWEN38_IMPORT_RAM_CACHE") != nullptr;
        if (import_requested_) {
            blocks_.resize((slots_ + kHostBlockRecords - 1u) /
                           kHostBlockRecords);
            import_limit_blocks_ = static_cast<uint32_t>(blocks_.size());
            if (const char* limit =
                    std::getenv("QWEN38_IMPORT_RAM_BLOCKS")) {
                const uint64_t requested = std::stoull(limit);
                if (requested == 0 || requested > blocks_.size())
                    throw std::runtime_error(
                        "QWEN38_IMPORT_RAM_BLOCKS is out of range");
                import_limit_blocks_ = static_cast<uint32_t>(requested);
            }
            if (const char* start =
                    std::getenv("QWEN38_IMPORT_RAM_BLOCK_START")) {
                const uint64_t requested = std::stoull(start);
                if (requested >= blocks_.size() ||
                    requested + import_limit_blocks_ > blocks_.size())
                    throw std::runtime_error(
                        "QWEN38_IMPORT_RAM_BLOCK_START is out of range");
                import_start_block_ = static_cast<uint32_t>(requested);
            }
        } else {
            base_ = static_cast<uint8_t*>(
                VirtualAlloc(nullptr, bytes_, MEM_RESERVE, PAGE_READWRITE));
            if (!base_)
                throw std::runtime_error("Could not reserve Qwen RAM cache");
        }
        entries_.resize(slots_);
        locations_.assign(uint64_t(kLayers) * kExperts, -1);
        frequency_.assign(locations_.size(), 0);
    }

    ~HostExpertCache() {
        for (HostBlock& block : blocks_) {
            if (block.imported.buffer.handle)
                destroy_buffer(runtime_, block.imported.buffer);
            if (block.mapping) {
                if (block.data) UnmapViewOfFile(block.data);
                CloseHandle(block.mapping);
            } else if (block.data) {
                VirtualFree(block.data, 0, MEM_RELEASE);
            }
        }
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
            uint8_t* pointer = ensure_record(slot, true);
            const uint32_t layer = key / kExperts;
            const uint32_t expert = key % kExperts;
            offsets.push_back(file_.offset(layer, expert));
            destinations.push_back(pointer);
            pending_slots.push_back(slot);
            pending_keys.push_back(key);
            if (offsets.size() == kTopK) flush();
        }
        flush();
        if (import_requested_)
            std::cout << "RAM cache Vulkan import: "
                      << double(imported_bytes_) / double(1ull << 30)
                      << " GiB in " << imported_blocks_
                      << " direct-transfer blocks\n";
        seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return added;
    }

    Many resolve_many(uint32_t layer, const uint32_t* experts,
                      uint32_t count) {
        if (count > 4u * kTopK)
            throw std::runtime_error("Too many Qwen expert-cache requests");
        Many result{};
        std::vector<bool> reserved(slots_);
        std::vector<uint64_t> offsets;
        std::vector<void*> destinations;
        offsets.reserve(kTopK);
        destinations.reserve(kTopK);
        auto flush_reads = [&]() {
            if (offsets.empty()) return;
            file_.read_batch(offsets, destinations);
            result.disk_reads += static_cast<uint32_t>(offsets.size());
            disk_bytes_ += uint64_t(offsets.size()) * kExpertRecordBytes;
            offsets.clear();
            destinations.clear();
        };
        for (uint32_t index = 0; index < count; ++index) {
            const uint32_t expert = experts[index];
            if (expert >= kExperts)
                throw std::runtime_error("Invalid Qwen many-cache expert");
            const uint32_t key = layer * kExperts + expert;
            ++frequency_[key];
            const int32_t location = locations_[key];
            if (location >= 0) {
                ++hits_;
                entries_[location].age = ++clock_;
                reserved[location] = true;
                result.pointers[index] = record_pointer(location);
                continue;
            }
            ++misses_;
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
                throw std::runtime_error("No Qwen many-cache victim");
            reserved[victim] = true;
            Entry& entry = entries_[victim];
            if (entry.key >= 0) locations_[entry.key] = -1;
            uint8_t* pointer = ensure_record(victim, false);
            entry.key = static_cast<int32_t>(key);
            entry.age = ++clock_;
            locations_[key] = static_cast<int32_t>(victim);
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
                result.pointers[rank] = record_pointer(location);
                set_imported_source(result, rank, location);
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
            uint8_t* pointer = ensure_record(victim, false);
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
    uint64_t imported_bytes() const { return imported_bytes_; }
    uint64_t disk_bytes() const { return disk_bytes_; }
    uint64_t committed_bytes() const {
        return uint64_t(committed_) * kExpertRecordBytes;
    }
    uint32_t capacity() const { return slots_; }
    void reset_metrics() {
        hits_ = misses_ = disk_bytes_ = admission_bypasses_ = 0;
    }

private:
    static constexpr uint32_t kHostBlockRecords = 32;
    struct Entry { int32_t key = -1; uint64_t age = 0; bool committed = false; };
    struct HostBlock {
        uint8_t* data = nullptr;
        HANDLE mapping = nullptr;
        uint32_t records = 0;
        dsv4::Dsv4ImportedRange imported{};
    };

    uint8_t* ensure_record(uint32_t slot, bool top_off) {
        if (slot >= slots_)
            throw std::runtime_error("Invalid Qwen RAM-cache slot");
        uint8_t* pointer = nullptr;
        if (import_requested_) {
            const uint32_t block_index = slot / kHostBlockRecords;
            HostBlock& block = blocks_[block_index];
            if (!block.data) {
                const uint32_t first = block_index * kHostBlockRecords;
                block.records = std::min<uint32_t>(
                    kHostBlockRecords, slots_ - first);
                const uint64_t block_bytes =
                    uint64_t(block.records) * kExpertRecordBytes;
                if (block_index >= import_start_block_ &&
                    block_index - import_start_block_ < import_limit_blocks_) {
                    block.mapping = CreateFileMappingW(
                        INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                        static_cast<DWORD>(block_bytes >> 32u),
                        static_cast<DWORD>(block_bytes), nullptr);
                    if (!block.mapping)
                        throw std::runtime_error(
                            "Qwen foreign RAM-cache mapping allocation failed");
                    block.data = static_cast<uint8_t*>(MapViewOfFile(
                        block.mapping, FILE_MAP_READ | FILE_MAP_WRITE,
                        0, 0, static_cast<SIZE_T>(block_bytes)));
                    if (!block.data)
                        throw std::runtime_error(
                            "Qwen foreign RAM-cache mapping view failed");
                    try {
                        block.imported = dsv4::import_dsv4_host_range(
                            runtime_, block.data, block_bytes,
                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
                            true);
                    } catch (const std::exception& error) {
                        throw std::runtime_error(
                            "Qwen foreign RAM-cache import failed at block " +
                            std::to_string(block_index) + " after " +
                            std::to_string(imported_blocks_) +
                            " successful blocks: " + error.what());
                    }
                    imported_bytes_ += block_bytes;
                    ++imported_blocks_;
                } else {
                    block.data = static_cast<uint8_t*>(VirtualAlloc(
                        nullptr, block_bytes, MEM_RESERVE | MEM_COMMIT,
                        PAGE_READWRITE));
                    if (!block.data)
                        throw std::runtime_error(
                            "Qwen hybrid RAM-cache block allocation failed");
                }
            }
            pointer = block.data +
                uint64_t(slot % kHostBlockRecords) * kExpertRecordBytes;
        } else {
            pointer = base_ + uint64_t(slot) * kExpertRecordBytes;
            if (!entries_[slot].committed &&
                !VirtualAlloc(pointer, kExpertRecordBytes, MEM_COMMIT,
                              PAGE_READWRITE))
                throw std::runtime_error(top_off
                    ? "Qwen RAM-cache top-off commit failed"
                    : "Qwen RAM-cache commit failed");
        }
        if (!entries_[slot].committed) {
            entries_[slot].committed = true;
            ++committed_;
        }
        return pointer;
    }

    uint8_t* record_pointer(uint32_t slot) const {
        if (slot >= slots_ || !entries_[slot].committed)
            throw std::runtime_error("Empty Qwen RAM-cache slot");
        if (!import_requested_)
            return base_ + uint64_t(slot) * kExpertRecordBytes;
        const HostBlock& block = blocks_[slot / kHostBlockRecords];
        return block.data +
            uint64_t(slot % kHostBlockRecords) * kExpertRecordBytes;
    }

    void set_imported_source(Batch& batch, uint32_t rank,
                             uint32_t slot) const {
        if (!import_requested_) return;
        const HostBlock& block = blocks_[slot / kHostBlockRecords];
        if (!block.imported.buffer.handle) return;
        batch.imported_buffers[rank] = block.imported.buffer.handle;
        batch.imported_offsets[rank] =
            uint64_t(slot % kHostBlockRecords) * kExpertRecordBytes;
    }

    const Runtime& runtime_;
    ExpertFile& file_;
    uint8_t* base_ = nullptr;
    std::vector<HostBlock> blocks_;
    uint64_t bytes_ = 0;
    uint64_t imported_bytes_ = 0;
    uint32_t slots_ = 0, committed_ = 0;
    uint32_t imported_blocks_ = 0;
    uint32_t import_limit_blocks_ = 0;
    uint32_t import_start_block_ = 0;
    std::vector<Entry> entries_;
    std::vector<int32_t> locations_;
    std::vector<uint32_t> frequency_;
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, disk_bytes_ = 0;
    uint64_t admission_bypasses_ = 0;
    bool tiny_lfu_ = false;
    bool import_requested_ = false;
};

class DeviceExpertCache {
public:
    DeviceExpertCache(const Runtime& runtime, uint32_t slots)
        : runtime_(runtime), slots_(slots) {
        if (slots_ < kTopK)
            throw std::runtime_error("Qwen device expert cache is too small");
        std::array<uint32_t, kLayers> layer_slots{};
        layer_slots.fill(slots_);
        if (const char* configured =
                std::getenv("QWEN38_DEVICE_LAYER_SLOTS")) {
            std::stringstream stream(configured);
            std::string item;
            uint32_t layer = 0;
            uint64_t total = 0;
            while (std::getline(stream, item, ',')) {
                if (layer >= kLayers || item.empty())
                    throw std::runtime_error(
                        "QWEN38_DEVICE_LAYER_SLOTS needs 48 values");
                const uint32_t value = static_cast<uint32_t>(std::stoul(item));
                if (value < kTopK || value > kExperts)
                    throw std::runtime_error(
                        "QWEN38_DEVICE_LAYER_SLOTS value is out of range");
                layer_slots[layer++] = value;
                total += value;
            }
            if (layer != kLayers || total != uint64_t(slots_) * kLayers)
                throw std::runtime_error(
                    "QWEN38_DEVICE_LAYER_SLOTS must contain 48 values with the uniform-slot total");
        }
        layers_.resize(kLayers);
        for (uint32_t index = 0; index < kLayers; ++index)
            layers_[index].entries.resize(layer_slots[index]);
        std::array<uint32_t, kLayers> allocation_order{};
        for (uint32_t index = 0; index < kLayers; ++index)
            allocation_order[index] = index;
        std::stable_sort(allocation_order.begin(), allocation_order.end(),
            [&](uint32_t left, uint32_t right) {
                return layer_slots[left] > layer_slots[right];
            });
        // Large arenas first avoid leaving the AMD heap with many small gaps
        // before the verifier's recurrent-state snapshots are allocated.
        for (uint32_t index : allocation_order) {
            Layer& layer = layers_[index];
            layer.arena = create_device_buffer(
                runtime, uint64_t(layer_slots[index]) * kExpertRecordBytes);
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
    struct Batch4 {
        std::array<std::array<uint32_t, kTopK>, 4> slots{};
        std::array<std::array<uint32_t, kTopK>, 4> unique_indices{};
        std::array<uint32_t, 4 * kTopK> unique_experts{};
        std::array<uint32_t, 4 * kTopK> unique_slots{};
        std::array<bool, 4 * kTopK> unique_misses{};
        uint32_t unique_count = 0;
        uint32_t reused_occurrences = 0;
    };

    Selection resolve(uint32_t layer_index,
                      const std::array<uint32_t, kTopK>& experts) {
        Layer& layer = layers_.at(layer_index);
        Selection result{};
        const uint32_t layer_slots = static_cast<uint32_t>(layer.entries.size());
        std::vector<bool> reserved(layer_slots);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            ++layer.frequency[experts[rank]];
            result.slots[rank] = UINT32_MAX;
            for (uint32_t slot = 0; slot < layer_slots; ++slot) {
                if (layer.entries[slot].expert == int32_t(experts[rank])) {
                    result.slots[rank] = slot;
                    reserved[slot] = true;
                    layer.entries[slot].age = ++clock_;
                    ++hits_;
                    ++layer_hits_[layer_index];
                    break;
                }
            }
        }
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (result.slots[rank] != UINT32_MAX) continue;
            ++misses_;
            ++layer_misses_[layer_index];
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < layer_slots; ++slot) {
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

    Batch4 resolve_four(
        uint32_t layer_index,
        const std::array<std::array<uint32_t, kTopK>, 4>& experts,
        uint32_t active = kTopK) {
        Layer& layer = layers_.at(layer_index);
        const uint32_t layer_slots = static_cast<uint32_t>(layer.entries.size());
        Batch4 result{};
        for (uint32_t row = 0; row < 4; ++row) {
            for (uint32_t rank = 0; rank < active; ++rank) {
                const uint32_t expert = experts[row][rank];
                uint32_t unique = 0;
                while (unique < result.unique_count &&
                       result.unique_experts[unique] != expert)
                    ++unique;
                if (unique == result.unique_count) {
                    result.unique_experts[result.unique_count++] = expert;
                } else {
                    ++result.reused_occurrences;
                }
                result.unique_indices[row][rank] = unique;
            }
        }
        std::vector<bool> reserved(layer_slots);
        for (uint32_t unique = 0; unique < result.unique_count; ++unique) {
            const uint32_t expert = result.unique_experts[unique];
            ++layer.frequency[expert];
            result.unique_slots[unique] = UINT32_MAX;
            for (uint32_t slot = 0; slot < layer_slots; ++slot) {
                if (layer.entries[slot].expert != int32_t(expert)) continue;
                result.unique_slots[unique] = slot;
                reserved[slot] = true;
                layer.entries[slot].age = ++clock_;
                ++hits_;
                ++layer_hits_[layer_index];
                break;
            }
        }
        for (uint32_t unique = 0; unique < result.unique_count; ++unique) {
            if (result.unique_slots[unique] != UINT32_MAX) continue;
            ++misses_;
            ++layer_misses_[layer_index];
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < layer_slots; ++slot) {
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
                throw std::runtime_error("No Qwen batch4 device-cache victim");
            layer.entries[victim].expert =
                int32_t(result.unique_experts[unique]);
            layer.entries[victim].age = ++clock_;
            reserved[victim] = true;
            result.unique_slots[unique] = victim;
            result.unique_misses[unique] = true;
        }
        for (uint32_t row = 0; row < 4; ++row)
            for (uint32_t rank = 0; rank < active; ++rank)
                result.slots[row][rank] =
                    result.unique_slots[result.unique_indices[row][rank]];
        return result;
    }

    DescriptorRange record(uint32_t layer, uint32_t slot) const {
        return arena_range(layers_.at(layer).arena,
                           uint64_t(slot) * kExpertRecordBytes,
                           kExpertRecordBytes);
    }
    Buffer& arena(uint32_t layer) { return layers_.at(layer).arena; }
    uint32_t slots() const { return slots_; }
    uint32_t layer_slots(uint32_t layer) const {
        return static_cast<uint32_t>(layers_.at(layer).entries.size());
    }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    const std::array<uint64_t, kLayers>& layer_misses() const {
        return layer_misses_;
    }
    uint64_t device_bytes() const { return device_bytes_; }
    void reset_metrics() {
        hits_ = misses_ = 0;
        layer_hits_.fill(0);
        layer_misses_.fill(0);
    }

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
    std::array<uint64_t, kLayers> layer_hits_{};
    std::array<uint64_t, kLayers> layer_misses_{};
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, device_bytes_ = 0;
};

class ParallelCopyPool {
public:
    struct Task { void* destination; const void* source; size_t bytes; };

    explicit ParallelCopyPool(bool enabled = false) : enabled_(enabled) {
        if (enabled_) {
            worker_count_ = 3;
            if (const char* configured = std::getenv("QWEN38_COPY_WORKERS")) {
                worker_count_ = static_cast<uint32_t>(std::stoul(configured));
                if (worker_count_ < 1 || worker_count_ > workers_.size())
                    throw std::runtime_error(
                        "QWEN38_COPY_WORKERS must be 1..3");
            }
            for (uint32_t index = 0; index < worker_count_; ++index)
                workers_[index] = std::thread([this] { worker_loop(); });
        }
    }
    ~ParallelCopyPool() {
        stop_.store(true, std::memory_order_release);
        activation_.notify_all();
        for (std::thread& worker : workers_)
            if (worker.joinable()) worker.join();
    }
    void copy(const std::vector<Task>& tasks) {
        if (tasks.empty()) return;
        if (tasks.size() == 1) {
            std::memcpy(tasks[0].destination, tasks[0].source,
                        tasks[0].bytes);
            return;
        }
        count_ = static_cast<uint32_t>(tasks.size());
        for (uint32_t index = 0; index < count_; ++index)
            tasks_[index] = tasks[index];
        next_.store(0, std::memory_order_relaxed);
        departed_.store(0, std::memory_order_relaxed);
        const uint64_t previous =
            generation_.fetch_add(1, std::memory_order_release);
        if (previous == 0) activation_.notify_all();
        perform_jobs();
        while (departed_.load(std::memory_order_acquire) !=
               worker_count_ + 1u)
            YieldProcessor();
    }

private:
    void perform_jobs() {
        for (;;) {
            const uint32_t index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) break;
            const Task& task = tasks_[index];
            std::memcpy(task.destination, task.source, task.bytes);
        }
        departed_.fetch_add(1, std::memory_order_release);
    }
    void worker_loop() {
        {
            std::unique_lock<std::mutex> lock(activation_mutex_);
            activation_.wait(lock, [&] {
                return stop_.load(std::memory_order_acquire) ||
                       generation_.load(std::memory_order_acquire) != 0;
            });
        }
        uint64_t observed = 0;
        while (!stop_.load(std::memory_order_acquire)) {
            const uint64_t generation =
                generation_.load(std::memory_order_acquire);
            if (generation == observed) {
                YieldProcessor();
                continue;
            }
            observed = generation;
            if (!stop_.load(std::memory_order_acquire)) perform_jobs();
        }
    }

    bool enabled_ = false;
    std::array<std::thread, 3> workers_{};
    std::array<Task, 4 * kTopK> tasks_{};
    std::atomic<uint32_t> next_{0}, departed_{0};
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool> stop_{false};
    uint32_t count_ = 0;
    uint32_t worker_count_ = 0;
    std::mutex activation_mutex_;
    std::condition_variable activation_;
};

struct Pipelines {
    VkPipeline embedding{}, rms{}, quant{}, q4{}, q4_small{}, q4_one{};
    VkPipeline q4_residual{}, q8{}, q4_batch4{}, q8_batch4{};
    VkPipeline swiglu{}, router{}, expert_gate{}, expert_down{}, reduce{};
    VkPipeline expert_gate_batch{}, expert_down_batch{};
    VkPipeline expert_gate_verify4{}, expert_down_verify4{};
    VkPipeline qk{}, store_value{}, attention{}, head_gate{};
#ifdef OVLLM_LONG_CONTEXT_FORK
    VkPipeline attention_reduce{};
#endif
    VkPipeline conv{}, delta{}, argmax{};
    VkPipeline group_rms{}, hc_act{}, hc_mix{}, hc_inject{};
    VkPipeline group_rms_batch4{}, hc_mix_batch4{}, hc_inject_batch4{};
    VkPipeline ple_gate{}, ple_conv_add{}, repeat_hc{};
};

class Kernels {
public:
    Kernels(const Runtime& runtime, const std::filesystem::path& directory)
        : runtime_(runtime), resources_(create_compute_resources(runtime, 24000)),
          dummy_(create_device_buffer(runtime, 4096)) {
        const auto load = [&](const char* name) {
            return dsv4::create_dsv4_pipeline(
                runtime_, resources_, directory / (std::string(name) + ".comp.spv"), 64);
        };
        const auto load32 = [&](const char* name) {
            return dsv4::create_dsv4_pipeline(
                runtime_, resources_, directory / (std::string(name) + ".comp.spv"), 32);
        };
        pipelines_.embedding = load("dsv4_embedding");
        pipelines_.rms = load("step37_rmsnorm");
        pipelines_.quant = load("dsv4_quantize_q8");
        pipelines_.q4 = load("dsv4_q4g64t_gemv");
        pipelines_.q4_small = load32("dsv4_q4g64t_gemv_wave32");
        pipelines_.q4_one = load("dsv4_q4g64t_gemv_one_lane");
        pipelines_.q4_residual = load("dsv4_q4g64t_gemv_residual");
        pipelines_.q8 = load("dsv4_q8_gemv");
        pipelines_.q4_batch4 = load("dsv4_q4g64t_gemv_batch4");
        pipelines_.q8_batch4 = load("dsv4_q8_gemv_batch4");
        pipelines_.swiglu = load("step37_swiglu");
        pipelines_.router = load("qwen38_router_top10");
#ifdef OVLLM_QWEN38_Q3_EXPERTS
        pipelines_.expert_gate = load("qwen38_expert_gate_up_q3");
        pipelines_.expert_down = load32("qwen38_expert_down_q3");
        pipelines_.expert_gate_batch =
            load("qwen38_expert_gate_up_q3_arena_batch");
        pipelines_.expert_down_batch =
            load32("qwen38_expert_down_q3_arena_batch");
        pipelines_.expert_gate_verify4 =
            load("qwen38_expert_gate_up_q3_verify4");
        pipelines_.expert_down_verify4 =
            load32("qwen38_expert_down_q3_verify4");
#else
        pipelines_.expert_gate = load("qwen38_expert_gate_up_q4");
        pipelines_.expert_down = load32("qwen38_expert_down_q4");
        pipelines_.expert_gate_batch =
            load("qwen38_expert_gate_up_q4_arena_batch");
        pipelines_.expert_down_batch =
            load32("qwen38_expert_down_q4_arena_batch");
#endif
        pipelines_.reduce = load("qwen38_reduce_shared_gate");
#ifdef OVLLM_LONG_CONTEXT_FORK
        pipelines_.qk = load("qwen35_long_qk_rope_cache");
        pipelines_.store_value = load("qwen35_long_store_value");
        pipelines_.attention = load("qwen38_long_attention_partial");
        pipelines_.attention_reduce = load("qwen35_long_attention_reduce");
#else
        pipelines_.qk = load("qwen35_qk_rope_cache");
        pipelines_.store_value = load("qwen35_store_value");
        pipelines_.attention = load("qwen35_attention");
#endif
        pipelines_.head_gate = load("qwen35_head_gate");
        pipelines_.conv = load("qwen35_conv_update");
        pipelines_.delta = load("qwen38_delta_recurrent_norm");
        pipelines_.argmax = load("qwen35_greedy_argmax");
        pipelines_.group_rms = load("qwen38_group_rmsnorm");
        pipelines_.hc_act = load("qwen38_hc_down_act");
        pipelines_.hc_mix = load("qwen38_hc_mix");
        pipelines_.hc_inject = load("qwen38_hc_inject");
        pipelines_.group_rms_batch4 = load("qwen38_group_rmsnorm_batch4");
        pipelines_.hc_mix_batch4 = load("qwen38_hc_mix_batch4");
        pipelines_.hc_inject_batch4 = load("qwen38_hc_inject_batch4");
        pipelines_.ple_gate = load("qwen38_ple_gate");
        pipelines_.ple_conv_add = load("qwen38_ple_conv_add");
        pipelines_.repeat_hc = load("qwen38_repeat_hc");
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
    if (const char* exact = std::getenv("QWEN38_CONTEXT_TOKENS")) {
        const uint64_t value = std::stoull(exact);
        if (value < 128 || value > UINT32_MAX)
            throw std::runtime_error("QWEN38_CONTEXT_TOKENS must be 128..2^32-1");
        return static_cast<uint32_t>(value);
    }
    if (const char* text = std::getenv("QWEN38_CONTEXT_GIB")) {
        const double gib = std::stod(text);
        if (gib < 0.05 || gib > 16.0)
            throw std::runtime_error("QWEN38_CONTEXT_GIB must be 0.05..16");
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
               const std::filesystem::path& ple_path,
               const std::filesystem::path& shader_directory,
               uint64_t ram_budget, uint32_t device_slots)
        : runtime_(runtime), weights_(runtime, index), expert_file_(expert_path),
          ple_lookup_(ple_path),
          host_cache_(runtime, expert_file_, ram_budget),
          device_cache_(runtime, device_slots),
          kernels_(runtime, shader_directory), compute_(runtime, runtime.queue),
          transfer_(runtime, runtime.secondary_queue),
          parallel_host_copy_(
              std::getenv("QWEN38_PARALLEL_HOST_COPY") != nullptr),
          copy_pool_(parallel_host_copy_) {
#ifdef OVLLM_LONG_CONTEXT_FORK
        max_context_ = requested_context_tokens();
        attention_chunks_ = (max_context_ + kAttentionChunk - 1u) /
            kAttentionChunk;
#endif
        batch_experts_ =
            std::getenv("QWEN38_EXPERT_BATCH_BDA") != nullptr;
        progressive_experts_ =
            std::getenv("QWEN38_PROGRESSIVE_EXPERTS") != nullptr;
        q4_wave32_mid_ =
            std::getenv("QWEN38_Q4_WAVE32_MID") != nullptr;
        q4_one_lane_ =
            std::getenv("QWEN38_Q4_ONE_LANE") != nullptr;
        q4_one_lane_mid_ =
            std::getenv("QWEN38_Q4_ONE_LANE_MID") != nullptr;
        verify4_enabled_ =
            std::getenv("QWEN38_VERIFY4_EXPERIMENT") != nullptr;
        verify4_batch_enabled_ = verify4_enabled_ &&
            std::getenv("QWEN38_VERIFY4_BATCH_Q4") != nullptr;
        verify_logits_host_enabled_ = verify4_batch_enabled_ &&
            std::getenv("QWEN38_RELAXED_REPEAT_GUARD") != nullptr;
        if (const char* text = std::getenv("QWEN38_VERIFY4_ACTIVE_TOPK")) {
            verify_active_topk_ = static_cast<uint32_t>(std::stoul(text));
            if (verify_active_topk_ < 1 || verify_active_topk_ > kTopK)
                throw std::runtime_error(
                    "QWEN38_VERIFY4_ACTIVE_TOPK must be between 1 and 10");
        }
        if (progressive_experts_)
            progressive_compute_ = std::make_unique<
                dsv4::experiment::FiniteQueueRing<12>>(
                    runtime_.device, runtime_.queue, runtime_.queue_family,
                    dsv4::finite_queue_ring_api());
        allocate_buffers();
        initialize_persistent_buffers();
#ifndef OVLLM_LONG_CONTEXT_FORK
        make_rope();
#endif
        build_sets();
        staging_.resize(kTopK);
        for (Buffer& buffer : staging_)
            buffer = dsv4::create_host_buffer_uninitialized(runtime_,
                                                             kExpertRecordBytes);
        if (verify4_enabled_) {
            verify_staging_.resize(4 * kTopK);
            for (Buffer& buffer : verify_staging_)
                buffer = dsv4::create_host_buffer_uninitialized(
                    runtime_, kExpertRecordBytes);
        }
    }

    ~QwenEngine() {
        for (Buffer& buffer : verify_staging_)
            destroy_buffer(runtime_, buffer);
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        destroy_all();
    }

    std::vector<uint32_t> generate(const Tokenizer& tokenizer,
                                   const std::vector<uint32_t>& prompt,
                                   uint32_t count) {
        if (prompt.empty()) throw std::runtime_error("Qwen prompt is empty");
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (uint64_t(prompt.size()) + count > max_context_)
            throw std::runtime_error("Qwen3.8 prompt plus generation exceeds configured context");
#endif
        uint32_t position = 0;
        uint32_t next = 0;
        for (uint32_t token : prompt) next = run_token(token, position++);
        if (std::getenv("QWEN38_FILL_RAM_CACHE")) {
            double fill_seconds = 0.0;
            const uint32_t filled =
                host_cache_.fill_remaining_uniform(fill_seconds);
            std::cout << "RAM cache top-off: " << filled << " records, "
                      << double(host_cache_.committed_bytes()) /
                             double(1ull << 30)
                      << " GiB cache, " << fill_seconds << " s\n";
        }
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (std::getenv("QWEN38_LONG_CONTEXT_STRESS")) {
            if (max_context_ <= count + 2u)
                throw std::runtime_error("Qwen3.8 context is too small for stress decode");
            const uint32_t target = max_context_ - count;
            expand_context_for_stress(prompt.size() - 1u, target);
            position = target;
            std::cout << "Qwen3.8 long-context stress positions: " << target
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
    const std::array<uint64_t, kLayers>& device_layer_misses() const {
        return device_cache_.layer_misses();
    }
    uint64_t ram_hits() const { return host_cache_.hits(); }
    uint64_t ram_misses() const { return host_cache_.misses(); }
    uint64_t ram_admission_bypasses() const {
        return host_cache_.admission_bypasses();
    }
    uint64_t disk_bytes() const { return host_cache_.disk_bytes(); }
    uint64_t ple_disk_bytes() const { return ple_lookup_.bytes_read(); }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
    uint64_t ple_transfer_bytes() const { return ple_transfer_bytes_; }
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
               + ple_host_.allocation_size + 8192
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
    double pre_submit_seconds() const { return pre_submit_seconds_; }
    double pre_wait_seconds() const { return pre_wait_seconds_; }
    double acquisition_seconds() const { return acquisition_seconds_; }
    double expert_seconds() const { return expert_seconds_; }

    // Isolated MTP accessors.  The ordinary generation path is unchanged;
    // these expose the final four-stream state and shared vocabulary weights
    // to the standalone speculative-decoding experiment.
    uint32_t process_experiment_token(uint32_t token, uint32_t position) {
        return run_token(token, position);
    }
    DescriptorRange hyper_experiment_range() const { return whole(hyper_); }
    TensorDevice embedding_experiment_tensor() const {
        return weights_.tensor("embed");
    }
    void reset_experiment_metrics() { reset_decode_metrics(); }
    uint32_t fill_experiment_ram_cache(double& seconds) {
        return host_cache_.fill_remaining_uniform(seconds);
    }
    uint32_t project_experiment_hidden(DescriptorRange source) {
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy copy{
                source.offset, 0, uint64_t(kDim) * sizeof(float)};
            vkfn::CmdCopyBuffer(command, source.buffer, hidden_.handle, 1, &copy);
            dsv4::transfer_barrier(command, hidden_);
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

    std::array<uint32_t, 4> verify4_experiment(
        const std::array<uint32_t, 4>& tokens, uint32_t position) {
        if (!verify4_enabled_)
            throw std::runtime_error("Qwen verify4 experiment is disabled");
        if (position + 3 >= kMaximumContext)
            throw std::runtime_error("Qwen verify4 context cap reached");
        if (verify4_batch_enabled_)
            return verify4_batch_experiment(tokens, position);
        const bool trace = std::getenv("QWEN38_VERIFY4_TRACE") != nullptr;
        if (trace) std::cerr << "verify4 begin position " << position << '\n';
        auto* token_words = static_cast<uint32_t*>(verify_tokens_.mapped);
        for (uint32_t row = 0; row < 4; ++row) token_words[row] = tokens[row];
        flush_buffer(runtime_, verify_tokens_);
        auto* ple_rows = static_cast<float*>(verify_ple_host_.mapped);
        for (uint32_t row = 0; row < 4; ++row) {
            ple_lookup_.lookup(tokens[row], ple_rows + uint64_t(row) * kDim);
            verify_ple_states_[row] = ple_lookup_.state();
        }
        dsv4::flush_buffer_range(runtime_, verify_ple_host_, 0,
                                 verify_ple_host_.size);

        const auto to_transfer = [](VkCommandBuffer command) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT;
            vkfn::CmdPipelineBarrier(command,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier,
                0, nullptr, 0, nullptr);
        };
        const auto to_compute = [](VkCommandBuffer command) {
            VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                    VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_SHADER_WRITE_BIT;
            vkfn::CmdPipelineBarrier(command,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier,
                0, nullptr, 0, nullptr);
        };

        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy ple_copy{0, 0, verify_ple_host_.size};
            vkfn::CmdCopyBuffer(command, verify_ple_host_.handle,
                                verify_ple_embedding_.handle, 1, &ple_copy);
            dsv4::transfer_barrier(command, verify_ple_embedding_);
            for (uint32_t row = 0; row < 4; ++row) {
                Push push{kVocabulary, kDim, kDim / 4, 0};
                kernels_.dispatch(command, kernels_.p().embedding,
                                  verify_embedding_sets_[row], &push,
                                  (kDim + 63) / 64);
                compute_barrier(command);
                push = {kDim, kHcCount, 0, 0};
                kernels_.dispatch(command, kernels_.p().repeat_hc,
                                  verify_repeat_sets_[row], &push,
                                  (kHcDim + 63) / 64);
                compute_barrier(command);
            }
        });
        compute_.wait(signal);
        ple_transfer_bytes_ += verify_ple_host_.size;
        if (trace) std::cerr << "verify4 embeddings ready\n";

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            if (trace) std::cerr << "verify4 layer " << layer << " pre\n";
            const auto pre_started = std::chrono::steady_clock::now();
            signal = compute_.submit([&](VkCommandBuffer command) {
                for (uint32_t row = 0; row < 4; ++row) {
                    to_transfer(command);
                    const VkBufferCopy hyper_copy{
                        0, 0, uint64_t(kHcDim) * sizeof(float)};
                    vkfn::CmdCopyBuffer(command, verify_hyper_[row].handle,
                                        hyper_.handle, 1, &hyper_copy);
                    if (layer == 1) {
                        const VkBufferCopy ple_copy{
                            uint64_t(row) * kDim * sizeof(float), 0,
                            uint64_t(kDim) * sizeof(float)};
                        vkfn::CmdCopyBuffer(command,
                            verify_ple_embedding_.handle,
                            ple_embedding_.handle, 1, &ple_copy);
                    }
                    to_compute(command);
                    if (layer == 1) record_ple(command);
                    record_hc_start(command, layers_[layer].attn_hc, true);
                    record_attention(command, layer, position + row);
                    record_hc_apply(command, layers_[layer].attn_hc);
                    record_hc_start(command, layers_[layer].mlp_hc, true);
                    record_router(command, layer);
                    to_transfer(command);
                    vkfn::CmdCopyBuffer(command, hyper_.handle,
                                        verify_hyper_[row].handle, 1,
                                        &hyper_copy);
                    const VkBufferCopy quant_copy{0, 0, quant_.size};
                    vkfn::CmdCopyBuffer(command, quant_.handle,
                                        verify_quant_[row].handle, 1,
                                        &quant_copy);
                    const VkBufferCopy route_copy{
                        0, uint64_t(row) * (16u + 2u * kTopK) *
                               sizeof(uint32_t),
                        (16u + kTopK) * sizeof(uint32_t)};
                    vkfn::CmdCopyBuffer(command, routing_.handle,
                                        verify_routing_.handle, 1,
                                        &route_copy);
                    if (row < 3 && !full_attention(layer)) {
                        const uint64_t linear = linear_index(layer);
                        const VkBufferCopy conv_copy{
                            linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                            linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                            uint64_t(kLinearQkv) * kConvWidth * 4};
                        vkfn::CmdCopyBuffer(command, conv_state_.handle,
                            verify_conv_snapshots_[row].handle, 1, &conv_copy);
                        const VkBufferCopy recurrent_copy{
                            linear * uint64_t(kLinearValueHeads) *
                                kLinearHeadDim * kLinearHeadDim * 4,
                            linear * uint64_t(kLinearValueHeads) *
                                kLinearHeadDim * kLinearHeadDim * 4,
                            uint64_t(kLinearValueHeads) * kLinearHeadDim *
                                kLinearHeadDim * 4};
                        vkfn::CmdCopyBuffer(command, recurrent_state_.handle,
                            verify_recurrent_snapshots_[row].handle, 1,
                            &recurrent_copy);
                    }
                    if (row < 3 && layer == 1) {
                        const VkBufferCopy ple_state_copy{0, 0, ple_state_.size};
                        vkfn::CmdCopyBuffer(command, ple_state_.handle,
                            verify_ple_snapshots_[row].handle, 1,
                            &ple_state_copy);
                    }
                    to_compute(command);
                }
            });
            compute_.wait(signal);
            if (trace) std::cerr << "verify4 layer " << layer << " pre ready\n";
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pre_started).count();

            invalidate_buffer(runtime_, verify_routing_);
            std::array<std::array<uint32_t, kTopK>, 4> experts{};
            const uint32_t* routes =
                static_cast<const uint32_t*>(verify_routing_.mapped);
            constexpr uint32_t route_stride = 16u + 2u * kTopK;
            for (uint32_t row = 0; row < 4; ++row) {
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    experts[row][rank] = routes[row * route_stride + rank];
                    if (experts[row][rank] >= kExperts)
                        throw std::runtime_error(
                            "Qwen verify4 router returned invalid expert");
                }
            }

            const auto acquire_started = std::chrono::steady_clock::now();
            const DeviceExpertCache::Batch4 selection =
                device_cache_.resolve_four(layer, experts);
            if (trace) std::cerr << "verify4 layer " << layer << " unique "
                                 << selection.unique_count << '\n';
            verify_unique_experts_ += selection.unique_count;
            verify_occurrences_ += 4 * kTopK;
            verify_reused_occurrences_ += selection.reused_occurrences;
            std::array<uint32_t, 4 * kTopK> missing_experts{};
            std::array<uint32_t, 4 * kTopK> missing_unique{};
            uint32_t missing_count = 0;
            for (uint32_t unique = 0; unique < selection.unique_count; ++unique) {
                if (!selection.unique_misses[unique]) continue;
                missing_experts[missing_count] =
                    selection.unique_experts[unique];
                missing_unique[missing_count++] = unique;
            }
            const HostExpertCache::Many sources = host_cache_.resolve_many(
                layer, missing_experts.data(), missing_count);
            if (sources.disk_reads) ++cold_stalled_layers_;
            std::vector<ParallelCopyPool::Task> copy_tasks;
            copy_tasks.reserve(missing_count);
            for (uint32_t index = 0; index < missing_count; ++index) {
                copy_tasks.push_back({verify_staging_[index].mapped,
                                      sources.pointers[index],
                                      size_t(kExpertRecordBytes)});
                host_copy_bytes_ += kExpertRecordBytes;
            }
            copy_pool_.copy(copy_tasks);
            if (trace) std::cerr << "verify4 layer " << layer << " copied "
                                 << missing_count << '\n';
            for (uint32_t index = 0; index < missing_count; ++index)
                dsv4::flush_buffer_range(runtime_, verify_staging_[index], 0,
                                         kExpertRecordBytes);
            uint64_t ready = 0;
            if (missing_count) {
                ready = transfer_.submit([&](VkCommandBuffer command) {
                    for (uint32_t index = 0; index < missing_count; ++index) {
                        const uint32_t unique = missing_unique[index];
                        const VkBufferCopy copy{
                            0, uint64_t(selection.unique_slots[unique]) *
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
                for (uint32_t row = 0; row < 4; ++row) {
                    to_transfer(command);
                    const VkBufferCopy hyper_copy{
                        0, 0, uint64_t(kHcDim) * sizeof(float)};
                    vkfn::CmdCopyBuffer(command, verify_hyper_[row].handle,
                                        hyper_.handle, 1, &hyper_copy);
                    const VkBufferCopy quant_copy{0, 0, quant_.size};
                    vkfn::CmdCopyBuffer(command, verify_quant_[row].handle,
                                        quant_.handle, 1, &quant_copy);
                    const VkBufferCopy route_copy{
                        uint64_t(row) * route_stride * sizeof(uint32_t), 0,
                        (16u + kTopK) * sizeof(uint32_t)};
                    vkfn::CmdCopyBuffer(command, verify_routing_.handle,
                                        routing_.handle, 1, &route_copy);
                    to_compute(command);
                    selected_slots_ = selection.slots[row];
                    record_shared(command, layer);
                    record_experts_individual(command, layer);
                    to_transfer(command);
                    vkfn::CmdCopyBuffer(command, hyper_.handle,
                                        verify_hyper_[row].handle, 1,
                                        &hyper_copy);
                    to_compute(command);
                }
            }, ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            compute_.wait(signal);
            if (trace) std::cerr << "verify4 layer " << layer
                                 << " expert ready\n";
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
        }

        signal = compute_.submit([&](VkCommandBuffer command) {
            for (uint32_t row = 0; row < 4; ++row) {
                to_transfer(command);
                const VkBufferCopy hyper_copy{
                    0, 0, uint64_t(kHcDim) * sizeof(float)};
                vkfn::CmdCopyBuffer(command, verify_hyper_[row].handle,
                                    hyper_.handle, 1, &hyper_copy);
                to_compute(command);
                record_hc_start(command, final_hc_, false);
                to_transfer(command);
                const VkBufferCopy hidden_copy{
                    0, 0, uint64_t(kDim) * sizeof(float)};
                vkfn::CmdCopyBuffer(command, hidden_.handle,
                                    verify_hidden_[row].handle, 1,
                                    &hidden_copy);
                to_compute(command);
            }
        });
        compute_.wait(signal);
        if (trace) std::cerr << "verify4 collapse ready\n";
        std::array<uint32_t, 4> result{};
        for (uint32_t row = 0; row < 4; ++row)
            result[row] = project_experiment_hidden(whole(verify_hidden_[row]));
        return result;
    }

    void accept_verify4_experiment(uint32_t consumed) {
        if (!verify4_enabled_ || consumed < 1 || consumed > 4)
            throw std::runtime_error("Invalid Qwen verify4 acceptance");
        const uint32_t row = consumed - 1;
        ple_lookup_.restore(verify_ple_states_[row]);
        if (consumed == 4) return;
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy hyper_copy{
                verify4_batch_enabled_
                    ? uint64_t(row) * kHcDim * sizeof(float) : 0,
                0, uint64_t(kHcDim) * sizeof(float)};
            vkfn::CmdCopyBuffer(command,
                verify4_batch_enabled_ ? verify_batch_hyper_.handle
                                       : verify_hyper_[row].handle,
                hyper_.handle, 1, &hyper_copy);
            const VkBufferCopy conv_copy{0, 0, conv_state_.size};
            vkfn::CmdCopyBuffer(command, verify_conv_snapshots_[row].handle,
                                conv_state_.handle, 1, &conv_copy);
            const VkBufferCopy recurrent_copy{0, 0, recurrent_state_.size};
            vkfn::CmdCopyBuffer(command,
                verify_recurrent_snapshots_[row].handle,
                recurrent_state_.handle, 1, &recurrent_copy);
            const VkBufferCopy ple_copy{0, 0, ple_state_.size};
            vkfn::CmdCopyBuffer(command, verify_ple_snapshots_[row].handle,
                                ple_state_.handle, 1, &ple_copy);
            dsv4::transfer_barrier(command, hyper_);
            dsv4::transfer_barrier(command, conv_state_);
            dsv4::transfer_barrier(command, recurrent_state_);
            dsv4::transfer_barrier(command, ple_state_);
        });
        compute_.wait(signal);
    }

    uint64_t verify_unique_experts() const { return verify_unique_experts_; }
    uint64_t verify_occurrences() const { return verify_occurrences_; }
    uint64_t verify_reused_occurrences() const {
        return verify_reused_occurrences_;
    }
    uint32_t safe_verify_token(
        uint32_t row, const std::vector<uint32_t>& history) const {
        if (!verify_logits_host_enabled_ || row >= kVerifyBatch)
            throw std::runtime_error("Qwen safe verifier logits unavailable");
        const float* logits =
            static_cast<const float*>(verify_logits_host_.mapped) +
            uint64_t(row) * kVocabulary;
        float best = -3.402823466e+38f;
        uint32_t best_token = 0;
        const size_t recent_begin = history.size() > 64
            ? history.size() - 64 : 0;
        for (uint32_t token = 0; token <= kEndOfText; ++token) {
            bool repeated = false;
            if (history.size() >= 3) {
                const size_t suffix = history.size() - 3;
                for (size_t prior = 0; prior + 3 < history.size(); ++prior) {
                    if (history[prior] == history[suffix] &&
                        history[prior + 1] == history[suffix + 1] &&
                        history[prior + 2] == history[suffix + 2] &&
                        history[prior + 3] == token) {
                        repeated = true;
                        break;
                    }
                }
            }
            float score = logits[token];
            if (token < kBaseVocabulary &&
                std::find(history.begin() + recent_begin, history.end(), token)
                    != history.end())
                score = score >= 0.0f ? score / 1.15f : score * 1.15f;
            if (!repeated &&
                (score > best ||
                 (score == best && token < best_token))) {
                best = score;
                best_token = token;
            }
        }
        if (logits[kImEnd] > best)
            best_token = kImEnd;
        return best_token;
    }
    uint32_t verify_active_topk() const { return verify_active_topk_; }
    void set_verify_active_topk(uint32_t active) {
        if (active < 1 || active > kTopK)
            throw std::runtime_error("Invalid Qwen verifier active top-k");
        verify_active_topk_ = active;
    }

private:
    static bool full_attention(uint32_t layer) { return layer % 4u == 3u; }
    static uint32_t full_index(uint32_t layer) { return layer / 4u; }
    static uint32_t linear_index(uint32_t layer) { return layer - layer / 4u; }

    VkPipeline q4_pipeline(uint32_t inner) const {
        // Wave32 is already the faster specialization for K=320/640.  The
        // opt-in also covers the many K=2560 projections, while retaining
        // Wave64 for K=6144/10240 where it amortizes the longer reduction.
        if ((q4_one_lane_ && inner > kMoeDim) ||
            (q4_one_lane_mid_ && inner > kMoeDim && inner <= kDim))
            return kernels_.p().q4_one;
        return (inner <= kMoeDim || (q4_wave32_mid_ && inner <= kDim))
            ? kernels_.p().q4_small : kernels_.p().q4;
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
        // Router output occupies the first 16 words.  The batched expert path
        // appends one 64-bit device address per route; Top-10 therefore needs
        // 36 words rather than the 32-word allocation inherited from Top-8.
        routing_ = create_buffer(
            runtime_, (16u + 2u * kTopK) * sizeof(uint32_t));
        hidden_ = device(uint64_t(kDim) * 4);
        normalized_ = device(uint64_t(kDim) * 4);
        hyper_ = device(uint64_t(kHcDim) * 4);
        hc_normed_ = device(uint64_t(kHcDim) * 4);
        hc_mix_weights_ = device(uint64_t(kHcDim) * 4);
        hc_low_ = device(uint64_t(kHcLowrank) * 4);
        hc_low_quant_ = device(128ull * 4);
        hc_injection_ = device(uint64_t(kHcCount) * 4);
        block_output_ = device(uint64_t(kDim) * 4);
        qgate_ = device(12288ull * 4);
        key_ = device(512ull * 4);
        value_ = device(512ull * 4);
        context_ = device(kLinearValue * 4ull);
        mixed_qkv_ = device(kLinearQkv * 4ull);
        convolved_qkv_ = device(kLinearQkv * 4ull);
        z_ = device(kLinearValue * 4ull);
        ab_ = device(96ull * 4);
        quant_ = device(4224ull * 4);
        shared_gate_values_ = device(kMoeDim * 4ull);
        shared_up_values_ = device(kMoeDim * 4ull);
        shared_intermediate_ = device(kMoeDim * 4ull);
        shared_output_ = device(kDim * 4ull);
        shared_expert_gate_ = device(4);
        router_logits_ = device(kExperts * 4ull);
        expert_intermediate_ = device(uint64_t(kTopK) * kMoeDim * 4);
        expert_quant_ = device(1664ull * 4);
        expert_outputs_ = device(uint64_t(kTopK) * kDim * 4);
        logits_ = device(uint64_t(kVocabulary) * 4);
        argmax_workspace_ = device(512ull * 4);
        conv_state_ = device(uint64_t(kLinearLayers) * kLinearQkv *
                             kConvWidth * 4);
        recurrent_state_ = device(uint64_t(kLinearLayers) *
                                  kLinearValueHeads * kLinearHeadDim *
                                  kLinearHeadDim * 4);
        ple_host_ = dsv4::create_host_buffer_uninitialized(runtime_, kDim * 4ull);
        ple_embedding_ = device(kDim * 4ull);
        ple_key_ = device(kHcDim * 4ull);
        ple_key_normed_ = device(kHcDim * 4ull);
        ple_value_ = device(kDim * 4ull);
        ple_gated_ = device(kHcDim * 4ull);
        ple_gated_normed_ = device(kHcDim * 4ull);
        ple_state_ = device(uint64_t(kHcDim) * kPleHistory * 4ull);
#ifdef OVLLM_LONG_CONTEXT_FORK
        const uint64_t kv_bytes = uint64_t(kFullLayers) * 2u * max_context_ *
            kKvHeads * kHeadDim * sizeof(uint16_t);
        kv_cache_ = dsv4::create_host_buffer_uninitialized(runtime_, kv_bytes);
        attention_partial_ = device(uint64_t(attention_chunks_) *
            kAttentionHeads * kAttentionPartialStride * sizeof(float));
#else
        kv_cache_ = device(uint64_t(kFullLayers) * 2 * kMaximumContext *
                           kKvHeads * kHeadDim * 4);
        rope_ = device(uint64_t(kMaximumContext) * kRopeDim * 4);
#endif
        if (verify4_enabled_) {
            verify_tokens_ = create_buffer(runtime_, 4u * sizeof(uint32_t));
            verify_routing_ = create_buffer(
                runtime_, 4u * (16u + 2u * kTopK) * sizeof(uint32_t));
            verify_ple_host_ = dsv4::create_host_buffer_uninitialized(
                runtime_, 4ull * kDim * sizeof(float));
            verify_ple_embedding_ = device(4ull * kDim * sizeof(float));
            for (uint32_t row = 0; row < 4; ++row) {
                verify_hidden_[row] = device(uint64_t(kDim) * sizeof(float));
                verify_hyper_[row] = device(uint64_t(kHcDim) * sizeof(float));
                verify_quant_[row] = device(quant_.size);
            }
            // Three snapshots cover the three rejection boundaries.  When
            // all four inputs are accepted the live state already is row 3.
            for (uint32_t row = 0; row < 3; ++row) {
                verify_conv_snapshots_[row] = device(conv_state_.size);
                verify_recurrent_snapshots_[row] = device(recurrent_state_.size);
                verify_ple_snapshots_[row] = device(ple_state_.size);
            }
            if (verify4_batch_enabled_) {
                verify_batch_hyper_ = device(
                    uint64_t(kVerifyBatch) * kHcDim * sizeof(float));
                verify_batch_hc_normed_ = device(
                    uint64_t(kVerifyBatch) * kHcDim * sizeof(float));
                verify_batch_hc_mix_ = device(
                    uint64_t(kVerifyBatch) * kHcDim * sizeof(float));
                verify_batch_hc_low_ = device(
                    uint64_t(kVerifyBatch) * kHcLowrank * sizeof(float));
                verify_batch_hc_injection_ = device(
                    uint64_t(kVerifyBatch) * kHcCount * sizeof(float));
                verify_batch_hidden_ = device(
                    uint64_t(kVerifyBatch) * kDim * sizeof(float));
                verify_batch_block_ = device(
                    uint64_t(kVerifyBatch) * kDim * sizeof(float));
                verify_batch_quant_ = device(
                    uint64_t(kVerifyBatch) * kVerifyHcQuantU32 * sizeof(uint32_t));
                verify_batch_low_quant_ = device(
                    uint64_t(kVerifyBatch) * kVerifyLowQuantU32 * sizeof(uint32_t));
                verify_batch_moe_quant_ = device(
                    uint64_t(kVerifyBatch) * kVerifyMoeQuantU32 * sizeof(uint32_t));
                verify_batch_qgate_ = device(
                    uint64_t(kVerifyBatch) * 12288 * sizeof(float));
                verify_batch_key_ = device(
                    uint64_t(kVerifyBatch) * 512 * sizeof(float));
                verify_batch_value_ = device(
                    uint64_t(kVerifyBatch) * 512 * sizeof(float));
                verify_batch_context_ = device(
                    uint64_t(kVerifyBatch) * kLinearValue * sizeof(float));
                verify_batch_mixed_qkv_ = device(
                    uint64_t(kVerifyBatch) * kLinearQkv * sizeof(float));
                verify_batch_convolved_qkv_ = device(
                    uint64_t(kVerifyBatch) * kLinearQkv * sizeof(float));
                verify_batch_z_ = device(
                    uint64_t(kVerifyBatch) * kLinearValue * sizeof(float));
                verify_batch_ab_ = device(
                    uint64_t(kVerifyBatch) * 96 * sizeof(float));
                verify_batch_router_logits_ = device(
                    uint64_t(kVerifyBatch) * kExperts * sizeof(float));
                verify_batch_shared_gate_ = device(
                    uint64_t(kVerifyBatch) * kMoeDim * sizeof(float));
                verify_batch_shared_up_ = device(
                    uint64_t(kVerifyBatch) * kMoeDim * sizeof(float));
                verify_batch_shared_intermediate_ = device(
                    uint64_t(kVerifyBatch) * kMoeDim * sizeof(float));
                verify_batch_shared_output_ = device(
                    uint64_t(kVerifyBatch) * kDim * sizeof(float));
                verify_batch_shared_expert_gate_ = device(
                    uint64_t(kVerifyBatch) * sizeof(float));
                verify_batch_logits_ = device(
                    uint64_t(kVerifyBatch) * kVocabulary * sizeof(float));
                if (verify_logits_host_enabled_)
                    verify_logits_host_ =
                        dsv4::create_host_buffer_uninitialized(
                            runtime_, verify_batch_logits_.size);
                verify_batch_argmax_ = device(
                    uint64_t(kVerifyBatch) * 512 * sizeof(float));
                verify_expert_meta_ = create_buffer(
                    runtime_, uint64_t(4 * kTopK) * 8 * sizeof(uint32_t));
                verify_batch_expert_intermediate_ = device(
                    uint64_t(kVerifyBatch) * kTopK * kMoeDim * sizeof(float));
                verify_batch_expert_quant_ = device(
                    uint64_t(kVerifyBatch) * kTopK *
                    (kMoeDim / 4 + kMoeDim / 128) * sizeof(uint32_t));
                verify_batch_expert_outputs_ = device(
                    uint64_t(kVerifyBatch) * kTopK * kDim * sizeof(float));
            }
        }
    }

    void initialize_persistent_buffers() {
        constexpr uint64_t chunk_bytes = 64ull * 1024 * 1024;
        Buffer zeros = dsv4::create_host_buffer_uninitialized(runtime_, chunk_bytes);
        std::memset(zeros.mapped, 0, static_cast<size_t>(chunk_bytes));
        dsv4::flush_buffer_range(runtime_, zeros, 0, chunk_bytes);
        for (Buffer* buffer : {&conv_state_, &recurrent_state_, &ple_state_
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

    void make_rope() {
#ifndef OVLLM_LONG_CONTEXT_FORK
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
#endif
    }

    struct HcSets {
        VkDescriptorSet norm{}, quant{}, down{}, inject{}, act{};
        VkDescriptorSet low_quant{}, up{}, mix{}, apply{};
    };

    struct VerifyHcSets {
        VkDescriptorSet norm{}, down{}, inject{}, act{}, up{}, mix{}, apply{};
        std::array<VkDescriptorSet, kVerifyBatch> quant{}, low_quant{};
    };

    struct VerifyLayerSets {
        VerifyHcSets attn_hc{}, mlp_hc{};
        std::array<VkDescriptorSet, kVerifyBatch> hidden_quant{};
        VkDescriptorSet qgate{}, key{}, value{}, attention_out{};
        std::array<VkDescriptorSet, kVerifyBatch> qk{}, store_value{};
        std::array<VkDescriptorSet, kVerifyBatch> attention{}, head_gate{};
        std::array<VkDescriptorSet, kVerifyBatch> context_quant{};
        VkDescriptorSet gdn_qkv{}, gdn_z{}, ab{}, gdn_out{};
        std::array<VkDescriptorSet, kVerifyBatch> conv{}, delta{};
        VkDescriptorSet router_gemv{};
        std::array<VkDescriptorSet, kVerifyBatch> router{};
        VkDescriptorSet shared_gate{}, shared_up{}, shared_swiglu{};
        std::array<VkDescriptorSet, kVerifyBatch> shared_quant{};
        VkDescriptorSet shared_down{}, shared_expert_gate{};
        std::array<VkDescriptorSet, kVerifyBatch> reduce{};
    };

    struct PleSets {
        VkDescriptorSet quant{}, key{}, value{}, key_norm{}, query_norm{};
        VkDescriptorSet gate{}, gated_norm{}, conv_add{};
    };

    struct LayerSets {
        HcSets attn_hc{}, mlp_hc{};
        VkDescriptorSet hidden_quant{};
        VkDescriptorSet qgate{}, key{}, value{}, qk{}, store_value{};
        VkDescriptorSet attention{}, head_gate{}, context_quant{}, attention_out{};
#ifdef OVLLM_LONG_CONTEXT_FORK
        VkDescriptorSet attention_reduce{};
#endif
        VkDescriptorSet gdn_qkv{}, gdn_z{}, ab{}, conv{}, delta{}, gdn_out{};
        VkDescriptorSet router_gemv{}, router{};
        VkDescriptorSet shared_gate{}, shared_up{}, shared_swiglu{};
        VkDescriptorSet shared_quant{}, shared_down{}, shared_expert_gate{};
        VkDescriptorSet expert_gate_batch{}, expert_down_batch{};
        std::vector<VkDescriptorSet> expert_gate, expert_down;
        std::vector<VkDescriptorSet> expert_gate_verify4, expert_down_verify4;
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

    VkDescriptorSet q4_batch4_set(DescriptorRange activation,
                                  const TensorDevice& weight,
                                  DescriptorRange output) {
        return kernels_.set(
            {activation, weight.data, weight.auxiliary, output});
    }

    VkDescriptorSet q8_batch4_set(DescriptorRange activation,
                                  const TensorDevice& weight,
                                  DescriptorRange output) {
        return kernels_.set(
            {activation, weight.data, weight.auxiliary, output});
    }

    VerifyHcSets build_verify_hc_sets(const std::string& prefix,
                                      bool with_injection) {
        VerifyHcSets sets{};
        const TensorDevice norm = tensor(prefix + "norm", TensorFormat::f32,
                                          kHcDim);
        const TensorDevice down = tensor(prefix + "down", TensorFormat::q4g64t,
                                          kHcLowrank, kHcDim);
        const TensorDevice up = tensor(prefix + "up", TensorFormat::q4g64t,
                                        kHcDim, kHcLowrank);
        sets.norm = kernels_.set({whole(verify_batch_hyper_), norm.data,
                                  whole(verify_batch_hc_normed_)});
        for (uint32_t row = 0; row < kVerifyBatch; ++row) {
            sets.quant[row] = kernels_.set({
                arena_range(verify_batch_hc_normed_,
                    uint64_t(row) * kHcDim * sizeof(float),
                    uint64_t(kHcDim) * sizeof(float)),
                arena_range(verify_batch_quant_,
                    uint64_t(row) * kVerifyHcQuantU32 * sizeof(uint32_t),
                    uint64_t(kVerifyHcQuantU32) * sizeof(uint32_t))});
            sets.low_quant[row] = kernels_.set({
                arena_range(verify_batch_hc_low_,
                    uint64_t(row) * kHcLowrank * sizeof(float),
                    uint64_t(kHcLowrank) * sizeof(float)),
                arena_range(verify_batch_low_quant_,
                    uint64_t(row) * kVerifyLowQuantU32 * sizeof(uint32_t),
                    uint64_t(kVerifyLowQuantU32) * sizeof(uint32_t))});
        }
        sets.down = q4_batch4_set(whole(verify_batch_quant_), down,
                                   whole(verify_batch_hc_low_));
        sets.act = kernels_.set({whole(verify_batch_hc_low_)});
        sets.up = q4_batch4_set(whole(verify_batch_low_quant_), up,
                                 whole(verify_batch_hc_mix_));
        sets.mix = kernels_.set({whole(verify_batch_hc_normed_),
                                  whole(verify_batch_hc_mix_),
                                  whole(verify_batch_hidden_)});
        if (with_injection) {
            const TensorDevice injection = tensor(prefix + "inject",
                TensorFormat::q4g64t, kHcCount, kHcDim);
            sets.inject = q4_batch4_set(whole(verify_batch_quant_), injection,
                                        whole(verify_batch_hc_injection_));
            sets.apply = kernels_.set({whole(verify_batch_block_),
                                       whole(verify_batch_hc_injection_),
                                       whole(verify_batch_hyper_)});
        }
        return sets;
    }

    void build_verify_batch_sets(const TensorDevice& embedding,
                                 const TensorDevice& lm_head) {
        for (uint32_t row = 0; row < kVerifyBatch; ++row) {
            const DescriptorRange hidden = arena_range(
                verify_batch_hidden_, uint64_t(row) * kDim * sizeof(float),
                uint64_t(kDim) * sizeof(float));
            const DescriptorRange hyper = arena_range(
                verify_batch_hyper_, uint64_t(row) * kHcDim * sizeof(float),
                uint64_t(kHcDim) * sizeof(float));
            verify_batch_embedding_sets_[row] = kernels_.set({
                embedding.data, embedding.auxiliary,
                arena_range(verify_tokens_, uint64_t(row) * sizeof(uint32_t),
                            sizeof(uint32_t)), hidden});
            verify_batch_repeat_sets_[row] = kernels_.set({hidden, hyper});
            verify_batch_final_quant_sets_[row] = kernels_.set({
                hidden, arena_range(verify_batch_quant_,
                    uint64_t(row) * kVerifyDimQuantU32 * sizeof(uint32_t),
                    uint64_t(kVerifyDimQuantU32) * sizeof(uint32_t))});
            verify_batch_argmax_sets_[row] = kernels_.set({
                arena_range(verify_batch_logits_,
                    uint64_t(row) * kVocabulary * sizeof(float),
                    uint64_t(kVocabulary) * sizeof(float)),
                arena_range(verify_tokens_, uint64_t(row) * sizeof(uint32_t),
                            sizeof(uint32_t)),
                arena_range(verify_batch_argmax_,
                    uint64_t(row) * 512 * sizeof(float),
                    uint64_t(512) * sizeof(float))});
        }
        verify_batch_final_hc_ = build_verify_hc_sets("final_hc_", false);
        verify_batch_lm_head_set_ = q8_batch4_set(
            whole(verify_batch_quant_), lm_head, whole(verify_batch_logits_));

        verify_batch_layers_.resize(kLayers);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            VerifyLayerSets& sets = verify_batch_layers_[layer];
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            sets.attn_hc = build_verify_hc_sets(prefix + "attn_hc_", true);
            sets.mlp_hc = build_verify_hc_sets(prefix + "mlp_hc_", true);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                sets.hidden_quant[row] = kernels_.set({
                    arena_range(verify_batch_hidden_,
                        uint64_t(row) * kDim * sizeof(float),
                        uint64_t(kDim) * sizeof(float)),
                    arena_range(verify_batch_quant_,
                        uint64_t(row) * kVerifyDimQuantU32 * sizeof(uint32_t),
                        uint64_t(kVerifyDimQuantU32) * sizeof(uint32_t))});
            }

            if (full_attention(layer)) {
                const TensorDevice query = tensor(prefix + "q_proj",
                    TensorFormat::q4g64t, 12288, kDim);
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
                sets.qgate = q4_batch4_set(whole(verify_batch_quant_), query,
                                            whole(verify_batch_qgate_));
                sets.key = q4_batch4_set(whole(verify_batch_quant_), key,
                                          whole(verify_batch_key_));
                sets.value = q4_batch4_set(whole(verify_batch_quant_), value,
                                            whole(verify_batch_value_));
                for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                    const DescriptorRange qgate = arena_range(
                        verify_batch_qgate_, uint64_t(row) * 12288 * 4,
                        uint64_t(12288) * 4);
                    const DescriptorRange key_row = arena_range(
                        verify_batch_key_, uint64_t(row) * 512 * 4,
                        uint64_t(512) * 4);
                    const DescriptorRange value_row = arena_range(
                        verify_batch_value_, uint64_t(row) * 512 * 4,
                        uint64_t(512) * 4);
                    const DescriptorRange context = arena_range(
                        verify_batch_context_, uint64_t(row) * kLinearValue * 4,
                        uint64_t(kLinearValue) * 4);
                    sets.qk[row] = kernels_.set({qgate, key_row, qnorm.data,
                        knorm.data, whole(kv_cache_), whole(rope_)});
                    sets.store_value[row] = kernels_.set(
                        {value_row, whole(kv_cache_)});
                    sets.attention[row] = kernels_.set(
                        {qgate, whole(kv_cache_), context});
                    sets.head_gate[row] = kernels_.set({context, qgate});
                    sets.context_quant[row] = kernels_.set({
                        context, arena_range(verify_batch_quant_,
                            uint64_t(row) * kVerifyContextQuantU32 * 4,
                            uint64_t(kVerifyContextQuantU32) * 4)});
                }
                sets.attention_out = q4_batch4_set(
                    whole(verify_batch_quant_), output,
                    whole(verify_batch_block_));
            } else {
                const TensorDevice qkv = tensor(prefix + "gdn_qkv",
                    TensorFormat::q4g64t, kLinearQkv, kDim);
                const TensorDevice z = tensor(prefix + "gdn_z",
                    TensorFormat::q4g64t, kLinearValue, kDim);
                const TensorDevice ab = tensor(prefix + "ab_proj",
                    TensorFormat::q4g64t, 96, kDim);
                const TensorDevice output = tensor(prefix + "gdn_out",
                    TensorFormat::q4g64t, kDim, kLinearValue);
                const TensorDevice conv = tensor(prefix + "conv",
                    TensorFormat::f32, kLinearQkv, kConvWidth);
                const TensorDevice params = tensor(prefix + "delta_params",
                    TensorFormat::f32, 224);
                sets.gdn_qkv = q4_batch4_set(whole(verify_batch_quant_), qkv,
                                              whole(verify_batch_mixed_qkv_));
                sets.gdn_z = q4_batch4_set(whole(verify_batch_quant_), z,
                                            whole(verify_batch_z_));
                sets.ab = q4_batch4_set(whole(verify_batch_quant_), ab,
                                         whole(verify_batch_ab_));
                const uint64_t linear = linear_index(layer);
                const DescriptorRange conv_state = arena_range(
                    conv_state_, linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                    uint64_t(kLinearQkv) * kConvWidth * 4);
                const DescriptorRange recurrent_state = arena_range(
                    recurrent_state_, linear * uint64_t(kLinearValueHeads) *
                        kLinearHeadDim * kLinearHeadDim * 4,
                    uint64_t(kLinearValueHeads) * kLinearHeadDim *
                        kLinearHeadDim * 4);
                for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                    const DescriptorRange qkv_row = arena_range(
                        verify_batch_mixed_qkv_, uint64_t(row) * kLinearQkv * 4,
                        uint64_t(kLinearQkv) * 4);
                    const DescriptorRange conv_row = arena_range(
                        verify_batch_convolved_qkv_,
                        uint64_t(row) * kLinearQkv * 4,
                        uint64_t(kLinearQkv) * 4);
                    const DescriptorRange z_row = arena_range(
                        verify_batch_z_, uint64_t(row) * kLinearValue * 4,
                        uint64_t(kLinearValue) * 4);
                    const DescriptorRange ab_row = arena_range(
                        verify_batch_ab_, uint64_t(row) * 96 * 4,
                        uint64_t(96) * 4);
                    const DescriptorRange context = arena_range(
                        verify_batch_context_, uint64_t(row) * kLinearValue * 4,
                        uint64_t(kLinearValue) * 4);
                    sets.conv[row] = kernels_.set(
                        {qkv_row, conv.data, conv_state, conv_row});
                    sets.delta[row] = kernels_.set(
                        {conv_row, z_row, ab_row, recurrent_state,
                         params.data, context});
                    sets.context_quant[row] = kernels_.set({
                        context, arena_range(verify_batch_quant_,
                            uint64_t(row) * kVerifyContextQuantU32 * 4,
                            uint64_t(kVerifyContextQuantU32) * 4)});
                }
                sets.gdn_out = q4_batch4_set(whole(verify_batch_quant_), output,
                                              whole(verify_batch_block_));
            }

            const TensorDevice router = tensor(prefix + "router",
                TensorFormat::q8_row, kExperts, kDim);
            sets.router_gemv = q8_batch4_set(whole(verify_batch_quant_), router,
                whole(verify_batch_router_logits_));
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                sets.router[row] = kernels_.set({
                    arena_range(verify_batch_router_logits_,
                        uint64_t(row) * kExperts * 4,
                        uint64_t(kExperts) * 4),
                    arena_range(verify_routing_,
                        uint64_t(row) * (16u + 2u * kTopK) * sizeof(uint32_t),
                        (16u + 2u * kTopK) * sizeof(uint32_t))});
            }
            sets.shared_gate = q4_batch4_set(whole(verify_batch_quant_), tensor(
                prefix + "shared_gate_proj", TensorFormat::q4g64t,
                kMoeDim, kDim), whole(verify_batch_shared_gate_));
            sets.shared_up = q4_batch4_set(whole(verify_batch_quant_), tensor(
                prefix + "shared_up_proj", TensorFormat::q4g64t,
                kMoeDim, kDim), whole(verify_batch_shared_up_));
            sets.shared_swiglu = kernels_.set({whole(verify_batch_shared_gate_),
                whole(verify_batch_shared_up_), kernels_.dummy(),
                whole(verify_batch_shared_intermediate_)});
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                sets.shared_quant[row] = kernels_.set({
                    arena_range(verify_batch_shared_intermediate_,
                        uint64_t(row) * kMoeDim * 4,
                        uint64_t(kMoeDim) * 4),
                    arena_range(verify_batch_moe_quant_,
                        uint64_t(row) * kVerifyMoeQuantU32 * 4,
                        uint64_t(kVerifyMoeQuantU32) * 4)});
                sets.reduce[row] = kernels_.set({
                    arena_range(verify_batch_expert_outputs_,
                        uint64_t(row) * kTopK * kDim * 4,
                        uint64_t(kTopK) * kDim * 4),
                    arena_range(verify_batch_shared_output_,
                        uint64_t(row) * kDim * 4, uint64_t(kDim) * 4),
                    arena_range(verify_batch_shared_expert_gate_,
                        uint64_t(row) * sizeof(float), sizeof(float)),
                    arena_range(verify_batch_block_,
                        uint64_t(row) * kDim * 4, uint64_t(kDim) * 4)});
            }
            sets.shared_down = q4_batch4_set(
                whole(verify_batch_moe_quant_), tensor(
                    prefix + "shared_down_proj", TensorFormat::q4g64t,
                    kDim, kMoeDim), whole(verify_batch_shared_output_));
            sets.shared_expert_gate = q4_batch4_set(
                whole(verify_batch_quant_), tensor(
                    prefix + "shared_expert_gate", TensorFormat::q4g64t,
                    1, kDim), whole(verify_batch_shared_expert_gate_));
        }
    }

    HcSets build_hc_sets(const std::string& prefix, bool with_injection) {
        HcSets sets{};
        const TensorDevice norm = tensor(prefix + "norm", TensorFormat::f32, kHcDim);
        const TensorDevice down = tensor(prefix + "down", TensorFormat::q4g64t,
                                         kHcLowrank, kHcDim);
        const TensorDevice up = tensor(prefix + "up", TensorFormat::q4g64t,
                                       kHcDim, kHcLowrank);
        sets.norm = kernels_.set({whole(hyper_), norm.data, whole(hc_normed_)});
        sets.quant = kernels_.set({whole(hc_normed_), whole(quant_)});
        sets.down = q4_set(whole(quant_), down, whole(hc_low_));
        sets.act = kernels_.set({whole(hc_low_)});
        sets.low_quant = kernels_.set({whole(hc_low_), whole(hc_low_quant_)});
        sets.up = q4_set(whole(hc_low_quant_), up, whole(hc_mix_weights_));
        sets.mix = kernels_.set({whole(hc_normed_), whole(hc_mix_weights_),
                                 whole(hidden_)});
        if (with_injection) {
            const TensorDevice injection = tensor(prefix + "inject",
                TensorFormat::q4g64t, kHcCount, kHcDim);
            sets.inject = q4_set(whole(quant_), injection, whole(hc_injection_));
            sets.apply = kernels_.set({whole(block_output_), whole(hc_injection_),
                                       whole(hyper_)});
        }
        return sets;
    }

    void build_sets() {
        const TensorDevice embedding =
            tensor("embed", TensorFormat::q8_row, kVocabulary, kDim);
        const TensorDevice lm_head =
            tensor("lm_head", TensorFormat::q8_row, kVocabulary, kDim);
        embedding_set_ = kernels_.set(
            {embedding.data, embedding.auxiliary, whole(token_), whole(hidden_)});
        repeat_set_ = kernels_.set({whole(hidden_), whole(hyper_)});
        if (verify4_enabled_) {
            for (uint32_t row = 0; row < 4; ++row) {
                verify_embedding_sets_[row] = kernels_.set({
                    embedding.data, embedding.auxiliary,
                    arena_range(verify_tokens_, uint64_t(row) * sizeof(uint32_t),
                                sizeof(uint32_t)),
                    whole(verify_hidden_[row])});
                verify_repeat_sets_[row] = kernels_.set(
                    {whole(verify_hidden_[row]), whole(verify_hyper_[row])});
            }
        }
        final_hc_ = build_hc_sets("final_hc_", false);
        final_quant_set_ = kernels_.set({whole(hidden_), whole(quant_)});
        lm_head_set_ = kernels_.set(
            {whole(quant_), lm_head.data, lm_head.auxiliary, whole(logits_)});
        argmax_set_ = kernels_.set(
            {whole(logits_), whole(token_), whole(argmax_workspace_)});
        expert_quant_set_ = kernels_.set(
            {whole(expert_intermediate_), whole(expert_quant_)});
        if (verify4_batch_enabled_)
            verify_batch_expert_quant_set_ = kernels_.set({
                whole(verify_batch_expert_intermediate_),
                whole(verify_batch_expert_quant_)});
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
                                    whole(shared_expert_gate_), whole(block_output_)});
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
            sets.attn_hc = build_hc_sets(prefix + "attn_hc_", true);
            sets.mlp_hc = build_hc_sets(prefix + "mlp_hc_", true);
            sets.hidden_quant = kernels_.set({whole(hidden_), whole(quant_)});

            if (full_attention(layer)) {
                const TensorDevice query = tensor(prefix + "q_proj",
                    TensorFormat::q4g64t, 12288, kDim);
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
#ifdef OVLLM_LONG_CONTEXT_FORK
                const uint64_t layer_cache_bytes = uint64_t(max_context_) * 2u *
                    kKvHeads * kHeadDim * sizeof(uint16_t);
                const DescriptorRange layer_cache = arena_range(kv_cache_,
                    uint64_t(full_index(layer)) * layer_cache_bytes,
                    layer_cache_bytes);
                sets.qk = kernels_.set({whole(qgate_), whole(key_), qnorm.data,
                                        knorm.data, layer_cache});
                sets.store_value = kernels_.set({whole(value_), layer_cache});
                sets.attention = kernels_.set(
                    {whole(qgate_), layer_cache, whole(attention_partial_)});
                sets.attention_reduce = kernels_.set(
                    {whole(attention_partial_), whole(context_)});
#else
                sets.qk = kernels_.set({whole(qgate_), whole(key_), qnorm.data,
                                        knorm.data, whole(kv_cache_), whole(rope_)});
                sets.store_value = kernels_.set({whole(value_), whole(kv_cache_)});
                sets.attention = kernels_.set(
                    {whole(qgate_), whole(kv_cache_), whole(context_)});
#endif
                sets.head_gate = kernels_.set({whole(context_), whole(qgate_)});
                sets.context_quant = kernels_.set({whole(context_), whole(quant_)});
                sets.attention_out = q4_set(whole(quant_), output,
                                            whole(block_output_));
            } else {
                const TensorDevice qkv = tensor(prefix + "gdn_qkv",
                    TensorFormat::q4g64t, kLinearQkv, kDim);
                const TensorDevice z = tensor(prefix + "gdn_z",
                    TensorFormat::q4g64t, kLinearValue, kDim);
                const TensorDevice ab = tensor(prefix + "ab_proj",
                    TensorFormat::q4g64t, 96, kDim);
                const TensorDevice output = tensor(prefix + "gdn_out",
                    TensorFormat::q4g64t, kDim, kLinearValue);
                const TensorDevice conv = tensor(prefix + "conv",
                    TensorFormat::f32, kLinearQkv, kConvWidth);
                const TensorDevice params = tensor(prefix + "delta_params",
                    TensorFormat::f32, 224);
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
                sets.gdn_out = q4_set(whole(quant_), output,
                                      whole(block_output_));
            }

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

            sets.expert_gate_batch = kernels_.set(
                {whole(quant_), whole(device_cache_.arena(layer)),
                 whole(routing_), whole(expert_intermediate_)});
            sets.expert_down_batch = kernels_.set(
                {whole(expert_quant_), whole(device_cache_.arena(layer)),
                 whole(routing_), whole(expert_outputs_)});

            const uint32_t layer_slot_count = device_cache_.layer_slots(layer);
            sets.expert_gate.resize(layer_slot_count);
            sets.expert_down.resize(layer_slot_count);
            if (verify4_batch_enabled_) {
                sets.expert_gate_verify4.resize(layer_slot_count);
                sets.expert_down_verify4.resize(layer_slot_count);
            }
            for (uint32_t slot = 0; slot < layer_slot_count; ++slot) {
                const DescriptorRange record = device_cache_.record(layer, slot);
                sets.expert_gate[slot] = kernels_.set(
                    {whole(quant_), record, whole(routing_),
                     whole(expert_intermediate_)});
                sets.expert_down[slot] = kernels_.set(
                    {whole(expert_quant_), record, whole(routing_),
                     whole(expert_outputs_)});
                if (verify4_batch_enabled_) {
                    sets.expert_gate_verify4[slot] = kernels_.set({
                        whole(verify_batch_quant_), record,
                        whole(verify_expert_meta_),
                        whole(verify_batch_expert_intermediate_)});
                    sets.expert_down_verify4[slot] = kernels_.set({
                        whole(verify_batch_expert_quant_), record,
                        whole(verify_expert_meta_),
                        whole(verify_batch_expert_outputs_)});
                }
            }
        }

        const std::string ple = "layers.1.";
        const TensorDevice ple_key = tensor(ple + "ple_key", TensorFormat::q4g64t,
                                            kHcDim, kDim);
        const TensorDevice ple_value = tensor(ple + "ple_value", TensorFormat::q4g64t,
                                              kDim, kDim);
        const TensorDevice ple_norm_key = tensor(ple + "ple_norm_key", TensorFormat::f32,
                                                 kHcDim);
        const TensorDevice ple_norm_query = tensor(ple + "ple_norm_query", TensorFormat::f32,
                                                   kHcDim);
        const TensorDevice ple_norm_conv = tensor(ple + "ple_norm_conv", TensorFormat::f32,
                                                  kHcDim);
        const TensorDevice ple_conv = tensor(ple + "ple_conv", TensorFormat::f32,
                                             kHcDim, 4);
        ple_sets_.quant = kernels_.set({whole(ple_embedding_), whole(quant_)});
        ple_sets_.key = q4_set(whole(quant_), ple_key, whole(ple_key_));
        ple_sets_.value = q4_set(whole(quant_), ple_value, whole(ple_value_));
        ple_sets_.key_norm = kernels_.set({whole(ple_key_), ple_norm_key.data,
                                           whole(ple_key_normed_)});
        ple_sets_.query_norm = kernels_.set({whole(hyper_), ple_norm_query.data,
                                             whole(hc_normed_)});
        ple_sets_.gate = kernels_.set({whole(ple_key_normed_), whole(hc_normed_),
                                       whole(ple_value_), whole(ple_gated_)});
        ple_sets_.gated_norm = kernels_.set({whole(ple_gated_), ple_norm_conv.data,
                                             whole(ple_gated_normed_)});
        ple_sets_.conv_add = kernels_.set({whole(ple_gated_), whole(ple_gated_normed_),
                                           ple_conv.data, whole(ple_state_), whole(hyper_)});
        if (verify4_batch_enabled_)
            build_verify_batch_sets(embedding, lm_head);
    }

    static void verify_to_transfer(VkCommandBuffer command) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;
        vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr,
            0, nullptr);
    }

    static void verify_to_compute(VkCommandBuffer command) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                                VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
        vkfn::CmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
            nullptr, 0, nullptr);
    }

    void record_verify_hc_start(VkCommandBuffer command, VerifyHcSets& sets,
                                bool with_injection) {
        Push push{kVerifyBatch, kHcCount, kDim, float_bits(1e-6f)};
        kernels_.dispatch(command, kernels_.p().group_rms_batch4, sets.norm,
                          &push, kVerifyBatch * kHcCount);
        compute_barrier(command);
        push = {kHcDim, 128, kHcDim / 4, kHcDim / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant, sets.quant[row],
                              &push, kHcDim / 128);
        compute_barrier(command);
        push = {kHcLowrank, kHcDim, kVerifyHcQuantU32, kHcLowrank};
        kernels_.dispatch(command, kernels_.p().q4_batch4, sets.down, &push,
                          (kHcLowrank + 1) / 2);
        if (with_injection) {
            push = {kHcCount, kHcDim, kVerifyHcQuantU32, kHcCount};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.inject,
                              &push, 2);
        }
        compute_barrier(command);
        push = {kVerifyBatch * kHcLowrank,
                float_bits(float(kHcCount)), 0, 0};
        kernels_.dispatch(command, kernels_.p().hc_act, sets.act, &push,
                          (kVerifyBatch * kHcLowrank + 63) / 64);
        compute_barrier(command);
        push = {kHcLowrank, 128, kHcLowrank / 4, kHcLowrank / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant,
                              sets.low_quant[row], &push,
                              (kHcLowrank + 127) / 128);
        compute_barrier(command);
        push = {kHcDim, kHcLowrank, kVerifyLowQuantU32, kHcDim};
        kernels_.dispatch(command, kernels_.p().q4_batch4, sets.up, &push,
                          kHcDim / 2);
        compute_barrier(command);
        push = {kVerifyBatch, kDim, kHcCount, 0};
        kernels_.dispatch(command, kernels_.p().hc_mix_batch4, sets.mix,
                          &push, (kVerifyBatch * kDim + 63) / 64);
        compute_barrier(command);
    }

    void record_verify_hc_apply(VkCommandBuffer command, VerifyHcSets& sets) {
        const Push push{kVerifyBatch, kDim, kHcCount,
                        float_bits(float(kHcCount))};
        kernels_.dispatch(command, kernels_.p().hc_inject_batch4, sets.apply,
                          &push, (kVerifyBatch * kHcDim + 63) / 64);
        compute_barrier(command);
    }

    void record_verify_attention(VkCommandBuffer command, uint32_t layer,
                                 uint32_t position) {
        VerifyLayerSets& sets = verify_batch_layers_[layer];
        Push push{kDim, 128, kDim / 4, kDim / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant,
                              sets.hidden_quant[row], &push, kDim / 128);
        compute_barrier(command);
        if (full_attention(layer)) {
            push = {12288, kDim, kVerifyDimQuantU32, 12288};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.qgate,
                              &push, 12288 / 2);
            push = {512, kDim, kVerifyDimQuantU32, 512};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.key,
                              &push, 512 / 2);
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.value,
                              &push, 512 / 2);
            compute_barrier(command);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                push = {full_index(layer), position + row,
                        kAttentionHeads, kRopeDim / 2};
                kernels_.dispatch(command, kernels_.p().qk, sets.qk[row],
                                  &push, kAttentionHeads);
                push = {full_index(layer), position + row, 0, 0};
                kernels_.dispatch(command, kernels_.p().store_value,
                                  sets.store_value[row], &push,
                                  (kKvHeads * kHeadDim + 63) / 64);
                compute_barrier(command);
                push = {full_index(layer), position + row,
                        kAttentionHeads, 0};
                kernels_.dispatch(command, kernels_.p().attention,
                                  sets.attention[row], &push,
                                  kAttentionHeads);
                compute_barrier(command);
                push = {kLinearValue, kAttentionHeads, 0, 0};
                kernels_.dispatch(command, kernels_.p().head_gate,
                                  sets.head_gate[row], &push,
                                  kLinearValue / 64);
                compute_barrier(command);
            }
        } else {
            push = {kLinearQkv, kDim, kVerifyDimQuantU32, kLinearQkv};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.gdn_qkv,
                              &push, kLinearQkv / 2);
            push = {kLinearValue, kDim, kVerifyDimQuantU32, kLinearValue};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.gdn_z,
                              &push, kLinearValue / 2);
            push = {96, kDim, kVerifyDimQuantU32, 96};
            kernels_.dispatch(command, kernels_.p().q4_batch4, sets.ab,
                              &push, 96 / 2);
            compute_barrier(command);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                push = {kLinearQkv, kConvWidth, 0, 0};
                kernels_.dispatch(command, kernels_.p().conv, sets.conv[row],
                                  &push, kLinearQkv / 64);
                compute_barrier(command);
                push = {kLinearValueHeads, kLinearHeadDim, 0, 0};
                kernels_.dispatch(command, kernels_.p().delta, sets.delta[row],
                                  &push, kLinearValueHeads);
                compute_barrier(command);
                if (row < 3) {
                    verify_to_transfer(command);
                    const uint64_t linear = linear_index(layer);
                    const VkBufferCopy conv_copy{
                        linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                        linear * uint64_t(kLinearQkv) * kConvWidth * 4,
                        uint64_t(kLinearQkv) * kConvWidth * 4};
                    vkfn::CmdCopyBuffer(command, conv_state_.handle,
                        verify_conv_snapshots_[row].handle, 1, &conv_copy);
                    const VkBufferCopy recurrent_copy{
                        linear * uint64_t(kLinearValueHeads) *
                            kLinearHeadDim * kLinearHeadDim * 4,
                        linear * uint64_t(kLinearValueHeads) *
                            kLinearHeadDim * kLinearHeadDim * 4,
                        uint64_t(kLinearValueHeads) * kLinearHeadDim *
                            kLinearHeadDim * 4};
                    vkfn::CmdCopyBuffer(command, recurrent_state_.handle,
                        verify_recurrent_snapshots_[row].handle, 1,
                        &recurrent_copy);
                    verify_to_compute(command);
                }
            }
        }
        push = {kLinearValue, 128, kLinearValue / 4, kLinearValue / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant,
                              sets.context_quant[row], &push,
                              kLinearValue / 128);
        compute_barrier(command);
        push = {kDim, kLinearValue, kVerifyContextQuantU32, kDim};
        kernels_.dispatch(command, kernels_.p().q4_batch4,
                          full_attention(layer) ? sets.attention_out
                                                : sets.gdn_out,
                          &push, kDim / 2);
        compute_barrier(command);
    }

    void record_verify_router_shared(VkCommandBuffer command, uint32_t layer) {
        VerifyLayerSets& sets = verify_batch_layers_[layer];
        Push push{kDim, 128, kDim / 4, kDim / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant,
                              sets.hidden_quant[row], &push, kDim / 128);
        compute_barrier(command);
        push = {kExperts, kDim, kVerifyDimQuantU32, kExperts};
        kernels_.dispatch(command, kernels_.p().q8_batch4, sets.router_gemv,
                          &push, kExperts / 4);
        compute_barrier(command);
        push = {kExperts, verify_active_topk_, 0, 0};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().router, sets.router[row],
                              &push, 1);
        compute_barrier(command);
        push = {kMoeDim, kDim, kVerifyDimQuantU32, kMoeDim};
        kernels_.dispatch(command, kernels_.p().q4_batch4, sets.shared_gate,
                          &push, kMoeDim / 2);
        kernels_.dispatch(command, kernels_.p().q4_batch4, sets.shared_up,
                          &push, kMoeDim / 2);
        push = {1, kDim, kVerifyDimQuantU32, 1};
        kernels_.dispatch(command, kernels_.p().q4_batch4,
                          sets.shared_expert_gate, &push, 1);
        compute_barrier(command);
        push = {kVerifyBatch * kMoeDim,
                float_bits(3.402823466e+38f), 0, 0};
        kernels_.dispatch(command, kernels_.p().swiglu, sets.shared_swiglu,
                          &push, kVerifyBatch * kMoeDim / 64);
        compute_barrier(command);
        push = {kMoeDim, 128, kMoeDim / 4, kMoeDim / 4};
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            kernels_.dispatch(command, kernels_.p().quant,
                              sets.shared_quant[row], &push,
                              kMoeDim / 128);
        compute_barrier(command);
        push = {kDim, kMoeDim, kVerifyMoeQuantU32, kDim};
        kernels_.dispatch(command, kernels_.p().q4_batch4, sets.shared_down,
                          &push, kDim / 2);
        compute_barrier(command);
    }

    std::array<uint32_t, 4> verify4_batch_experiment(
        const std::array<uint32_t, 4>& tokens, uint32_t position) {
        auto* token_words = static_cast<uint32_t*>(verify_tokens_.mapped);
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            token_words[row] = tokens[row];
        flush_buffer(runtime_, verify_tokens_);
        auto* ple_rows = static_cast<float*>(verify_ple_host_.mapped);
        for (uint32_t row = 0; row < kVerifyBatch; ++row) {
            ple_lookup_.lookup(tokens[row], ple_rows + uint64_t(row) * kDim);
            verify_ple_states_[row] = ple_lookup_.state();
        }
        dsv4::flush_buffer_range(runtime_, verify_ple_host_, 0,
                                 verify_ple_host_.size);
        const bool trace = std::getenv("QWEN38_VERIFY4_TRACE") != nullptr;

        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy ple_copy{0, 0, verify_ple_host_.size};
            vkfn::CmdCopyBuffer(command, verify_ple_host_.handle,
                                verify_ple_embedding_.handle, 1, &ple_copy);
            dsv4::transfer_barrier(command, verify_ple_embedding_);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                Push push{kVocabulary, kDim, kDim / 4, 0};
                kernels_.dispatch(command, kernels_.p().embedding,
                                  verify_batch_embedding_sets_[row], &push,
                                  (kDim + 63) / 64);
                compute_barrier(command);
                push = {kDim, kHcCount, 0, 0};
                kernels_.dispatch(command, kernels_.p().repeat_hc,
                                  verify_batch_repeat_sets_[row], &push,
                                  (kHcDim + 63) / 64);
                compute_barrier(command);
            }
        });
        compute_.wait(signal);
        ple_transfer_bytes_ += verify_ple_host_.size;

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            const auto pre_started = std::chrono::steady_clock::now();
            signal = compute_.submit([&](VkCommandBuffer command) {
                if (layer == 1) {
                    for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                        verify_to_transfer(command);
                        const VkBufferCopy hyper_to_single{
                            uint64_t(row) * kHcDim * sizeof(float), 0,
                            uint64_t(kHcDim) * sizeof(float)};
                        vkfn::CmdCopyBuffer(command, verify_batch_hyper_.handle,
                                            hyper_.handle, 1,
                                            &hyper_to_single);
                        const VkBufferCopy ple_to_single{
                            uint64_t(row) * kDim * sizeof(float), 0,
                            uint64_t(kDim) * sizeof(float)};
                        vkfn::CmdCopyBuffer(command,
                            verify_ple_embedding_.handle,
                            ple_embedding_.handle, 1, &ple_to_single);
                        verify_to_compute(command);
                        record_ple(command);
                        verify_to_transfer(command);
                        const VkBufferCopy hyper_from_single{
                            0, uint64_t(row) * kHcDim * sizeof(float),
                            uint64_t(kHcDim) * sizeof(float)};
                        vkfn::CmdCopyBuffer(command, hyper_.handle,
                            verify_batch_hyper_.handle, 1,
                            &hyper_from_single);
                        if (row < 3) {
                            const VkBufferCopy state_copy{0, 0, ple_state_.size};
                            vkfn::CmdCopyBuffer(command, ple_state_.handle,
                                verify_ple_snapshots_[row].handle, 1,
                                &state_copy);
                        }
                        verify_to_compute(command);
                    }
                }
                VerifyLayerSets& sets = verify_batch_layers_[layer];
                record_verify_hc_start(command, sets.attn_hc, true);
                record_verify_attention(command, layer, position);
                record_verify_hc_apply(command, sets.attn_hc);
                record_verify_hc_start(command, sets.mlp_hc, true);
                record_verify_router_shared(command, layer);
            });
            compute_.wait(signal);
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pre_started).count();
            if (trace) std::cerr << "batch4 layer " << layer << " pre ready\n";

            invalidate_buffer(runtime_, verify_routing_);
            constexpr uint32_t route_stride = 16u + 2u * kTopK;
            const uint32_t* routes =
                static_cast<const uint32_t*>(verify_routing_.mapped);
            std::array<std::array<uint32_t, kTopK>, kVerifyBatch> experts{};
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                for (uint32_t rank = 0; rank < verify_active_topk_; ++rank) {
                    experts[row][rank] = routes[row * route_stride + rank];
                    if (experts[row][rank] >= kExperts)
                        throw std::runtime_error(
                            "Qwen batch4 router returned invalid expert");
                }
            }
            const auto acquire_started = std::chrono::steady_clock::now();
            const DeviceExpertCache::Batch4 selection =
                device_cache_.resolve_four(layer, experts,
                                           verify_active_topk_);
            auto* meta = static_cast<uint32_t*>(verify_expert_meta_.mapped);
            std::fill(meta, meta + uint64_t(4 * kTopK) * 8, UINT32_MAX);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                for (uint32_t rank = 0; rank < verify_active_topk_; ++rank) {
                    const uint32_t unique = selection.unique_indices[row][rank];
                    meta[unique * 8u + row] = rank;
                    meta[unique * 8u + 4u + row] =
                        routes[row * route_stride + 16u + rank];
                }
            }
            dsv4::flush_buffer_range(runtime_, verify_expert_meta_, 0,
                uint64_t(selection.unique_count) * 8 * sizeof(uint32_t));
            verify_unique_experts_ += selection.unique_count;
            verify_occurrences_ += kVerifyBatch * verify_active_topk_;
            verify_reused_occurrences_ += selection.reused_occurrences;
            std::array<uint32_t, 4 * kTopK> missing_experts{};
            std::array<uint32_t, 4 * kTopK> missing_unique{};
            uint32_t missing_count = 0;
            for (uint32_t unique = 0; unique < selection.unique_count; ++unique) {
                if (!selection.unique_misses[unique]) continue;
                missing_experts[missing_count] =
                    selection.unique_experts[unique];
                missing_unique[missing_count++] = unique;
            }
            const HostExpertCache::Many sources = host_cache_.resolve_many(
                layer, missing_experts.data(), missing_count);
            if (sources.disk_reads) ++cold_stalled_layers_;
            std::vector<ParallelCopyPool::Task> copy_tasks;
            copy_tasks.reserve(missing_count);
            for (uint32_t index = 0; index < missing_count; ++index) {
                copy_tasks.push_back({verify_staging_[index].mapped,
                                      sources.pointers[index],
                                      size_t(kExpertRecordBytes)});
                host_copy_bytes_ += kExpertRecordBytes;
            }
            copy_pool_.copy(copy_tasks);
            for (uint32_t index = 0; index < missing_count; ++index)
                dsv4::flush_buffer_range(runtime_, verify_staging_[index], 0,
                                         kExpertRecordBytes);
            uint64_t ready = 0;
            if (missing_count) {
                ready = transfer_.submit([&](VkCommandBuffer command) {
                    for (uint32_t index = 0; index < missing_count; ++index) {
                        const uint32_t unique = missing_unique[index];
                        const VkBufferCopy copy{
                            0, uint64_t(selection.unique_slots[unique]) *
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
                VerifyLayerSets& sets = verify_batch_layers_[layer];
                for (uint32_t unique = 0;
                     unique < selection.unique_count; ++unique) {
                    Push push{unique, kVerifyDimQuantU32, 0, 0};
                    kernels_.dispatch(command,
                        kernels_.p().expert_gate_verify4,
                        layers_[layer].expert_gate_verify4[
                            selection.unique_slots[unique]],
                        &push, kMoeDim / 2);
                }
                compute_barrier(command);
                Push push{kVerifyBatch * kTopK * kMoeDim, 128,
                          kVerifyBatch * kTopK * kMoeDim / 4,
                          kVerifyBatch * kTopK * kMoeDim / 4};
                kernels_.dispatch(command, kernels_.p().quant,
                    verify_batch_expert_quant_set_, &push,
                    kVerifyBatch * kTopK * kMoeDim / 128);
                compute_barrier(command);
                constexpr uint32_t expert_scale_base =
                    kVerifyBatch * kTopK * kMoeDim / 4;
                for (uint32_t unique = 0;
                     unique < selection.unique_count; ++unique) {
                    push = {unique, expert_scale_base, 0, 0};
                    kernels_.dispatch(command,
                        kernels_.p().expert_down_verify4,
                        layers_[layer].expert_down_verify4[
                            selection.unique_slots[unique]],
                        &push, kDim / 2);
                }
                compute_barrier(command);
                push = {kDim, verify_active_topk_, 0, 0};
                for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                    kernels_.dispatch(command, kernels_.p().reduce,
                                      sets.reduce[row], &push, kDim / 64);
                }
                compute_barrier(command);
                record_verify_hc_apply(command, sets.mlp_hc);
            }, ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            compute_.wait(signal);
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
            if (trace) std::cerr << "batch4 layer " << layer
                                 << " expert ready\n";
        }

        signal = compute_.submit([&](VkCommandBuffer command) {
            record_verify_hc_start(command, verify_batch_final_hc_, false);
            Push push{kDim, 128, kDim / 4, kDim / 4};
            for (uint32_t row = 0; row < kVerifyBatch; ++row)
                kernels_.dispatch(command, kernels_.p().quant,
                    verify_batch_final_quant_sets_[row], &push, kDim / 128);
            compute_barrier(command);
            push = {kVocabulary, kDim, kVerifyDimQuantU32, kVocabulary};
            kernels_.dispatch(command, kernels_.p().q8_batch4,
                              verify_batch_lm_head_set_, &push,
                              (kVocabulary + 3) / 4);
            compute_barrier(command);
            for (uint32_t row = 0; row < kVerifyBatch; ++row) {
                push = {kVocabulary, 256, 0, 0};
                kernels_.dispatch(command, kernels_.p().argmax,
                    verify_batch_argmax_sets_[row], &push, 256);
                compute_barrier(command);
                push = {kVocabulary, 256, 1, 0};
                kernels_.dispatch(command, kernels_.p().argmax,
                    verify_batch_argmax_sets_[row], &push, 1);
                compute_barrier(command);
            }
            verify_to_transfer(command);
            if (verify_logits_host_enabled_) {
                const VkBufferCopy logits_copy{
                    0, 0, verify_batch_logits_.size};
                vkfn::CmdCopyBuffer(command, verify_batch_logits_.handle,
                    verify_logits_host_.handle, 1, &logits_copy);
            }
            const VkBufferCopy hyper_copy{
                uint64_t(kVerifyBatch - 1) * kHcDim * sizeof(float), 0,
                uint64_t(kHcDim) * sizeof(float)};
            vkfn::CmdCopyBuffer(command, verify_batch_hyper_.handle,
                                hyper_.handle, 1, &hyper_copy);
            verify_to_compute(command);
        });
        compute_.wait(signal);
        if (verify_logits_host_enabled_)
            invalidate_buffer(runtime_, verify_logits_host_);
        invalidate_buffer(runtime_, verify_tokens_);
        std::array<uint32_t, 4> result{};
        token_words = static_cast<uint32_t*>(verify_tokens_.mapped);
        for (uint32_t row = 0; row < kVerifyBatch; ++row)
            result[row] = token_words[row];
        return result;
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

    void reset_decode_metrics() {
        host_cache_.reset_metrics();
        device_cache_.reset_metrics();
        ple_lookup_.reset_metrics();
        transfer_bytes_ = 0;
        ple_transfer_bytes_ = 0;
        host_copy_bytes_ = 0;
        cold_stalled_layers_ = 0;
        progressive_layers_ = progressive_resident_ranks_ = 0;
        progressive_ram_ranks_ = progressive_disk_ranks_ = 0;
        pre_seconds_ = pre_submit_seconds_ = pre_wait_seconds_ = 0;
        acquisition_seconds_ = expert_seconds_ = 0;
        verify_unique_experts_ = verify_occurrences_ = 0;
        verify_reused_occurrences_ = 0;
    }

    uint32_t run_token(uint32_t token, uint32_t position) {
#ifdef OVLLM_LONG_CONTEXT_FORK
        if (position >= max_context_)
#else
        if (position >= kMaximumContext)
#endif
            throw std::runtime_error("Qwen runtime context cap reached");
        *static_cast<uint32_t*>(token_.mapped) = token;
        flush_buffer(runtime_, token_);
        ple_lookup_.lookup(token, static_cast<float*>(ple_host_.mapped));
        dsv4::flush_buffer_range(runtime_, ple_host_, 0, kDim * 4ull);
        const uint64_t ple_ready = transfer_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{0, 0, kDim * 4ull};
            vkfn::CmdCopyBuffer(command, ple_host_.handle,
                                ple_embedding_.handle, 1, &copy);
            dsv4::transfer_barrier(command, ple_embedding_);
        });
        ple_transfer_bytes_ += kDim * 4ull;
        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            Push push{kVocabulary, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().embedding, embedding_set_,
                              &push, (kDim + 63) / 64);
            compute_barrier(command);
            push = {kDim, kHcCount, 0, 0};
            kernels_.dispatch(command, kernels_.p().repeat_hc, repeat_set_,
                              &push, (kHcDim + 63) / 64);
            compute_barrier(command);
        });
        compute_.wait(signal);

        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            auto started = std::chrono::steady_clock::now();
            signal = compute_.submit([&](VkCommandBuffer command) {
                if (layer == 1) record_ple(command);
                record_hc_start(command, layers_[layer].attn_hc, true);
                record_attention(command, layer, position);
                record_hc_apply(command, layers_[layer].attn_hc);
                record_hc_start(command, layers_[layer].mlp_hc, true);
                record_router(command, layer);
            }, layer == 1 ? transfer_.semaphore() : VK_NULL_HANDLE,
               layer == 1 ? ple_ready : 0,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            const auto submitted = std::chrono::steady_clock::now();
            pre_submit_seconds_ += std::chrono::duration<double>(
                submitted - started).count();
            compute_.wait(signal);
            pre_wait_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - submitted).count();
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
            const DeviceExpertCache::Selection selection =
                device_cache_.resolve(layer, experts);
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
                    if (!batch.sources.imported_buffers[rank]) {
                        std::memcpy(staging_[rank].mapped,
                                    batch.sources.pointers[rank],
                                    kExpertRecordBytes);
                        host_copy_bytes_ += kExpertRecordBytes;
                        dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                                 kExpertRecordBytes);
                    }
                    ram_ranks.push_back(rank);
                }
                uint64_t last_ready = 0;
                if (!ram_ranks.empty()) {
                    last_ready = transfer_.submit([&](VkCommandBuffer command) {
                        for (uint32_t rank : ram_ranks) {
                            const VkBufferCopy copy{
                                batch.sources.imported_buffers[rank]
                                    ? batch.sources.imported_offsets[rank] : 0,
                                uint64_t(selected_slots_[rank]) *
                                       kExpertRecordBytes,
                                kExpertRecordBytes};
                            vkfn::CmdCopyBuffer(
                                command,
                                batch.sources.imported_buffers[rank]
                                    ? batch.sources.imported_buffers[rank]
                                    : staging_[rank].handle,
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
                        record_expert_finish(command, layer);
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
                layer, experts, selection.misses, direct_destinations);
            if (sources.disk_reads) ++cold_stalled_layers_;

            std::vector<uint32_t> copied;
            std::vector<ParallelCopyPool::Task> host_tasks;
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                if (!selection.misses[rank]) continue;
                if (!sources.direct[rank]) {
                    if (!sources.imported_buffers[rank]) {
                        host_tasks.push_back({staging_[rank].mapped,
                                              sources.pointers[rank],
                                              kExpertRecordBytes});
                        host_copy_bytes_ += kExpertRecordBytes;
                    }
                }
                copied.push_back(rank);
            }
            if (parallel_host_copy_)
                copy_pool_.copy(host_tasks);
            else
                for (const ParallelCopyPool::Task& task : host_tasks)
                    std::memcpy(task.destination, task.source, task.bytes);
            for (uint32_t rank : copied)
                if (!sources.imported_buffers[rank])
                    dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                             kExpertRecordBytes);
            uint64_t ready = 0;
            if (!copied.empty()) {
                ready = transfer_.submit([&](VkCommandBuffer command) {
                    for (uint32_t rank : copied) {
                        VkBufferCopy copy{
                            sources.imported_buffers[rank]
                                ? sources.imported_offsets[rank] : 0,
                            uint64_t(selected_slots_[rank]) * kExpertRecordBytes,
                            kExpertRecordBytes};
                        vkfn::CmdCopyBuffer(
                                            command,
                                            sources.imported_buffers[rank]
                                                ? sources.imported_buffers[rank]
                                                : staging_[rank].handle,
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
            expert_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - expert_started).count();
        }

        signal = compute_.submit([&](VkCommandBuffer command) {
            record_hc_start(command, final_hc_, false);
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

    void record_hc_start(VkCommandBuffer command, HcSets& sets,
                         bool with_injection) {
        Push push{kHcCount, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().group_rms, sets.norm, &push,
                          kHcCount);
        compute_barrier(command);
        push = {kHcDim, 128, kHcDim / 4, kHcDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.quant, &push,
                          (kHcDim + 127) / 128);
        compute_barrier(command);
        push = {kHcLowrank, kHcDim, kHcDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(push.b), sets.down, &push,
                          (kHcLowrank + 7) / 8);
        if (with_injection) {
            push = {kHcCount, kHcDim, kHcDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.inject, &push, 1);
        }
        compute_barrier(command);
        push = {kHcLowrank, float_bits(float(kHcCount)), 0, 0};
        kernels_.dispatch(command, kernels_.p().hc_act, sets.act, &push,
                          (kHcLowrank + 63) / 64);
        compute_barrier(command);
        push = {kHcLowrank, 128, kHcLowrank / 4, kHcLowrank / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.low_quant, &push,
                          (kHcLowrank + 127) / 128);
        compute_barrier(command);
        push = {kHcDim, kHcLowrank, kHcLowrank / 4, 0};
        kernels_.dispatch(command, q4_pipeline(push.b), sets.up, &push,
                          kHcDim / 8);
        compute_barrier(command);
        push = {kDim, kHcCount, 0, 0};
        kernels_.dispatch(command, kernels_.p().hc_mix, sets.mix, &push,
                          (kDim + 63) / 64);
        compute_barrier(command);
    }

    void record_hc_apply(VkCommandBuffer command, HcSets& sets) {
        const Push push{kDim, kHcCount, float_bits(float(kHcCount)), 0};
        kernels_.dispatch(command, kernels_.p().hc_inject, sets.apply, &push,
                          (kHcDim + 63) / 64);
        compute_barrier(command);
    }

    void record_ple(VkCommandBuffer command) {
        Push push{kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, ple_sets_.quant, &push,
                          kDim / 128);
        compute_barrier(command);
        push = {kHcDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(push.b), ple_sets_.key, &push,
                          kHcDim / 8);
        push = {kDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(push.b), ple_sets_.value, &push,
                          kDim / 8);
        compute_barrier(command);
        push = {kHcCount, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().group_rms, ple_sets_.key_norm,
                          &push, kHcCount);
        kernels_.dispatch(command, kernels_.p().group_rms, ple_sets_.query_norm,
                          &push, kHcCount);
        compute_barrier(command);
        push = {kDim, kHcCount, 0, 0};
        kernels_.dispatch(command, kernels_.p().ple_gate, ple_sets_.gate,
                          &push, kHcCount);
        compute_barrier(command);
        push = {kHcCount, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().group_rms,
                          ple_sets_.gated_norm, &push, kHcCount);
        compute_barrier(command);
        push = {kHcDim, kPleHistory, 3, 4};
        kernels_.dispatch(command, kernels_.p().ple_conv_add,
                          ple_sets_.conv_add, &push,
                          (kHcDim + 63) / 64);
        compute_barrier(command);
    }

    void record_attention(VkCommandBuffer command, uint32_t layer,
                          uint32_t position) {
        LayerSets& sets = layers_[layer];
        Push push{kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.hidden_quant, &push,
                          kDim / 128);
        compute_barrier(command);

        if (full_attention(layer)) {
            push = {12288, kDim, kDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.qgate, &push, 1536);
            push = {512, kDim, kDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.key, &push, 64);
            kernels_.dispatch(command, q4_pipeline(push.b), sets.value, &push, 64);
            compute_barrier(command);
#ifdef OVLLM_LONG_CONTEXT_FORK
            push = {position, kAttentionHeads, kRopeDim / 2, max_context_};
#else
            push = {full_index(layer), position, kAttentionHeads, kRopeDim / 2};
#endif
            kernels_.dispatch(command, kernels_.p().qk, sets.qk, &push,
                              kAttentionHeads);
#ifdef OVLLM_LONG_CONTEXT_FORK
            push = {position, max_context_, 0, 0};
#else
            push = {full_index(layer), position, 0, 0};
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
                              &push, active_chunks * (kAttentionHeads / 8u));
            compute_barrier(command);
            push = {active_chunks, kAttentionHeads, 0, 0};
            kernels_.dispatch(command, kernels_.p().attention_reduce,
                              sets.attention_reduce, &push, kAttentionHeads);
#else
            push = {full_index(layer), position, kAttentionHeads, 0};
            kernels_.dispatch(command, kernels_.p().attention, sets.attention,
                              &push, kAttentionHeads);
#endif
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
            kernels_.dispatch(command, q4_pipeline(push.b),
                               sets.attention_out, &push, kDim / 8);
        } else {
            push = {kLinearQkv, kDim, kDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.gdn_qkv, &push,
                              kLinearQkv / 8);
            push = {kLinearValue, kDim, kDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.gdn_z, &push,
                              kLinearValue / 8);
            push = {96, kDim, kDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(push.b), sets.ab, &push, 12);
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
            kernels_.dispatch(command, q4_pipeline(push.b), sets.gdn_out,
                               &push, kDim / 8);
        }
        compute_barrier(command);
    }

    void record_router(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        Push push{kDim, 128, kDim / 4, kDim / 4};
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
        kernels_.dispatch(command, q4_pipeline(push.b), sets.shared_gate, &push,
                          kMoeDim / 8);
        kernels_.dispatch(command, q4_pipeline(push.b), sets.shared_up, &push,
                          kMoeDim / 8);
        push = {1, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(push.b), sets.shared_expert_gate,
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
        kernels_.dispatch(command, q4_pipeline(push.b), sets.shared_down, &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_experts(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        if (batch_experts_) {
            auto* words = static_cast<uint32_t*>(routing_.mapped);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                words[26u + rank] = selected_slots_[rank];
            }
            flush_buffer(runtime_, routing_);
            Push push{0, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate_batch,
                               sets.expert_gate_batch, &push,
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
                               sets.expert_down_batch, &push,
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
        record_hc_apply(command, sets.mlp_hc);
    }

    void record_experts_individual(VkCommandBuffer command, uint32_t layer) {
        LayerSets& sets = layers_[layer];
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            Push push{rank, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate,
                              sets.expert_gate[selected_slots_[rank]], &push,
                              kMoeDim / 8);
        }
        compute_barrier(command);
        Push push{kTopK * kMoeDim, 128, kTopK * kMoeDim / 4,
                  kTopK * kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, expert_quant_set_, &push,
                          kTopK * kMoeDim / 128);
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            push = {rank, kTopK * kMoeDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_down,
                              sets.expert_down[selected_slots_[rank]], &push,
                              kDim / 8);
        }
        compute_barrier(command);
        push = {kDim, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
        record_hc_apply(command, sets.mlp_hc);
    }

    void record_routed_experts_individual(VkCommandBuffer command,
                                           uint32_t layer) {
        LayerSets& sets = layers_[layer];
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            Push push{rank, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate,
                              sets.expert_gate[selected_slots_[rank]], &push,
                              kMoeDim / 8);
        }
        compute_barrier(command);
        Push push{kTopK * kMoeDim, 128, kTopK * kMoeDim / 4,
                  kTopK * kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, expert_quant_set_, &push,
                          kTopK * kMoeDim / 128);
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            push = {rank, kTopK * kMoeDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_down,
                              sets.expert_down[selected_slots_[rank]], &push,
                              kDim / 8);
        }
        compute_barrier(command);
    }

    void record_expert_rank(VkCommandBuffer command, uint32_t layer,
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
        push = {rank, kTopK * kMoeDim / 4u, 0, 0};
        kernels_.dispatch(command, kernels_.p().expert_down,
                          sets.expert_down[selected_slots_[rank]], &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_expert_finish(VkCommandBuffer command, uint32_t layer) {
        const Push push{kDim, kTopK, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
        record_hc_apply(command, layers_[layer].mlp_hc);
    }

    void destroy_all() {
        for (Buffer* buffer : {&rope_, &kv_cache_,
#ifdef OVLLM_LONG_CONTEXT_FORK
                               &attention_partial_,
#endif
                               &recurrent_state_,
                               &conv_state_, &argmax_workspace_, &logits_,
                               &ple_state_, &ple_gated_normed_, &ple_gated_,
                               &ple_value_, &ple_key_normed_, &ple_key_,
                               &ple_embedding_, &ple_host_,
                               &expert_outputs_, &expert_quant_,
                               &expert_intermediate_, &router_logits_,
                               &shared_expert_gate_, &shared_output_,
                               &shared_intermediate_, &shared_up_values_,
                               &shared_gate_values_, &quant_, &ab_, &z_,
                               &convolved_qkv_, &mixed_qkv_, &context_,
                               &value_, &key_, &qgate_, &block_output_,
                               &hc_injection_, &hc_low_quant_, &hc_low_,
                               &hc_mix_weights_, &hc_normed_, &hyper_,
                               &normalized_, &hidden_,
                               &routing_, &token_})
            destroy_buffer(runtime_, *buffer);
        if (verify4_enabled_) {
            for (Buffer& buffer : verify_recurrent_snapshots_)
                destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_conv_snapshots_)
                destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_ple_snapshots_)
                destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_quant_)
                destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_hyper_)
                destroy_buffer(runtime_, buffer);
            for (Buffer& buffer : verify_hidden_)
                destroy_buffer(runtime_, buffer);
            for (Buffer* buffer : {&verify_ple_embedding_, &verify_ple_host_,
                                   &verify_routing_, &verify_tokens_})
                destroy_buffer(runtime_, *buffer);
            if (verify4_batch_enabled_) {
                if (verify_logits_host_enabled_)
                    destroy_buffer(runtime_, verify_logits_host_);
                for (Buffer* buffer : {
                         &verify_batch_expert_outputs_,
                         &verify_batch_expert_quant_,
                         &verify_batch_expert_intermediate_,
                         &verify_expert_meta_,
                         &verify_batch_argmax_, &verify_batch_logits_,
                         &verify_batch_shared_expert_gate_,
                         &verify_batch_shared_output_,
                         &verify_batch_shared_intermediate_,
                         &verify_batch_shared_up_, &verify_batch_shared_gate_,
                         &verify_batch_router_logits_, &verify_batch_ab_,
                         &verify_batch_z_, &verify_batch_convolved_qkv_,
                         &verify_batch_mixed_qkv_, &verify_batch_context_,
                         &verify_batch_value_, &verify_batch_key_,
                         &verify_batch_qgate_, &verify_batch_moe_quant_,
                         &verify_batch_low_quant_, &verify_batch_quant_,
                         &verify_batch_block_, &verify_batch_hidden_,
                         &verify_batch_hc_injection_, &verify_batch_hc_low_,
                         &verify_batch_hc_mix_, &verify_batch_hc_normed_,
                         &verify_batch_hyper_})
                    destroy_buffer(runtime_, *buffer);
            }
        }
    }

    const Runtime& runtime_;
    DeviceWeights weights_;
    ExpertFile expert_file_;
    PleLookup ple_lookup_;
    HostExpertCache host_cache_;
    DeviceExpertCache device_cache_;
    Kernels kernels_;
    dsv4::FiniteQueue compute_, transfer_;
    Buffer token_{}, routing_{}, hidden_{}, normalized_{};
    Buffer hyper_{}, hc_normed_{}, hc_mix_weights_{}, hc_low_{};
    Buffer hc_low_quant_{}, hc_injection_{}, block_output_{};
    Buffer qgate_{}, key_{}, value_{}, context_{};
    Buffer mixed_qkv_{}, convolved_qkv_{}, z_{}, ab_{}, quant_{};
    Buffer shared_gate_values_{}, shared_up_values_{}, shared_intermediate_{};
    Buffer shared_output_{}, shared_expert_gate_{}, router_logits_{};
    Buffer expert_intermediate_{}, expert_quant_{}, expert_outputs_{};
    Buffer logits_{}, argmax_workspace_{}, conv_state_{}, recurrent_state_{};
    Buffer ple_host_{}, ple_embedding_{}, ple_key_{}, ple_key_normed_{};
    Buffer ple_value_{}, ple_gated_{}, ple_gated_normed_{}, ple_state_{};
    Buffer kv_cache_{}, rope_{};
    Buffer verify_tokens_{}, verify_routing_{};
    Buffer verify_ple_host_{}, verify_ple_embedding_{};
    std::array<Buffer, 4> verify_hidden_{}, verify_hyper_{}, verify_quant_{};
    std::array<Buffer, 3> verify_conv_snapshots_{};
    std::array<Buffer, 3> verify_recurrent_snapshots_{};
    std::array<Buffer, 3> verify_ple_snapshots_{};
    Buffer verify_batch_hyper_{}, verify_batch_hc_normed_{};
    Buffer verify_batch_hc_mix_{}, verify_batch_hc_low_{};
    Buffer verify_batch_hc_injection_{}, verify_batch_hidden_{};
    Buffer verify_batch_block_{}, verify_batch_quant_{};
    Buffer verify_batch_low_quant_{}, verify_batch_moe_quant_{};
    Buffer verify_batch_qgate_{}, verify_batch_key_{}, verify_batch_value_{};
    Buffer verify_batch_context_{}, verify_batch_mixed_qkv_{};
    Buffer verify_batch_convolved_qkv_{}, verify_batch_z_{}, verify_batch_ab_{};
    Buffer verify_batch_router_logits_{}, verify_batch_shared_gate_{};
    Buffer verify_batch_shared_up_{}, verify_batch_shared_intermediate_{};
    Buffer verify_batch_shared_output_{}, verify_batch_shared_expert_gate_{};
    Buffer verify_batch_logits_{}, verify_batch_argmax_{};
    Buffer verify_logits_host_{};
    Buffer verify_expert_meta_{}, verify_batch_expert_intermediate_{};
    Buffer verify_batch_expert_quant_{}, verify_batch_expert_outputs_{};
#ifdef OVLLM_LONG_CONTEXT_FORK
    Buffer attention_partial_{};
    uint32_t max_context_ = kDefaultContext;
    uint32_t attention_chunks_ = 2;
#endif
    std::vector<Buffer> staging_, verify_staging_;
    std::vector<LayerSets> layers_;
    std::array<uint32_t, kTopK> selected_slots_{};
    HcSets final_hc_{};
    PleSets ple_sets_{};
    VkDescriptorSet embedding_set_{}, repeat_set_{}, final_quant_set_{};
    std::array<VkDescriptorSet, 4> verify_embedding_sets_{};
    std::array<VkDescriptorSet, 4> verify_repeat_sets_{};
    std::array<VkDescriptorSet, kVerifyBatch> verify_batch_embedding_sets_{};
    std::array<VkDescriptorSet, kVerifyBatch> verify_batch_repeat_sets_{};
    std::array<VkDescriptorSet, kVerifyBatch> verify_batch_final_quant_sets_{};
    std::array<VkDescriptorSet, kVerifyBatch> verify_batch_argmax_sets_{};
    VkDescriptorSet verify_batch_lm_head_set_{};
    VkDescriptorSet verify_batch_expert_quant_set_{};
    VerifyHcSets verify_batch_final_hc_{};
    std::vector<VerifyLayerSets> verify_batch_layers_;
    VkDescriptorSet lm_head_set_{}, argmax_set_{}, expert_quant_set_{};
    std::array<VkDescriptorSet, kTopK> expert_rank_quant_sets_{};
    VkDescriptorSet reduce_set_{}, expert_gate_batch_set_{}, expert_down_batch_set_{};
    bool batch_experts_ = false;
    bool progressive_experts_ = false;
    bool q4_wave32_mid_ = false;
    bool q4_one_lane_ = false;
    bool q4_one_lane_mid_ = false;
    bool parallel_host_copy_ = false;
    bool verify4_enabled_ = false;
    bool verify4_batch_enabled_ = false;
    uint32_t verify_active_topk_ = kTopK;
    bool verify_logits_host_enabled_ = false;
    ParallelCopyPool copy_pool_;
    std::unique_ptr<dsv4::experiment::FiniteQueueRing<12>> progressive_compute_;
    uint64_t activation_device_bytes_ = 0;
    uint64_t transfer_bytes_ = 0, host_copy_bytes_ = 0;
    uint64_t ple_transfer_bytes_ = 0;
    uint64_t cold_stalled_layers_ = 0;
    uint64_t progressive_layers_ = 0;
    uint64_t progressive_resident_ranks_ = 0;
    uint64_t progressive_ram_ranks_ = 0;
    uint64_t progressive_disk_ranks_ = 0;
    double decode_seconds_ = 0, pre_seconds_ = 0;
    double pre_submit_seconds_ = 0, pre_wait_seconds_ = 0;
    double acquisition_seconds_ = 0, expert_seconds_ = 0;
    std::array<PleLookup::State, 4> verify_ple_states_{};
    uint64_t verify_unique_experts_ = 0;
    uint64_t verify_occurrences_ = 0;
    uint64_t verify_reused_occurrences_ = 0;
};

static uint64_t ram_budget() {
    const char* text = std::getenv("QWEN38_RAM_GIB");
    const double gib = text ? std::stod(text) : 24.0;
    if (gib < 2.0 || gib > 80.0)
        throw std::runtime_error("QWEN38_RAM_GIB must be 2..80");
    return static_cast<uint64_t>(gib * 1024.0 * 1024.0 * 1024.0);
}

static uint32_t device_slots() {
    const char* text = std::getenv("QWEN38_DEVICE_SLOTS_PER_LAYER");
    const uint32_t slots = text ? static_cast<uint32_t>(std::stoul(text)) : 48;
    if (slots < kTopK || slots > kExperts)
        throw std::runtime_error("QWEN38_DEVICE_SLOTS_PER_LAYER must be 10..512");
    return slots;
}

} // namespace qwen38

int qwen38_cli_main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: amd_qwen38.exe <runtime-dir> "
                         "<prompt|--inspect|--tokenize> [new-tokens]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file((directory / "tokenizer.ovb").string());
        qwen38::Tokenizer tokenizer(tokenizer_file);
        if (std::strcmp(argv[2], "--tokenize") == 0) {
            if (argc < 4) throw std::runtime_error("tokenize text required");
            const bool thinking = std::getenv("QWEN38_NO_THINK") == nullptr;
            const std::vector<uint32_t> tokens =
                tokenizer.chat_prompt(argv[3], thinking);
            std::cout << "tokens:";
            for (uint32_t token : tokens) std::cout << ' ' << token;
            std::cout << '\n';
            return 0;
        }
        qwen38::SharedIndex index(directory / "model-q4g64.ovs");
#ifdef OVLLM_QWEN38_Q3_EXPERTS
        const auto expert_path = directory / "experts-q3g64.ovx";
#else
        const auto expert_path = directory / "experts-q4g64.ovx";
#endif
        qwen38::ExpertFile inspect_experts(expert_path);
        qwen38::PleLookup inspect_ple(directory / "ple-fp8.ovp");
        if (std::strcmp(argv[2], "--inspect") == 0) {
            std::cout << "Qwen3.8-Flash-Next runtime containers validated\n";
            return 0;
        }

        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << '\n';
        const uint32_t count = argc >= 4
            ? static_cast<uint32_t>(std::stoul(argv[3])) : 8;
        const uint64_t budget = qwen38::ram_budget();
        const uint32_t slots = qwen38::device_slots();
        const bool thinking = std::getenv("QWEN38_NO_THINK") == nullptr;
        std::vector<uint32_t> result;
        double decode = 0, pre = 0, pre_submit = 0, pre_wait = 0;
        double acquisition = 0, expert = 0;
        uint64_t device_hits = 0, device_misses = 0;
        std::array<uint64_t, qwen38::kLayers> device_layer_misses{};
        uint64_t ram_hits = 0, ram_misses = 0, disk = 0, transfer = 0;
        uint64_t ple_disk = 0, ple_transfer = 0;
        uint64_t ram_bypasses = 0;
        uint64_t host_copy = 0, cold_layers = 0, ram = 0, vram = 0;
        uint64_t progressive_layers = 0, progressive_resident = 0;
        uint64_t progressive_ram = 0, progressive_disk = 0;
        uint32_t host_slots = 0;
        {
            qwen38::QwenEngine engine(
                 runtime, index, expert_path,
                directory / "ple-fp8.ovp",
                std::filesystem::absolute(argv[0]).parent_path(), budget, slots);
#ifdef OVLLM_QWEN38_Q3_EXPERTS
            std::cout << "precision: Q3G64T experts, Q4G64T shared/dense, "
#else
            std::cout << "precision: Q4G64T experts/shared/dense, "
#endif
                         "Q8 embedding/head/router, official FP8 PLE\n"
                      << "RAM budget: "
                      << double(budget) / double(1ull << 30) << " GiB\n"
                      << "expert slots device/RAM: " << slots
                      << " per layer / " << engine.host_slots() << " global\n";
            result = engine.generate(tokenizer,
                tokenizer.chat_prompt(argv[2], thinking), count);
            decode = engine.decode_seconds();
            pre = engine.pre_seconds();
            pre_submit = engine.pre_submit_seconds();
            pre_wait = engine.pre_wait_seconds();
            acquisition = engine.acquisition_seconds();
            expert = engine.expert_seconds();
            device_hits = engine.device_hits();
            device_misses = engine.device_misses();
            device_layer_misses = engine.device_layer_misses();
            ram_hits = engine.ram_hits();
            ram_misses = engine.ram_misses();
            ram_bypasses = engine.ram_admission_bypasses();
            disk = engine.disk_bytes();
            ple_disk = engine.ple_disk_bytes();
            transfer = engine.transfer_bytes();
            ple_transfer = engine.ple_transfer_bytes();
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
                  << device_misses << "\ndevice misses by layer:";
        for (uint64_t misses : device_layer_misses)
            std::cout << ' ' << misses;
        std::cout << "\nRAM hits/misses: " << ram_hits << '/'
                  << ram_misses << "\nRAM admission bypasses: "
                  << ram_bypasses << '\n'
                  << "expert SSD / host-copy / H2D bytes per output: "
                  << disk / divisor << " / " << host_copy / divisor << " / "
                  << transfer / divisor << '\n'
                  << "PLE SSD / H2D bytes per output: "
                  << ple_disk / divisor << " / " << ple_transfer / divisor << '\n'
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
                  << "attention+router submit / GPU-wait wall s: "
                  << pre_submit << " / " << pre_wait << '\n'
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
        std::cerr << "Qwen3.8 runtime error: " << error.what() << '\n';
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}

#ifndef OVLLM_QWEN38_RUNTIME_ONLY
int main(int argc, char** argv) {
    return qwen38_cli_main(argc, argv);
}
#endif
