#define OVLLM_DSV4_RUNTIME_ONLY
#include "m13_deepseek_v4.cpp"

#include <future>
#include <map>

namespace dsv4 {

constexpr uint32_t kDsparkStages = 3;
constexpr uint32_t kDsparkBlock = 5;
constexpr uint32_t kDsparkNoiseToken = 128799;
constexpr uint32_t kDsparkMarkovRank = 256;
// Each draft stage keeps a small independent expert cache.  Stage-local slots
// avoid cross-stage eviction churn during a speculative pass while keeping the
// bounded cache footprint modest enough for ten main-model slots per layer.
constexpr uint32_t kDsparkCacheSlots = 12;
constexpr uint32_t kDsparkKvSlots = kWindow + kDsparkBlock;
constexpr uint64_t kDsparkHostStagingBytes =
    static_cast<uint64_t>(kTopK) * kExpertRecordBytes;

#pragma pack(push, 1)
struct DsparkHeader {
    char magic[8];
    uint32_t version, header_bytes, tensor_entry_bytes, shared_format;
    uint32_t stages, block_size, noise_token, dimension, moe_dimension;
    uint32_t experts, top_k, hc, hc_iterations, vocabulary, markov_rank;
    uint32_t target0, target1, target2, reserved0, reserved1;
    uint64_t group_table_offset, group_count, tensor_table_offset, tensor_count;
    uint64_t data_offset, file_bytes, expert_record_bytes, expert_records;
    uint64_t reserved_q0, reserved_q1;
    uint8_t padding[88];
};

struct DsparkExpertHeader {
    char magic[8];
    uint32_t version, header_bytes, stages, dimension, moe_dimension;
    uint32_t experts, record_bytes, reserved;
    uint64_t data_offset, total_records, file_bytes;
    uint64_t w1_weight_offset, w1_scale_offset, w3_weight_offset;
    uint64_t w3_scale_offset, w2_weight_offset, w2_scale_offset;
    uint8_t padding[16];
};
#pragma pack(pop)

static_assert(sizeof(DsparkHeader) == 256, "DSpark header ABI drift");
static_assert(sizeof(DsparkExpertHeader) == 128, "DSpark expert header ABI drift");

class DsparkIndex {
public:
    explicit DsparkIndex(const ReadOnlyMapping& file) : file_(file) {
        if (file.size() < kFileHeaderBytes) throw std::runtime_error("Truncated dspark.ovs");
        std::memcpy(&header_, file.data(), sizeof(header_));
        if (std::memcmp(header_.magic, "OVD4MTP\0", 8) != 0 ||
            header_.version != 1 || header_.header_bytes != kFileHeaderBytes ||
            header_.tensor_entry_bytes != sizeof(TensorEntry) || header_.shared_format != 3 ||
            header_.stages != kDsparkStages || header_.block_size != kDsparkBlock ||
            header_.noise_token != kDsparkNoiseToken || header_.dimension != kDimension ||
            header_.moe_dimension != kMoeDimension || header_.experts != kExperts ||
            header_.top_k != kTopK || header_.hc != kHcMultiplicity ||
            header_.hc_iterations != 20 || header_.vocabulary != kVocabulary ||
            header_.markov_rank != kDsparkMarkovRank || header_.target0 != 40 ||
            header_.target1 != 41 || header_.target2 != 42 ||
            header_.expert_record_bytes != kExpertRecordBytes ||
            header_.expert_records != static_cast<uint64_t>(kDsparkStages) * kExperts ||
            header_.file_bytes != file.size() || header_.group_table_offset != sizeof(header_) ||
            header_.tensor_table_offset < kFileHeaderBytes ||
            header_.data_offset < header_.tensor_table_offset) {
            throw std::runtime_error("Unsupported DSpark shared container");
        }
        if (header_.group_count != kDsparkStages || header_.tensor_count == 0 ||
            header_.tensor_count > 1024 ||
            header_.tensor_table_offset + header_.tensor_count * sizeof(TensorEntry) >
                header_.data_offset) {
            throw std::runtime_error("Invalid DSpark metadata bounds");
        }
        stage_begin_.fill(UINT64_MAX);
        const auto* groups = reinterpret_cast<const GroupEntry*>(
            file.data() + header_.group_table_offset);
        const auto* entries = reinterpret_cast<const TensorEntry*>(
            file.data() + header_.tensor_table_offset);
        std::vector<bool> seen(static_cast<size_t>(header_.tensor_count));
        for (uint32_t ordinal = 0; ordinal < header_.group_count; ++ordinal) {
            const GroupEntry& group = groups[ordinal];
            if (group.kind != static_cast<int32_t>(GroupKind::mtp) || group.index < 0 ||
                static_cast<uint32_t>(group.index) >= kDsparkStages ||
                group.data_begin < header_.data_offset || group.data_end <= group.data_begin ||
                group.data_end > file.size() || (group.data_begin & 4095u) ||
                (group.data_end & 4095u) ||
                static_cast<uint64_t>(group.first_tensor) + group.tensor_count >
                    header_.tensor_count) {
                throw std::runtime_error("Invalid DSpark tensor group");
            }
            const uint32_t stage = static_cast<uint32_t>(group.index);
            if (stage_begin_[stage] != UINT64_MAX)
                throw std::runtime_error("Duplicate DSpark stage group");
            stage_begin_[stage] = group.data_begin;
            stage_end_[stage] = group.data_end;
            for (uint32_t local = 0; local < group.tensor_count; ++local) {
                const uint32_t index = group.first_tensor + local;
                if (seen[index]) throw std::runtime_error("Duplicate DSpark tensor entry");
                seen[index] = true;
                add(entries[index], stage, group.data_begin, group.data_end);
            }
        }
        if (std::find(seen.begin(), seen.end(), false) != seen.end())
            throw std::runtime_error("Ungrouped DSpark tensor");
        for (uint32_t stage = 0; stage < kDsparkStages; ++stage)
            if (stage_begin_[stage] == UINT64_MAX)
                throw std::runtime_error("Missing DSpark stage group");
    }

    const DsparkHeader& header() const { return header_; }
    const TensorView& require(const std::string& name) const {
        const auto found = tensors_.find(name);
        if (found == tensors_.end()) throw std::runtime_error("Missing DSpark tensor: " + name);
        return found->second;
    }

private:
    void add(const TensorEntry& entry, uint32_t stage, uint64_t begin, uint64_t end) {
        if (entry.rank > 8 || entry.data_offset < begin || entry.data_offset > end ||
            entry.data_bytes > end - entry.data_offset ||
            (entry.auxiliary_bytes && (entry.auxiliary_offset < begin ||
             entry.auxiliary_offset > end || entry.auxiliary_bytes > end - entry.auxiliary_offset)))
            throw std::runtime_error("Invalid DSpark tensor entry");
        TensorView view{};
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
        view.layer = stage;
        const std::string name = bounded_name(entry.name, sizeof(entry.name));
        if (!tensors_.emplace(name, view).second)
            throw std::runtime_error("Duplicate DSpark tensor: " + name);
    }

    const ReadOnlyMapping& file_;
    DsparkHeader header_{};
    std::unordered_map<std::string, TensorView> tensors_;
    std::array<uint64_t, kDsparkStages> stage_begin_{};
    std::array<uint64_t, kDsparkStages> stage_end_{};
};

class DsparkExpertIndex {
public:
    explicit DsparkExpertIndex(const ReadOnlyMapping& file) : file_(file) {
        if (file.size() < kFileHeaderBytes) throw std::runtime_error("Truncated DSpark experts");
        std::memcpy(&header_, file.data(), sizeof(header_));
        if (std::memcmp(header_.magic, "OVD4MEX\0", 8) != 0 || header_.version != 1 ||
            header_.header_bytes != kFileHeaderBytes || header_.stages != kDsparkStages ||
            header_.dimension != kDimension || header_.moe_dimension != kMoeDimension ||
            header_.experts != kExperts || header_.record_bytes != kExpertRecordBytes ||
            header_.data_offset != kFileHeaderBytes ||
            header_.total_records != static_cast<uint64_t>(kDsparkStages) * kExperts ||
            header_.file_bytes != file.size() ||
            header_.data_offset + header_.total_records * kExpertRecordBytes != file.size() ||
            header_.w1_weight_offset != kExpertW1Offset ||
            header_.w1_scale_offset != kExpertW1ScaleOffset ||
            header_.w3_weight_offset != kExpertW3Offset ||
            header_.w3_scale_offset != kExpertW3ScaleOffset ||
            header_.w2_weight_offset != kExpertW2Offset ||
            header_.w2_scale_offset != kExpertW2ScaleOffset) {
            throw std::runtime_error("Unsupported DSpark expert container");
        }
    }
    uint64_t record_offset(uint32_t stage, uint32_t expert) const {
        if (stage >= kDsparkStages || expert >= kExperts)
            throw std::runtime_error("Invalid DSpark expert key");
        return header_.data_offset +
            (static_cast<uint64_t>(stage) * kExperts + expert) * kExpertRecordBytes;
    }
    const uint8_t* mapped_record(uint32_t stage, uint32_t expert) const {
        return file_.data() + record_offset(stage, expert);
    }
    const std::string& path() const { return file_.path(); }
private:
    const ReadOnlyMapping& file_;
    DsparkExpertHeader header_{};
};

static Buffer upload_dspark_file(const Runtime& runtime, const ReadOnlyMapping& file) {
    Buffer destination = create_device_buffer(runtime, file.size());
    const VkDeviceSize chunk_bytes = std::min<VkDeviceSize>(file.size(), 64ull * 1024 * 1024);
    Buffer staging = create_host_buffer_uninitialized(runtime, chunk_bytes);
    FiniteQueue queue(runtime, runtime.queue);
    for (uint64_t offset = 0; offset < file.size(); offset += chunk_bytes) {
        const VkDeviceSize bytes = std::min<VkDeviceSize>(chunk_bytes, file.size() - offset);
        std::memcpy(staging.mapped, file.data() + offset, static_cast<size_t>(bytes));
        flush_buffer(runtime, staging);
        const uint64_t done = queue.submit([&](VkCommandBuffer command) {
            VkBufferCopy copy{0, offset, bytes};
            vkfn::CmdCopyBuffer(command, staging.handle, destination.handle, 1, &copy);
            transfer_barrier(command, destination);
        });
        queue.wait(done);
    }
    destroy_buffer(runtime, staging);
    return destination;
}

struct DsparkPipelines {
    VkPipeline main_mean = VK_NULL_HANDLE, embedding_batch = VK_NULL_HANDLE;
    VkPipeline embedding_indexed = VK_NULL_HANDLE, quantize = VK_NULL_HANDLE;
    VkPipeline quantize_strided_batch = VK_NULL_HANDLE;
    VkPipeline q8_batch = VK_NULL_HANDLE, q4_batch = VK_NULL_HANDLE;
    VkPipeline q8_residual_batch = VK_NULL_HANDLE, q4_residual_batch = VK_NULL_HANDLE;
    VkPipeline q8_grouped_batch = VK_NULL_HANDLE, q4_grouped_batch = VK_NULL_HANDLE;
    VkPipeline rmsnorm = VK_NULL_HANDLE, hc_mix = VK_NULL_HANDLE;
    VkPipeline hc_mix_scale = VK_NULL_HANDLE, hc_head_apply = VK_NULL_HANDLE;
    VkPipeline hc_sinkhorn = VK_NULL_HANDLE, hc_pre = VK_NULL_HANDLE;
    VkPipeline hc_post = VK_NULL_HANDLE, hc_head = VK_NULL_HANDLE;
    VkPipeline rope_batch = VK_NULL_HANDLE, attention_batch = VK_NULL_HANDLE;
    VkPipeline main_attention_batch = VK_NULL_HANDLE, compress_ratio4 = VK_NULL_HANDLE;
    VkPipeline router_batch = VK_NULL_HANDLE, router_hash_batch = VK_NULL_HANDLE;
    VkPipeline expert_gate = VK_NULL_HANDLE;
    VkPipeline expert_down = VK_NULL_HANDLE;
    VkPipeline expert_gate_q4 = VK_NULL_HANDLE, expert_down_q4 = VK_NULL_HANDLE;
    VkPipeline expert_gate_q4_swar = VK_NULL_HANDLE, expert_down_q4_swar = VK_NULL_HANDLE;
    VkPipeline expert_gate_q4_unique = VK_NULL_HANDLE;
    VkPipeline expert_down_q4_unique = VK_NULL_HANDLE;
    VkPipeline swiglu_batch = VK_NULL_HANDLE;
    VkPipeline reduce_batch = VK_NULL_HANDLE, logits_bias = VK_NULL_HANDLE;
    VkPipeline greedy_argmax = VK_NULL_HANDLE;
};

class DsparkKernels {
public:
    DsparkKernels(const Runtime& runtime, const std::filesystem::path& directory)
        : runtime_(runtime), resources_(create_compute_resources(runtime, 8192)),
          dummy_(create_device_buffer(runtime, 1024)) {
        const auto load = [&](const char* name) {
            return create_dsv4_pipeline(runtime_, resources_,
                directory / (std::string(name) + ".comp.spv"), 64u);
        };
        p_.main_mean = load("dsv4_dspark_main_mean");
        p_.embedding_batch = load("dsv4_dspark_embedding_batch");
        p_.embedding_indexed = load("dsv4_dspark_embedding_indexed");
        p_.quantize = load("dsv4_quantize_q8");
        p_.quantize_strided_batch = load("dsv4_quantize_q8_strided_batch");
        p_.q8_batch = load("dsv4_q8_gemv_batch");
        p_.q4_batch = load("dsv4_q4g64t_gemv_batch");
        p_.q8_residual_batch = load("dsv4_q8_gemv_residual_batch");
        p_.q4_residual_batch = load("dsv4_q4g64t_gemv_residual_batch");
        p_.q8_grouped_batch = load("dsv4_q8_grouped_gemv_batch");
        p_.q4_grouped_batch = load("dsv4_q4g64t_grouped_gemv_batch");
        p_.rmsnorm = load("dsv4_rmsnorm");
        p_.hc_mix = load("dsv4_hc_mix");
        p_.hc_mix_scale = load("dsv4_dspark_hc_mix_scale");
        p_.hc_head_apply = load("dsv4_dspark_hc_head_apply");
        p_.hc_sinkhorn = load("dsv4_hc_sinkhorn");
        p_.hc_pre = load("dsv4_hc_pre");
        p_.hc_post = load("dsv4_hc_post");
        p_.hc_head = load("dsv4_hc_head");
        p_.rope_batch = load("dsv4_partial_rope_batch");
        p_.attention_batch = load("dsv4_dspark_attention_batch");
        p_.main_attention_batch = load("dsv4_main_attention_batch");
        p_.compress_ratio4 = load("dsv4_compress_ratio4");
        p_.router_batch = load("dsv4_router_top6_batch");
        p_.router_hash_batch = load("dsv4_router_top6_hash_batch");
        p_.expert_gate = load("dsv4_expert_gate_up_fp4");
        p_.expert_down = load("dsv4_expert_down_fp4");
        p_.expert_gate_q4 = load("dsv4_expert_gate_up_q4g64t");
        p_.expert_down_q4 = load("dsv4_expert_down_q4g64t");
        p_.expert_gate_q4_swar = load("dsv4_expert_gate_up_q4g64t_swar");
        p_.expert_down_q4_swar = load("dsv4_expert_down_q4g64t_swar");
        p_.expert_gate_q4_unique = load("dsv4_expert_gate_up_q4g64t_unique_batch");
        p_.expert_down_q4_unique = load("dsv4_expert_down_q4g64t_unique_batch");
        p_.swiglu_batch = load("dsv4_swiglu_batch");
        p_.reduce_batch = load("dsv4_reduce_experts_batch");
        p_.logits_bias = load("dsv4_dspark_logits_bias");
        p_.greedy_argmax = load("dsv4_greedy_argmax");
    }
    ~DsparkKernels() {
        for (VkPipeline pipeline : resources_.pipelines)
            vkfn::DestroyPipeline(runtime_.device, pipeline, nullptr);
        for (VkShaderModule module : resources_.shader_modules)
            vkfn::DestroyShaderModule(runtime_.device, module, nullptr);
        if (resources_.descriptor_pool)
            vkfn::DestroyDescriptorPool(runtime_.device, resources_.descriptor_pool, nullptr);
        if (resources_.pipeline_layout)
            vkfn::DestroyPipelineLayout(runtime_.device, resources_.pipeline_layout, nullptr);
        if (resources_.descriptor_layout)
            vkfn::DestroyDescriptorSetLayout(runtime_.device, resources_.descriptor_layout, nullptr);
        destroy_buffer(runtime_, dummy_);
    }
    VkDescriptorSet set(std::initializer_list<DescriptorRange> active) const {
        return create_dsv4_set(runtime_, resources_, dsv4_bindings(whole(dummy_), active));
    }
    VkDescriptorSet set(const std::array<DescriptorRange, 6>& ranges) const {
        return create_dsv4_set(runtime_, resources_, ranges);
    }
    void update(VkDescriptorSet set, uint32_t binding, const DescriptorRange& range) const {
        if (!set || binding >= 6 || !range.buffer || !range.range)
            throw std::runtime_error("Invalid DSpark descriptor update");
        VkDescriptorBufferInfo info{range.buffer, range.offset, range.range};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set; write.dstBinding = binding; write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; write.pBufferInfo = &info;
        vkfn::UpdateDescriptorSets(runtime_.device, 1, &write, 0, nullptr);
    }
    void dispatch(VkCommandBuffer command, VkPipeline pipeline, VkDescriptorSet set,
                  const void* push, uint32_t x, uint32_t y = 1) const {
        dispatch_dsv4(command, resources_, pipeline, set, push, x, y);
    }
    const DsparkPipelines& p() const { return p_; }
    DescriptorRange dummy() const { return whole(dummy_); }
private:
    const Runtime& runtime_;
    ComputeResources resources_{};
    Buffer dummy_{};
    DsparkPipelines p_{};
};

class DsparkExpertCache {
public:
    struct Resolution {
        std::array<uint32_t, kTopK> slots{};
        uint64_t ready = 0;
        uint32_t misses = 0;
    };

    DsparkExpertCache(const Runtime& runtime, const DsparkExpertIndex& index)
        : runtime_(runtime), index_(index), transfer_(runtime, runtime.secondary_queue) {
        // Keep the exact 36-record policy and footprint, but avoid requiring a
        // single 459-MiB WDDM allocation.  Stages never share records, so three
        // bounded 153-MiB arenas are the natural physical representation.
        for (Buffer& arena : arenas_)
            arena = create_device_buffer(runtime_,
                static_cast<VkDeviceSize>(kDsparkCacheSlots) * kExpertRecordBytes);
        for (Buffer& buffer : staging_)
            buffer = create_host_buffer_uninitialized(runtime_, kExpertRecordBytes);
        for (auto& locations : location_) locations.fill(UINT32_MAX);
        direct_io_ = std::getenv("DSV4_RAM_GIB") != nullptr;
        direct_files_.fill(INVALID_HANDLE_VALUE);
        read_events_.fill(nullptr);
        if (direct_io_) {
            for (uint32_t worker = 0; worker < kDsparkIoWorkers; ++worker) {
                direct_files_[worker] = CreateFileA(index_.path().c_str(), GENERIC_READ,
                    FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                    FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED,
                    nullptr);
                if (direct_files_[worker] == INVALID_HANDLE_VALUE)
                    throw std::runtime_error("Could not open budgeted DSpark expert stream");
                read_events_[worker] = CreateEventA(nullptr, FALSE, FALSE, nullptr);
                if (!read_events_[worker])
                    throw std::runtime_error("Could not create budgeted DSpark read event");
            }
        }
    }
    ~DsparkExpertCache() {
        for (HANDLE event : read_events_) if (event) CloseHandle(event);
        for (HANDLE file : direct_files_)
            if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        for (Buffer& arena : arenas_) destroy_buffer(runtime_, arena);
    }

    Resolution resolve_token(uint32_t stage, const uint32_t* route_words) {
        const auto started = std::chrono::steady_clock::now();
        if (stage >= kDsparkStages) throw std::runtime_error("Invalid DSpark cache stage");
        Resolution result;
        std::array<bool, kDsparkCacheSlots> reserved{};
        std::map<uint32_t, uint32_t> selected;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint32_t expert = route_words[rank];
            if (expert >= kExperts) throw std::runtime_error("DSpark router selected invalid expert");
            ++frequency_[stage][expert];
            const uint32_t slot = location_[stage][expert];
            if (slot != UINT32_MAX) {
                reserved[slot] = true;
                entries_[stage][slot].age = ++clock_;
                selected.emplace(expert, slot);
            }
        }
        struct Miss { uint32_t expert, slot, staging; };
        std::vector<Miss> misses;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const uint32_t expert = route_words[rank];
            auto found = selected.find(expert);
            uint32_t slot = UINT32_MAX;
            if (found != selected.end()) {
                slot = found->second;
                ++hits_;
            } else {
                uint32_t least = UINT32_MAX;
                uint64_t oldest = UINT64_MAX;
                for (uint32_t candidate = 0; candidate < kDsparkCacheSlots; ++candidate) {
                    if (reserved[candidate]) continue;
                    const Entry& entry = entries_[stage][candidate];
                    const uint32_t uses = entry.expert < 0 ? 0u :
                        frequency_[stage][static_cast<uint32_t>(entry.expert)];
                    if (uses < least || (uses == least && entry.age < oldest)) {
                        slot = candidate; least = uses; oldest = entry.age;
                    }
                }
                if (slot == UINT32_MAX) throw std::runtime_error("DSpark cache has no finite victim");
                Entry& victim = entries_[stage][slot];
                if (victim.expert >= 0)
                    location_[stage][static_cast<uint32_t>(victim.expert)] = UINT32_MAX;
                victim.stage = static_cast<int32_t>(stage);
                victim.expert = static_cast<int32_t>(expert);
                victim.age = ++clock_;
                location_[stage][expert] = slot;
                reserved[slot] = true;
                selected.emplace(expert, slot);
                misses.push_back({expert, slot, static_cast<uint32_t>(misses.size())});
                ++misses_;
            }
            result.slots[rank] = slot;
        }
        result.misses = static_cast<uint32_t>(misses.size());
        if (!misses.empty()) {
            const uint32_t workers = std::min<uint32_t>(
                kDsparkIoWorkers, static_cast<uint32_t>(misses.size()));
            std::atomic<DWORD> read_error{ERROR_SUCCESS};
            std::vector<std::thread> threads;
            for (uint32_t worker = 0; worker < workers; ++worker) {
                threads.emplace_back([&, worker] {
                    for (uint32_t i = worker; i < misses.size(); i += workers) {
                        const Miss& miss = misses[i];
                        if (!direct_io_) {
                            std::memcpy(staging_[miss.staging].mapped,
                                index_.mapped_record(stage, miss.expert), kExpertRecordBytes);
                            continue;
                        }
                        const uint64_t offset = index_.record_offset(stage, miss.expert);
                        if ((offset & 4095u) != 0u || (kExpertRecordBytes & 4095u) != 0u ||
                            (reinterpret_cast<uintptr_t>(staging_[miss.staging].mapped) &
                             4095u) != 0u) {
                            read_error.store(ERROR_INVALID_PARAMETER);
                            continue;
                        }
                        OVERLAPPED operation{};
                        operation.Offset = static_cast<DWORD>(offset);
                        operation.OffsetHigh = static_cast<DWORD>(offset >> 32u);
                        operation.hEvent = read_events_[worker];
                        ResetEvent(operation.hEvent);
                        const BOOL started = ReadFile(direct_files_[worker],
                            staging_[miss.staging].mapped,
                            static_cast<DWORD>(kExpertRecordBytes), nullptr, &operation);
                        const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
                        DWORD transferred = 0;
                        bool ok = false;
                        if (started || start_error == ERROR_IO_PENDING) {
                            const DWORD waited = WaitForSingleObject(operation.hEvent, 10000u);
                            if (waited == WAIT_OBJECT_0) {
                                ok = GetOverlappedResult(direct_files_[worker], &operation,
                                    &transferred, FALSE) != FALSE;
                            } else {
                                CancelIoEx(direct_files_[worker], &operation);
                            }
                        }
                        if (!ok || transferred != kExpertRecordBytes) {
                            DWORD error = GetLastError();
                            if (error == ERROR_SUCCESS) error = ERROR_READ_FAULT;
                            read_error.store(error);
                        }
                    }
                });
            }
            for (std::thread& thread : threads) thread.join();
            if (read_error.load() != ERROR_SUCCESS)
                throw std::runtime_error("Budgeted DSpark expert read failed (Win32 " +
                    std::to_string(read_error.load()) + ")");
            for (const Miss& miss : misses) flush_buffer(runtime_, staging_[miss.staging]);
            result.ready = transfer_.submit([&](VkCommandBuffer command) {
                for (const Miss& miss : misses) {
                    VkBufferCopy copy{0,
                        static_cast<VkDeviceSize>(miss.slot) * kExpertRecordBytes,
                        kExpertRecordBytes};
                    vkfn::CmdCopyBuffer(command, staging_[miss.staging].handle,
                        arenas_[stage].handle, 1, &copy);
                }
                transfer_barrier(command, arenas_[stage]);
            });
            transfer_bytes_ += static_cast<uint64_t>(misses.size()) * kExpertRecordBytes;
        }
        acquisition_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }
    DescriptorRange record(uint32_t stage, uint32_t slot) const {
        if (stage >= kDsparkStages || slot >= kDsparkCacheSlots)
            throw std::runtime_error("Invalid DSpark device expert slot");
        return arena_range(arenas_[stage],
            static_cast<VkDeviceSize>(slot) * kExpertRecordBytes,
            kExpertRecordBytes);
    }
    VkSemaphore semaphore() const { return transfer_.semaphore(); }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
    double acquisition_seconds() const { return acquisition_seconds_; }
private:
    static constexpr uint32_t kDsparkIoWorkers = 3;
    struct Entry { int32_t stage = -1, expert = -1; uint64_t age = 0; };
    const Runtime& runtime_;
    const DsparkExpertIndex& index_;
    FiniteQueue transfer_;
    std::array<Buffer, kDsparkStages> arenas_{};
    std::array<Buffer, kTopK> staging_{};
    std::array<HANDLE, kDsparkIoWorkers> direct_files_{};
    std::array<HANDLE, kDsparkIoWorkers> read_events_{};
    std::array<std::array<Entry, kDsparkCacheSlots>, kDsparkStages> entries_{};
    std::array<std::array<uint32_t, kExperts>, kDsparkStages> location_{};
    std::array<std::array<uint32_t, kExperts>, kDsparkStages> frequency_{};
    uint64_t clock_ = 0, hits_ = 0, misses_ = 0, transfer_bytes_ = 0;
    double acquisition_seconds_ = 0.0;
    bool direct_io_ = false;
};

class DsparkProgram {
public:
    DsparkProgram(const Runtime& runtime, ExecutorScaffold& main_executor,
                  const SharedIndex& main_index, const DsparkIndex& index,
                  const ReadOnlyMapping& shared_file,
                  const DsparkExpertIndex& expert_index,
                  const std::filesystem::path& shader_directory)
        : runtime_(runtime), main_executor_(main_executor), main_index_(main_index),
          index_(index), shared_file_(shared_file), compute_(runtime, runtime.queue),
          kernels_(runtime, shader_directory) {
        shared_arena_ = upload_dspark_file(runtime_, shared_file_);
        hidden_ = device(floats(kDsparkBlock * kHcMultiplicity * kDimension));
        hidden_alt_ = device(floats(kDsparkBlock * kHcMultiplicity * kDimension));
        reduced_ = device(floats(kDsparkBlock * kDimension));
        normalized_ = device(floats(kDsparkBlock * kDimension));
        main_concat_ = device(floats(3u * kDimension));
        main_x_ = device(floats(kDimension));
        main_kv_raw_ = device(floats(kHeadDimension));
        main_kv_ = device(floats(kHeadDimension));
        q_rank_ = device(floats(kDsparkBlock * 1024u));
        q_rank_norm_ = device(floats(kDsparkBlock * 1024u));
        query_ = device(floats(kDsparkBlock * kHeads * kHeadDimension));
        kv_raw_ = device(floats(kDsparkBlock * kHeadDimension));
        kv_ = device(floats(kDsparkBlock * kHeadDimension));
        context_ = device(floats(kDsparkBlock * kHeads * kHeadDimension));
        o_rank_ = device(floats(kDsparkBlock * 8192u));
        router_logits_ = device(floats(kDsparkBlock * kExperts));
        routing_ = host(5u * 64u * sizeof(uint32_t));
        ffn_ = device(floats(kDsparkBlock * kTopK * kMoeDimension));
        accumulator_ = device(floats(kDsparkBlock * kTopK * kDimension));
        routed_reduced_ = device(floats(kDsparkBlock * kDimension));
        dense_gate_ = device(floats(kDsparkBlock * kMoeDimension));
        dense_up_ = device(floats(kDsparkBlock * kMoeDimension));
        shared_intermediate_ = device(floats(kDsparkBlock * kMoeDimension));
        const uint32_t max_packed = kDsparkBlock * kHeads * kHeadDimension / 4u;
        const uint32_t max_scales = kDsparkBlock * kHeads * kHeadDimension / 128u;
        quantized_ = device(static_cast<VkDeviceSize>(max_packed + max_scales) * 4u);
        expert_quantized_ = device(static_cast<VkDeviceSize>(kDsparkBlock) *
                                   (1024u + 128u) * 4u);
        intermediate_quantized_ = device(static_cast<VkDeviceSize>(kDsparkBlock) *
            kTopK * (512u + 64u) * 4u);
        kv_cache_ = device(floats(kDsparkStages * kDsparkKvSlots * kHeadDimension));
        indices_ = host(160u * sizeof(uint32_t));
        draft_input_ids_ = host(kDsparkBlock * sizeof(uint32_t));
        output_ids_ = host((kDsparkBlock + 1u) * 4u * sizeof(uint32_t));
        ones_ = host(1024u * sizeof(float));
        rope_cos_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        rope_sin_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        hc_mixes_ = device(floats(kDsparkBlock * 24u));
        hc_split_ = device(floats(kDsparkBlock * 24u));
        hc_head_mixes_ = device(floats(kDsparkBlock * kHcMultiplicity));
        head_hidden_ = device(floats(kDsparkBlock * kDimension));
        head_norm_ = device(floats(kDsparkBlock * kDimension));
        logits_ = device(floats(kDsparkBlock * kVocabulary));
        markov_embed_ = device(floats(kDsparkMarkovRank));
        markov_quantized_ = device((64u + 2u) * sizeof(uint32_t));
        markov_bias_ = device(floats(kVocabulary));
        combined_logits_ = device(floats(kVocabulary));
        argmax_workspace_ = device(256u * 2u * sizeof(uint32_t));
        hc_head_params_ = host(5u * sizeof(float));
        initialize_constants();
        build_sets();
        cache_ = std::make_unique<DsparkExpertCache>(runtime_, expert_index);
    }

    ~DsparkProgram() {
        cache_.reset();
        for (Buffer* buffer : owned_buffers()) destroy_buffer(runtime_, *buffer);
    }

    void prefill_main(uint32_t position) {
        if (position >= kShortContext) throw std::runtime_error("DSpark prefill position overflow");
        const uint64_t done = compute_.submit([&](VkCommandBuffer command) {
            record_main_x(command);
            for (uint32_t stage = 0; stage < kDsparkStages; ++stage)
                record_main_kv(command, stage, position);
        });
        compute_.wait(done);
    }

    void prefill_captured(const Buffer& captures, uint32_t token, uint32_t position) {
        if (token >= kDsparkBlock || position >= kShortContext)
            throw std::runtime_error("Invalid captured DSpark prefill");
        const VkDeviceSize block = floats(kHcMultiplicity * kDimension);
        const uint64_t done = compute_.submit([&](VkCommandBuffer command) {
            for (uint32_t target = 0; target < 3u; ++target)
                copy_compute_result(command, captures,
                    static_cast<VkDeviceSize>(token * 3u + target) * block,
                    main_executor_.main_targets(), static_cast<VkDeviceSize>(target) * block,
                    block);
            record_main_x(command);
            for (uint32_t stage = 0; stage < kDsparkStages; ++stage)
                record_main_kv(command, stage, position);
        });
        compute_.wait(done);
    }

    std::array<uint32_t, kDsparkBlock + 1u> draft(uint32_t known_token,
                                                   uint32_t position) {
        if (position == 0 || position + kDsparkBlock >= kShortContext + kDsparkBlock)
            throw std::runtime_error("DSpark draft position is outside bounded context");
        const auto started = std::chrono::steady_clock::now();
        auto* input = static_cast<uint32_t*>(draft_input_ids_.mapped);
        input[0] = known_token;
        for (uint32_t i = 1; i < kDsparkBlock; ++i) input[i] = kDsparkNoiseToken;
        flush_buffer(runtime_, draft_input_ids_);
        auto* output = static_cast<uint32_t*>(output_ids_.mapped);
        std::fill(output, output + (kDsparkBlock + 1u) * 4u, 0u);
        output[0] = known_token;
        flush_buffer(runtime_, output_ids_);
        update_indices(position);

        for (uint32_t stage = 0; stage < kDsparkStages; ++stage) {
            const auto pre_start = std::chrono::steady_clock::now();
            const uint64_t pre = compute_.submit([&](VkCommandBuffer command) {
                if (stage == 0u) {
                    record_main_x(command);
                    record_embedding(command);
                }
                record_stage_pre(command, stage, position);
            });
            compute_.wait(pre);
            draft_gpu_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pre_start).count();
            invalidate_buffer(runtime_, routing_);
            const auto* routes = static_cast<const uint32_t*>(routing_.mapped);
            for (uint32_t token = 0; token < kDsparkBlock; ++token) {
                const auto resolved = cache_->resolve_token(stage, routes + token * 64u);
                current_slots_[token] = resolved.slots;
                update_expert_sets(stage, token);
                const auto post_start = std::chrono::steady_clock::now();
                const uint64_t post = compute_.submit([&](VkCommandBuffer command) {
                    record_stage_expert(command, stage, token);
                }, resolved.ready ? cache_->semaphore() : VK_NULL_HANDLE, resolved.ready);
                compute_.wait(post);
                draft_gpu_seconds_ += std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - post_start).count();
            }
            const auto shared_start = std::chrono::steady_clock::now();
            const uint64_t shared = compute_.submit([&](VkCommandBuffer command) {
                record_stage_shared(command, stage);
            });
            compute_.wait(shared);
            draft_gpu_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - shared_start).count();
        }
        const auto head_start = std::chrono::steady_clock::now();
        const uint64_t head = compute_.submit([&](VkCommandBuffer command) {
            record_head(command);
        });
        compute_.wait(head);
        draft_gpu_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - head_start).count();
        invalidate_buffer(runtime_, output_ids_);
        std::array<uint32_t, kDsparkBlock + 1u> result{};
        output = static_cast<uint32_t*>(output_ids_.mapped);
        for (uint32_t i = 0; i <= kDsparkBlock; ++i) {
            result[i] = output[i * 4u];
            if (result[i] >= kVocabulary) throw std::runtime_error("DSpark emitted invalid token");
        }
        ++draft_passes_;
        draft_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }

    double draft_seconds() const { return draft_seconds_; }
    double draft_gpu_seconds() const { return draft_gpu_seconds_; }
    uint64_t draft_passes() const { return draft_passes_; }
    const DsparkExpertCache& cache() const { return *cache_; }

private:
    struct LinearSet { VkDescriptorSet set = VK_NULL_HANDLE; TensorFormat format{}; };
    struct StageSets {
        VkDescriptorSet quant_hc_attn{}, hc_attn_mix_scale{}, hc_attn_split{},
            hc_attn_pre{}, attn_norm{}, quant_normalized{};
        LinearSet hc_attn_linear{};
        LinearSet wq_a{}, wq_b{}, wkv{}, wkv_main{}, wo_a{}, wo_b{};
        VkDescriptorSet q_norm{}, quant_q_rank{}, q_head_norm{}, kv_norm{}, kv_main_norm{};
        VkDescriptorSet query_rope{}, inverse_rope{}, kv_rope{}, main_kv_rope{};
        VkDescriptorSet attention{}, quant_context{}, quant_o_rank{}, hc_attn_post{};
        VkDescriptorSet quant_hc_ffn{}, hc_ffn_mix_scale{}, hc_ffn_split{},
            hc_ffn_pre{}, ffn_norm{};
        LinearSet hc_ffn_linear{};
        LinearSet router{}, shared_w1{}, shared_w3{}, shared_w2{};
        VkDescriptorSet router_select{}, shared_swiglu{}, quant_shared{}, hc_ffn_post{};
        std::array<VkDescriptorSet, kDsparkBlock> reduce{};
        std::array<VkDescriptorSet, kDsparkBlock> quant_expert{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> quant_intermediate{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> expert_gate{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> expert_down{};
    };

    static VkDeviceSize floats(uint64_t count) { return count * sizeof(float); }
    Buffer device(VkDeviceSize bytes) { return create_device_buffer(runtime_, bytes); }
    Buffer host(VkDeviceSize bytes) { return create_buffer(runtime_, bytes); }

    std::vector<Buffer*> owned_buffers() {
        return {&argmax_workspace_, &combined_logits_, &markov_bias_, &markov_quantized_,
            &markov_embed_, &logits_, &head_norm_, &head_hidden_, &hc_head_params_,
            &hc_head_mixes_, &hc_split_, &hc_mixes_, &rope_sin_, &rope_cos_, &ones_, &output_ids_,
            &draft_input_ids_, &indices_, &kv_cache_, &intermediate_quantized_,
            &expert_quantized_, &quantized_, &shared_intermediate_, &dense_up_,
            &dense_gate_, &routed_reduced_, &accumulator_, &ffn_, &routing_, &router_logits_, &o_rank_,
            &context_, &kv_, &kv_raw_, &query_, &q_rank_norm_, &q_rank_, &main_kv_,
            &main_kv_raw_, &main_x_, &main_concat_, &normalized_, &reduced_,
            &hidden_alt_, &hidden_, &shared_arena_};
    }

    VkDescriptorSet set(std::initializer_list<DescriptorRange> ranges) {
        return kernels_.set(ranges);
    }
    DescriptorRange data(const TensorView& tensor) const {
        return tensor_data_range(shared_arena_, 0, tensor);
    }
    DescriptorRange auxiliary(const TensorView& tensor) const {
        return tensor_auxiliary_range(shared_arena_, 0, tensor);
    }
    LinearSet linear(const DescriptorRange& activation, const TensorView& tensor,
                     const DescriptorRange& output, const DescriptorRange* residual = nullptr) {
        if (tensor.format != TensorFormat::q8_row && tensor.format != TensorFormat::q4g64t)
            throw std::runtime_error("DSpark matrix is not Q8/Q4");
        std::array<DescriptorRange, 6> ranges{};
        ranges.fill(kernels_.dummy());
        ranges[0] = activation; ranges[1] = data(tensor); ranges[2] = auxiliary(tensor);
        ranges[3] = output; if (residual) ranges[4] = *residual;
        return {kernels_.set(ranges), tensor.format};
    }

    static uint32_t packed_words(uint32_t count) { return divide_up(count, 4u); }
    static uint32_t scale_words(uint32_t count, uint32_t group) {
        return divide_up(count, group);
    }
    uint32_t rms_bits() const { return float_word(main_index_.header().rms_epsilon); }
    uint32_t swiglu_bits() const { return float_word(main_index_.header().swiglu_limit); }

    void initialize_constants() {
        auto* ones = static_cast<float*>(ones_.mapped);
        std::fill(ones, ones + 1024u, 1.0f);
        flush_buffer(runtime_, ones_);
        auto* cosine = static_cast<float*>(rope_cos_.mapped);
        auto* sine = static_cast<float*>(rope_sin_.mapped);
        for (uint32_t position = 0; position <= kShortContext + kDsparkBlock; ++position) {
            for (uint32_t frequency = 0; frequency < 32u; ++frequency) {
                const float inverse = 1.0f / std::pow(main_index_.header().rope_theta,
                    static_cast<float>(frequency * 2u) / kRopeDimension);
                const float angle = static_cast<float>(position) * inverse;
                cosine[position * 32u + frequency] = std::cos(angle);
                sine[position * 32u + frequency] = std::sin(angle);
            }
        }
        flush_buffer(runtime_, rope_cos_);
        flush_buffer(runtime_, rope_sin_);
        const TensorView& scale = index_.require("mtp.2.hc_head_scale");
        const TensorView& base = index_.require("mtp.2.hc_head_base");
        if (scale.format != TensorFormat::f32 || scale.bytes != sizeof(float) ||
            base.format != TensorFormat::f32 || base.bytes != 4u * sizeof(float))
            throw std::runtime_error("Invalid DSpark HC-head parameters");
        auto* parameters = static_cast<float*>(hc_head_params_.mapped);
        std::memcpy(parameters, shared_file_.data() + scale.offset, sizeof(float));
        std::memcpy(parameters + 1, shared_file_.data() + base.offset, 4u * sizeof(float));
        flush_buffer(runtime_, hc_head_params_);
    }

    void require_matrix(const TensorView& tensor, uint32_t rows, uint32_t inner,
                        const char* name) const {
        require_linear_matrix(tensor, rows, inner, name);
        if (tensor.format != TensorFormat::q8_row && tensor.format != TensorFormat::q4g64t)
            throw std::runtime_error(std::string("Unsupported DSpark matrix format: ") + name);
    }

    void build_sets() {
        const auto whole_main_global = [&](const TensorView& tensor, bool auxiliary_range) {
            const Buffer& arena = main_executor_.global_arena();
            const uint64_t base = main_executor_.global_file_base();
            return auxiliary_range ? tensor_auxiliary_range(arena, base, tensor) :
                                     tensor_data_range(arena, base, tensor);
        };
        main_mean_set_ = set({whole(main_executor_.main_targets()), whole(main_concat_)});
        quant_main_concat_ = set({whole(main_concat_), whole(quantized_)});
        const TensorView& main_proj = index_.require("mtp.0.main_proj.weight");
        require_matrix(main_proj, kDimension, 3u * kDimension, "DSpark main projection");
        main_proj_ = linear(whole(quantized_), main_proj, whole(main_x_));
        main_norm_set_ = set({whole(main_x_), data(index_.require("mtp.0.main_norm.weight")),
                              whole(main_x_)});
        quant_main_x_ = set({whole(main_x_), whole(quantized_)});

        const TensorView& embedding = main_index_.require("embed.weight");
        require_matrix(embedding, kVocabulary, kDimension, "DSpark shared embedding");
        embedding_set_ = set({whole_main_global(embedding, false),
                              whole_main_global(embedding, true),
                              whole(draft_input_ids_), whole(hidden_)});
        const TensorView& head = main_index_.require("head.weight");
        require_matrix(head, kVocabulary, kDimension, "DSpark shared head");
        {
            std::array<DescriptorRange, 6> ranges{};
            ranges.fill(kernels_.dummy());
            ranges[0] = whole(quantized_);
            ranges[1] = whole_main_global(head, false);
            ranges[2] = whole_main_global(head, true);
            ranges[3] = whole(logits_);
            head_ = {kernels_.set(ranges), head.format};
        }

        for (uint32_t stage = 0; stage < kDsparkStages; ++stage) {
            StageSets& sets = stages_[stage];
            const std::string prefix = "mtp." + std::to_string(stage) + ".";
            const auto tensor = [&](const char* suffix) -> const TensorView& {
                return index_.require(prefix + suffix);
            };
            const auto matrix = [&](const char* suffix, uint32_t rows, uint32_t inner) ->
                    const TensorView& {
                const TensorView& result = tensor(suffix);
                require_matrix(result, rows, inner, suffix);
                return result;
            };

            sets.quant_hc_attn = set({whole(hidden_), whole(quantized_)});
            sets.hc_attn_linear = linear(whole(quantized_),
                matrix("hc_attn_fn", 24u, kHcMultiplicity * kDimension), whole(hc_mixes_));
            sets.hc_attn_mix_scale = set({whole(hidden_), whole(hc_mixes_)});
            sets.hc_attn_split = set({whole(hc_mixes_), data(tensor("hc_attn_scale")),
                data(tensor("hc_attn_base")), whole(hc_split_)});
            sets.hc_attn_pre = set({whole(hidden_), whole(hc_split_), whole(reduced_)});
            sets.attn_norm = set({whole(reduced_), data(tensor("attn_norm.weight")),
                                  whole(normalized_)});
            sets.quant_normalized = set({whole(normalized_), whole(quantized_)});
            sets.wq_a = linear(whole(quantized_),
                matrix("attn.wq_a.weight", 1024u, kDimension), whole(q_rank_));
            sets.wkv = linear(whole(quantized_),
                matrix("attn.wkv.weight", kHeadDimension, kDimension), whole(kv_raw_));
            sets.wkv_main = linear(whole(quantized_),
                matrix("attn.wkv.weight", kHeadDimension, kDimension), whole(main_kv_raw_));
            sets.q_norm = set({whole(q_rank_), data(tensor("attn.q_norm.weight")),
                               whole(q_rank_norm_)});
            sets.quant_q_rank = set({whole(q_rank_norm_), whole(quantized_)});
            sets.wq_b = linear(whole(quantized_),
                matrix("attn.wq_b.weight", kHeads * kHeadDimension, 1024u), whole(query_));
            sets.q_head_norm = set({whole(query_), whole(ones_), whole(context_)});
            sets.query_rope = set({whole(context_), whole(rope_cos_), whole(rope_sin_), whole(query_)});
            sets.kv_norm = set({whole(kv_raw_), data(tensor("attn.kv_norm.weight")), whole(kv_)});
            sets.kv_rope = set({whole(kv_), whole(rope_cos_), whole(rope_sin_), whole(kv_raw_)});
            sets.kv_main_norm = set({whole(main_kv_raw_), data(tensor("attn.kv_norm.weight")),
                                     whole(main_kv_)});
            sets.main_kv_rope = set({whole(main_kv_), whole(rope_cos_), whole(rope_sin_),
                                     whole(main_kv_raw_)});
            const VkDeviceSize kv_stage_bytes = floats(kDsparkKvSlots * kHeadDimension);
            sets.attention = set({whole(query_), arena_range(kv_cache_,
                static_cast<VkDeviceSize>(stage) * kv_stage_bytes, kv_stage_bytes),
                whole(indices_), data(tensor("attn.attn_sink")), whole(context_)});
            sets.inverse_rope = set({whole(context_), whole(rope_cos_), whole(rope_sin_), whole(query_)});
            sets.quant_context = set({whole(query_), whole(quantized_)});
            sets.wo_a = linear(whole(quantized_),
                matrix("attn.wo_a.weight", 8192u, 4096u), whole(o_rank_));
            sets.quant_o_rank = set({whole(o_rank_), whole(quantized_)});
            sets.wo_b = linear(whole(quantized_),
                matrix("attn.wo_b.weight", kDimension, 8192u), whole(reduced_));
            sets.hc_attn_post = set({whole(reduced_), whole(hidden_), whole(hc_split_), whole(hidden_alt_)});

            sets.quant_hc_ffn = set({whole(hidden_alt_), whole(quantized_)});
            sets.hc_ffn_linear = linear(whole(quantized_),
                matrix("hc_ffn_fn", 24u, kHcMultiplicity * kDimension), whole(hc_mixes_));
            sets.hc_ffn_mix_scale = set({whole(hidden_alt_), whole(hc_mixes_)});
            sets.hc_ffn_split = set({whole(hc_mixes_), data(tensor("hc_ffn_scale")),
                data(tensor("hc_ffn_base")), whole(hc_split_)});
            sets.hc_ffn_pre = set({whole(hidden_alt_), whole(hc_split_), whole(reduced_)});
            sets.ffn_norm = set({whole(reduced_), data(tensor("ffn_norm.weight")), whole(normalized_)});
            sets.router = linear(whole(quantized_),
                matrix("ffn.gate.weight", kExperts, kDimension), whole(router_logits_));
            sets.router_select = set({whole(router_logits_), data(tensor("ffn.gate.bias")),
                                      kernels_.dummy(), whole(routing_)});

            for (uint32_t token = 0; token < kDsparkBlock; ++token) {
                const DescriptorRange normalized = arena_range(normalized_,
                    floats(static_cast<uint64_t>(token) * kDimension), floats(kDimension));
                const DescriptorRange expert_q = arena_range(expert_quantized_,
                    static_cast<VkDeviceSize>(token) * (1024u + 128u) * sizeof(uint32_t),
                    (1024u + 128u) * sizeof(uint32_t));
                sets.quant_expert[token] = set({normalized, expert_q});
                const DescriptorRange route = arena_range(routing_, token * 64u * sizeof(uint32_t),
                                                           64u * sizeof(uint32_t));
                const DescriptorRange token_ffn = arena_range(ffn_,
                    floats(static_cast<uint64_t>(token) * kTopK * kMoeDimension),
                    floats(kTopK * kMoeDimension));
                const DescriptorRange token_accumulator = arena_range(accumulator_,
                    floats(static_cast<uint64_t>(token) * kTopK * kDimension),
                    floats(kTopK * kDimension));
                for (uint32_t rank = 0; rank < kTopK; ++rank) {
                    sets.expert_gate[token][rank] = set({expert_q, kernels_.dummy(), route, token_ffn});
                    const DescriptorRange intermediate = arena_range(intermediate_quantized_,
                        static_cast<VkDeviceSize>(token * kTopK + rank) *
                            (512u + 64u) * sizeof(uint32_t),
                        (512u + 64u) * sizeof(uint32_t));
                    sets.quant_intermediate[token][rank] = set({arena_range(ffn_,
                        floats((static_cast<uint64_t>(token) * kTopK + rank) * kMoeDimension),
                        floats(kMoeDimension)), intermediate});
                    sets.expert_down[token][rank] = set({intermediate, kernels_.dummy(),
                                                         route, token_accumulator});
                }
            }
            for (uint32_t token = 0; token < kDsparkBlock; ++token) {
                sets.reduce[token] = set({arena_range(accumulator_,
                    floats(static_cast<uint64_t>(token) * kTopK * kDimension),
                    floats(kTopK * kDimension)), arena_range(routed_reduced_,
                    floats(static_cast<uint64_t>(token) * kDimension), floats(kDimension))});
            }
            sets.shared_w1 = linear(whole(quantized_),
                matrix("ffn.shared_experts.w1.weight", kMoeDimension, kDimension),
                whole(dense_gate_));
            sets.shared_w3 = linear(whole(quantized_),
                matrix("ffn.shared_experts.w3.weight", kMoeDimension, kDimension),
                whole(dense_up_));
            sets.shared_swiglu = set({whole(dense_gate_), whole(dense_up_), kernels_.dummy(),
                                      whole(shared_intermediate_)});
            sets.quant_shared = set({whole(shared_intermediate_), whole(quantized_)});
            const DescriptorRange residual = whole(routed_reduced_);
            sets.shared_w2 = linear(whole(quantized_),
                matrix("ffn.shared_experts.w2.weight", kDimension, kMoeDimension),
                whole(reduced_), &residual);
            sets.hc_ffn_post = set({whole(reduced_), whole(hidden_alt_), whole(hc_split_), whole(hidden_)});
        }

        quant_head_state_ = set({whole(hidden_), whole(quantized_)});
        const TensorView& hc_head = index_.require("mtp.2.hc_head_fn");
        require_matrix(hc_head, kHcMultiplicity, kHcMultiplicity * kDimension, "DSpark HC head");
        hc_head_linear_ = linear(whole(quantized_), hc_head, whole(hc_head_mixes_));
        hc_head_apply_ = set({whole(hidden_), whole(hc_head_mixes_), whole(hc_head_params_),
                              whole(head_hidden_)});
        head_norm_set_ = set({whole(head_hidden_), data(index_.require("mtp.2.norm.weight")),
                              whole(head_norm_)});
        quant_head_ = set({whole(head_norm_), whole(quantized_)});
        const TensorView& markov_w1 = index_.require("mtp.2.markov_head.markov_w1.weight");
        const TensorView& markov_w2 = index_.require("mtp.2.markov_head.markov_w2.weight");
        require_matrix(markov_w1, kVocabulary, kDsparkMarkovRank, "DSpark Markov embedding");
        require_matrix(markov_w2, kVocabulary, kDsparkMarkovRank, "DSpark Markov head");
        markov_embedding_ = set({data(markov_w1), auxiliary(markov_w1), whole(output_ids_),
                                 whole(markov_embed_)});
        quant_markov_ = set({whole(markov_embed_), whole(markov_quantized_)});
        markov_head_ = linear(whole(markov_quantized_), markov_w2, whole(markov_bias_));
        for (uint32_t i = 0; i < kDsparkBlock; ++i) {
            logits_bias_[i] = set({whole(logits_), whole(markov_bias_), whole(combined_logits_)});
            argmax_[i] = set({whole(combined_logits_), arena_range(output_ids_,
                static_cast<VkDeviceSize>(i + 1u) * 4u * sizeof(uint32_t),
                4u * sizeof(uint32_t)), whole(argmax_workspace_)});
        }
    }

    VkPipeline gemv_pipeline(const LinearSet& item, bool residual, bool grouped) const {
        if (grouped) return item.format == TensorFormat::q4g64t ?
            kernels_.p().q4_grouped_batch : kernels_.p().q8_grouped_batch;
        if (residual) return item.format == TensorFormat::q4g64t ?
            kernels_.p().q4_residual_batch : kernels_.p().q8_residual_batch;
        return item.format == TensorFormat::q4g64t ? kernels_.p().q4_batch : kernels_.p().q8_batch;
    }
    void quantize(VkCommandBuffer command, VkDescriptorSet descriptor, uint32_t count,
                  uint32_t group, uint32_t packed, uint32_t scale) {
        const QuantizePush push{count, group, packed, scale};
        kernels_.dispatch(command, kernels_.p().quantize, descriptor, &push,
                          divide_up(count, group));
    }
    void gemv(VkCommandBuffer command, const LinearSet& item, uint32_t rows,
              uint32_t inner, uint32_t batch, uint32_t activation_scale,
              bool residual = false) {
        const GemvPush push{rows, inner, activation_scale, batch};
        kernels_.dispatch(command, gemv_pipeline(item, residual, false), item.set, &push,
                          divide_up(rows, 8u), batch);
    }
    void grouped_gemv(VkCommandBuffer command, const LinearSet& item, uint32_t groups,
                      uint32_t rows, uint32_t inner, uint32_t batch,
                      uint32_t activation_scale) {
        const GroupedGemvPush push{groups, rows, inner, activation_scale};
        kernels_.dispatch(command, gemv_pipeline(item, false, true), item.set, &push,
                          divide_up(rows, 8u), batch * groups);
    }
    void hc_pre(VkCommandBuffer command, VkDescriptorSet quant_set,
                const LinearSet& projection, VkDescriptorSet scale_set,
                VkDescriptorSet split_set, VkDescriptorSet pre_set,
                VkDescriptorSet norm_set, const Buffer&) {
        const uint32_t count = kDsparkBlock * kHcMultiplicity * kDimension;
        quantize(command, quant_set, count, 128u, packed_words(count), packed_words(count));
        compute_barrier(command);
        gemv(command, projection, 24u, kHcMultiplicity * kDimension, kDsparkBlock,
             packed_words(count));
        compute_barrier(command);
        const HcMixPush mix{kDimension, kHcMultiplicity, 24u, rms_bits()};
        kernels_.dispatch(command, kernels_.p().hc_mix_scale, scale_set, &mix, kDsparkBlock);
        compute_barrier(command);
        const HcSplitPush split{kDsparkBlock, kHcMultiplicity, 20u, rms_bits()};
        kernels_.dispatch(command, kernels_.p().hc_sinkhorn, split_set, &split, kDsparkBlock);
        compute_barrier(command);
        const HcApplyPush pre{kDimension, kHcMultiplicity, 24u, 0u};
        kernels_.dispatch(command, kernels_.p().hc_pre, pre_set, &pre,
                          divide_up(kDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        const RmsPush norm{kDsparkBlock, kDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, norm_set, &norm, kDsparkBlock);
        compute_barrier(command);
    }

    void record_main_x(VkCommandBuffer command) {
        const HcApplyPush mean{kDimension, 3u, kHcMultiplicity, 0u};
        kernels_.dispatch(command, kernels_.p().main_mean, main_mean_set_, &mean,
                          divide_up(3u * kDimension, 64u));
        compute_barrier(command);
        const uint32_t count = 3u * kDimension;
        quantize(command, quant_main_concat_, count, 128u, packed_words(count), packed_words(count));
        compute_barrier(command);
        gemv(command, main_proj_, kDimension, count, 1u, packed_words(count));
        compute_barrier(command);
        const RmsPush norm{1u, kDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, main_norm_set_, &norm, 1u);
        compute_barrier(command);
    }

    void record_main_kv(VkCommandBuffer command, uint32_t stage, uint32_t position) {
        StageSets& sets = stages_.at(stage);
        quantize(command, quant_main_x_, kDimension, 128u, 1024u, 1024u);
        compute_barrier(command);
        gemv(command, sets.wkv_main, kHeadDimension, kDimension, 1u, 1024u);
        compute_barrier(command);
        const RmsPush norm{1u, kHeadDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, sets.kv_main_norm, &norm, 1u);
        compute_barrier(command);
        const RopePush rope{1u, kHeadDimension, kRopeDimension, position};
        kernels_.dispatch(command, kernels_.p().rope_batch, sets.main_kv_rope, &rope,
                          divide_up(kHeadDimension, 64u), 1u);
        compute_barrier(command);
        const VkDeviceSize destination = floats((static_cast<uint64_t>(stage) *
            kDsparkKvSlots + (position % kWindow)) * kHeadDimension);
        copy_compute_result(command, main_kv_raw_, 0, kv_cache_, destination,
                            floats(kHeadDimension));
    }

    void record_embedding(VkCommandBuffer command) {
        const HcApplyPush push{kVocabulary, kDimension, kDimension / 4u, kHcMultiplicity};
        kernels_.dispatch(command, kernels_.p().embedding_batch, embedding_set_, &push,
                          divide_up(kDimension, 64u), kDsparkBlock * kHcMultiplicity);
        compute_barrier(command);
    }

    void record_stage_pre(VkCommandBuffer command, uint32_t stage, uint32_t position) {
        StageSets& sets = stages_.at(stage);
        record_main_kv(command, stage, position);
        hc_pre(command, sets.quant_hc_attn, sets.hc_attn_linear,
               sets.hc_attn_mix_scale, sets.hc_attn_split, sets.hc_attn_pre,
               sets.attn_norm, hidden_);
        const uint32_t normalized_count = kDsparkBlock * kDimension;
        quantize(command, sets.quant_normalized, normalized_count, 128u,
                 packed_words(normalized_count), packed_words(normalized_count));
        compute_barrier(command);
        gemv(command, sets.wq_a, 1024u, kDimension, kDsparkBlock,
             packed_words(normalized_count));
        gemv(command, sets.wkv, kHeadDimension, kDimension, kDsparkBlock,
             packed_words(normalized_count));
        compute_barrier(command);
        const RmsPush qnorm{kDsparkBlock, 1024u, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, sets.q_norm, &qnorm, kDsparkBlock);
        const RmsPush kvnorm{kDsparkBlock, kHeadDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, sets.kv_norm, &kvnorm, kDsparkBlock);
        compute_barrier(command);
        const uint32_t q_count = kDsparkBlock * 1024u;
        quantize(command, sets.quant_q_rank, q_count, 128u,
                 packed_words(q_count), packed_words(q_count));
        compute_barrier(command);
        gemv(command, sets.wq_b, kHeads * kHeadDimension, 1024u, kDsparkBlock,
             packed_words(q_count));
        compute_barrier(command);
        const RmsPush qhead{kDsparkBlock * kHeads, kHeadDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, sets.q_head_norm, &qhead,
                          kDsparkBlock * kHeads);
        compute_barrier(command);
        const RopePush qrope{kHeads, kHeadDimension, kRopeDimension, position + 1u};
        kernels_.dispatch(command, kernels_.p().rope_batch, sets.query_rope, &qrope,
                          divide_up(kHeads * kHeadDimension, 64u), kDsparkBlock);
        const RopePush kvrope{1u, kHeadDimension, kRopeDimension, position + 1u};
        kernels_.dispatch(command, kernels_.p().rope_batch, sets.kv_rope, &kvrope,
                          divide_up(kHeadDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        const VkDeviceSize stage_base = floats(static_cast<uint64_t>(stage) *
                                               kDsparkKvSlots * kHeadDimension);
        for (uint32_t token = 0; token < kDsparkBlock; ++token)
            copy_compute_result(command, kv_raw_, floats(static_cast<uint64_t>(token) * kHeadDimension),
                kv_cache_, stage_base + floats(static_cast<uint64_t>(kWindow + token) * kHeadDimension),
                floats(kHeadDimension));
        const AttentionPush attention{kHeads, kHeadDimension, position + 1u + kDsparkBlock,
                                      kDsparkBlock};
        kernels_.dispatch(command, kernels_.p().attention_batch, sets.attention, &attention,
                          kHeads, kDsparkBlock);
        compute_barrier(command);
        const RopePush inverse{kHeads, kHeadDimension, kRopeDimension,
                               (position + 1u) | 0x80000000u};
        kernels_.dispatch(command, kernels_.p().rope_batch, sets.inverse_rope, &inverse,
                          divide_up(kHeads * kHeadDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        const uint32_t context_count = kDsparkBlock * kHeads * kHeadDimension;
        quantize(command, sets.quant_context, context_count, 128u,
                 packed_words(context_count), packed_words(context_count));
        compute_barrier(command);
        grouped_gemv(command, sets.wo_a, 8u, 1024u, 4096u, kDsparkBlock,
                     packed_words(context_count));
        compute_barrier(command);
        const uint32_t o_count = kDsparkBlock * 8192u;
        quantize(command, sets.quant_o_rank, o_count, 128u,
                 packed_words(o_count), packed_words(o_count));
        compute_barrier(command);
        gemv(command, sets.wo_b, kDimension, 8192u, kDsparkBlock, packed_words(o_count));
        compute_barrier(command);
        const HcApplyPush apply{kDimension, kHcMultiplicity, 24u, 0u};
        kernels_.dispatch(command, kernels_.p().hc_post, sets.hc_attn_post, &apply,
                          divide_up(kHcMultiplicity * kDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        hc_pre(command, sets.quant_hc_ffn, sets.hc_ffn_linear,
               sets.hc_ffn_mix_scale, sets.hc_ffn_split, sets.hc_ffn_pre,
               sets.ffn_norm, hidden_alt_);
        quantize(command, sets.quant_normalized, normalized_count, 128u,
                 packed_words(normalized_count), packed_words(normalized_count));
        compute_barrier(command);
        gemv(command, sets.router, kExperts, kDimension, kDsparkBlock,
             packed_words(normalized_count));
        compute_barrier(command);
        const RouterPush router{kExperts, kTopK, float_word(main_index_.header().route_scale),
                                kDsparkBlock};
        kernels_.dispatch(command, kernels_.p().router_batch, sets.router_select, &router,
                          kDsparkBlock);
        compute_barrier(command);
    }

    void update_expert_sets(uint32_t stage, uint32_t token) {
        StageSets& sets = stages_.at(stage);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const DescriptorRange record = cache_->record(stage, current_slots_[token][rank]);
            kernels_.update(sets.expert_gate[token][rank], 1u, record);
            kernels_.update(sets.expert_down[token][rank], 1u, record);
        }
    }

    void record_stage_expert(VkCommandBuffer command, uint32_t stage, uint32_t token) {
        StageSets& sets = stages_.at(stage);
        const QuantizePush quant{kDimension, 32u, 1024u, 1024u};
        kernels_.dispatch(command, kernels_.p().quantize, sets.quant_expert[token], &quant,
                          divide_up(kDimension, 32u));
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const ExpertPush gate{rank, 1024u, swiglu_bits(), 0u};
            kernels_.dispatch(command, kernels_.p().expert_gate,
                sets.expert_gate[token][rank], &gate, divide_up(kMoeDimension, 8u));
        }
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const QuantizePush intermediate{kMoeDimension, 32u, 512u, 512u};
            kernels_.dispatch(command, kernels_.p().quantize,
                sets.quant_intermediate[token][rank], &intermediate,
                divide_up(kMoeDimension, 32u));
        }
        compute_barrier(command);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            const ExpertPush down{rank, 512u, 0u, 0u};
            kernels_.dispatch(command, kernels_.p().expert_down,
                sets.expert_down[token][rank], &down, divide_up(kDimension, 8u));
        }
        compute_barrier(command);
        const QuantizePush reduce{kDimension, kTopK, 1u, 0u};
        kernels_.dispatch(command, kernels_.p().reduce_batch, sets.reduce[token], &reduce,
                          divide_up(kDimension, 64u), 1u);
        compute_barrier(command);
    }

    void record_stage_shared(VkCommandBuffer command, uint32_t stage) {
        StageSets& sets = stages_.at(stage);
        const uint32_t normalized_count = kDsparkBlock * kDimension;
        gemv(command, sets.shared_w1, kMoeDimension, kDimension, kDsparkBlock,
             packed_words(normalized_count));
        gemv(command, sets.shared_w3, kMoeDimension, kDimension, kDsparkBlock,
             packed_words(normalized_count));
        compute_barrier(command);
        const SwigluPush swiglu{kMoeDimension, swiglu_bits(), kDsparkBlock, 0u};
        kernels_.dispatch(command, kernels_.p().swiglu_batch, sets.shared_swiglu, &swiglu,
                          divide_up(kMoeDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        const uint32_t shared_count = kDsparkBlock * kMoeDimension;
        quantize(command, sets.quant_shared, shared_count, 128u,
                 packed_words(shared_count), packed_words(shared_count));
        compute_barrier(command);
        gemv(command, sets.shared_w2, kDimension, kMoeDimension, kDsparkBlock,
             packed_words(shared_count), true);
        compute_barrier(command);
        const HcApplyPush apply{kDimension, kHcMultiplicity, 24u, 0u};
        kernels_.dispatch(command, kernels_.p().hc_post, sets.hc_ffn_post, &apply,
                          divide_up(kHcMultiplicity * kDimension, 64u), kDsparkBlock);
        compute_barrier(command);
    }

    void record_head(VkCommandBuffer command) {
        const uint32_t hc_count = kDsparkBlock * kHcMultiplicity * kDimension;
        quantize(command, quant_head_state_, hc_count, 128u,
                 packed_words(hc_count), packed_words(hc_count));
        compute_barrier(command);
        gemv(command, hc_head_linear_, kHcMultiplicity, kHcMultiplicity * kDimension,
             kDsparkBlock, packed_words(hc_count));
        compute_barrier(command);
        const HcApplyPush head{kDimension, kHcMultiplicity, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().hc_head_apply, hc_head_apply_, &head,
                          divide_up(kDimension, 64u), kDsparkBlock);
        compute_barrier(command);
        const RmsPush norm{kDsparkBlock, kDimension, rms_bits(), 0u};
        kernels_.dispatch(command, kernels_.p().rmsnorm, head_norm_set_, &norm, kDsparkBlock);
        compute_barrier(command);
        const uint32_t head_count = kDsparkBlock * kDimension;
        quantize(command, quant_head_, head_count, 128u,
                 packed_words(head_count), packed_words(head_count));
        compute_barrier(command);
        gemv(command, head_, kVocabulary, kDimension, kDsparkBlock, packed_words(head_count));
        compute_barrier(command);
        const uint32_t groups = divide_up(kVocabulary, 64u * 8u);
        for (uint32_t i = 0; i < kDsparkBlock; ++i) {
            const HcApplyPush embedding{kVocabulary, kDsparkMarkovRank,
                                        kDsparkMarkovRank / 4u, i * 4u};
            kernels_.dispatch(command, kernels_.p().embedding_indexed, markov_embedding_,
                              &embedding, divide_up(kDsparkMarkovRank, 64u));
            compute_barrier(command);
            quantize(command, quant_markov_, kDsparkMarkovRank, 128u, 64u, 64u);
            compute_barrier(command);
            gemv(command, markov_head_, kVocabulary, kDsparkMarkovRank, 1u, 64u);
            compute_barrier(command);
            const HcApplyPush bias{kVocabulary, i * kVocabulary, 0u, 0u};
            kernels_.dispatch(command, kernels_.p().logits_bias, logits_bias_[i], &bias,
                              divide_up(kVocabulary, 64u));
            compute_barrier(command);
            const ArgmaxPush first{kVocabulary, groups, 0u, 0u};
            kernels_.dispatch(command, kernels_.p().greedy_argmax, argmax_[i], &first, groups);
            compute_barrier(command);
            const ArgmaxPush second{kVocabulary, groups, 1u, 0u};
            kernels_.dispatch(command, kernels_.p().greedy_argmax, argmax_[i], &second, 1u);
            compute_barrier(command);
        }
    }

    void update_indices(uint32_t position) {
        auto* indices = static_cast<uint32_t*>(indices_.mapped);
        uint32_t count = 0;
        const uint32_t history = std::min<uint32_t>(position + 1u, kWindow);
        const uint32_t start = position + 1u - history;
        for (uint32_t i = 0; i < history; ++i) indices[count++] = (start + i) % kWindow;
        for (uint32_t i = 0; i < kDsparkBlock; ++i) indices[count++] = kWindow + i;
        while (count < 160u) indices[count++] = UINT32_MAX;
        flush_buffer(runtime_, indices_);
    }

    const Runtime& runtime_;
    ExecutorScaffold& main_executor_;
    const SharedIndex& main_index_;
    const DsparkIndex& index_;
    const ReadOnlyMapping& shared_file_;
    FiniteQueue compute_;
    DsparkKernels kernels_;
    std::unique_ptr<DsparkExpertCache> cache_;
    Buffer shared_arena_{}, hidden_{}, hidden_alt_{}, reduced_{}, normalized_{};
    Buffer main_concat_{}, main_x_{}, main_kv_raw_{}, main_kv_{};
    Buffer q_rank_{}, q_rank_norm_{}, query_{}, kv_raw_{}, kv_{}, context_{}, o_rank_{};
    Buffer router_logits_{}, routing_{}, ffn_{}, accumulator_{}, routed_reduced_{};
    Buffer dense_gate_{}, dense_up_{}, shared_intermediate_{};
    Buffer quantized_{}, expert_quantized_{}, intermediate_quantized_{}, kv_cache_{};
    Buffer indices_{}, draft_input_ids_{}, output_ids_{}, ones_{}, rope_cos_{}, rope_sin_{};
    Buffer hc_mixes_{}, hc_split_{}, hc_head_mixes_{}, head_hidden_{}, head_norm_{};
    Buffer logits_{}, markov_embed_{}, markov_quantized_{}, markov_bias_{};
    Buffer combined_logits_{}, argmax_workspace_{}, hc_head_params_{};
    VkDescriptorSet main_mean_set_{}, quant_main_concat_{}, main_norm_set_{}, quant_main_x_{};
    LinearSet main_proj_{}, head_{}, hc_head_linear_{}, markov_head_{};
    VkDescriptorSet embedding_set_{}, quant_head_state_{}, hc_head_apply_{}, head_norm_set_{};
    VkDescriptorSet quant_head_{}, markov_embedding_{}, quant_markov_{};
    std::array<VkDescriptorSet, kDsparkBlock> logits_bias_{}, argmax_{};
    std::array<StageSets, kDsparkStages> stages_{};
    std::array<std::array<uint32_t, kTopK>, kDsparkBlock> current_slots_{};
    uint64_t draft_passes_ = 0;
    double draft_seconds_ = 0.0, draft_gpu_seconds_ = 0.0;
};

class MainBatchVerifier {
public:
#ifndef OVLLM_DSPARK_VERIFY_BLOCK
#define OVLLM_DSPARK_VERIFY_BLOCK 2
#endif
    static constexpr uint32_t kDsparkBlock = OVLLM_DSPARK_VERIFY_BLOCK;
    static_assert(kDsparkBlock >= 1u && kDsparkBlock <= dsv4::kDsparkBlock,
                  "DSpark verifier block must fit the five published proposals");
    struct UniqueExpertMetrics {
        uint64_t authoritative_layers = 0;
        uint64_t routed_occurrences = 0;
        uint64_t unique_experts = 0;
        uint64_t initial_resident_unique = 0;
        uint64_t acquired_unique = 0;
        uint64_t reused_occurrences = 0;
        uint64_t layers_with_reuse = 0;
        uint64_t maximum_unique_experts = 0;
        uint64_t gate_dispatches = 0;
        uint64_t down_dispatches = 0;
    };
    MainBatchVerifier(const Runtime& runtime, ExecutorScaffold& executor,
                      DeepSeekProgram& main_program, const SharedIndex& index,
                      const ReadOnlyMapping& shared_file,
                      const std::filesystem::path& shader_directory)
        : runtime_(runtime), executor_(executor), main_program_(main_program), index_(index),
          shared_file_(shared_file), compute_(runtime, runtime.queue),
          kernels_(runtime, shader_directory) {
        q4_experts_ = std::getenv("DSV4_Q4_EXPERTS") != nullptr;
        q4_swar_ = q4_experts_ && std::getenv("DSV4_Q4_SWAR") != nullptr;
        batch_quant_fused_ = std::getenv("DSV4_BATCH_QUANT_FUSED") != nullptr;
        unique_expert_verify_ =
            std::getenv("DSV4_DSPARK_UNIQUE_EXPERT_VERIFY") != nullptr;
        if (unique_expert_verify_ && (!q4_experts_ || !q4_swar_))
            throw std::runtime_error(
                "DSV4_DSPARK_UNIQUE_EXPERT_VERIFY requires Q4 experts and SWAR");
        hidden_ = device(floats(kDsparkBlock * kHcMultiplicity * kDimension));
        hidden_alt_ = device(floats(kDsparkBlock * kHcMultiplicity * kDimension));
        reduced_ = device(floats(kDsparkBlock * kDimension));
        normalized_ = device(floats(kDsparkBlock * kDimension));
        q_rank_ = device(floats(kDsparkBlock * 1024u));
        q_rank_norm_ = device(floats(kDsparkBlock * 1024u));
        query_ = device(floats(kDsparkBlock * kHeads * kHeadDimension));
        kv_raw_ = device(floats(kDsparkBlock * kHeadDimension));
        kv_ = device(floats(kDsparkBlock * kHeadDimension));
        context_ = device(floats(kDsparkBlock * kHeads * kHeadDimension));
        o_rank_ = device(floats(kDsparkBlock * 8192u));
        router_logits_ = device(floats(kDsparkBlock * kExperts));
        routing_ = host(kDsparkBlock * 64u * sizeof(uint32_t));
        ffn_ = device(floats(kDsparkBlock * kTopK * kMoeDimension));
        accumulator_ = device(floats(kDsparkBlock * kTopK * kDimension));
        routed_reduced_ = device(floats(kDsparkBlock * kDimension));
        dense_gate_ = device(floats(kDsparkBlock * kMoeDimension));
        dense_up_ = device(floats(kDsparkBlock * kMoeDimension));
        shared_intermediate_ = device(floats(kDsparkBlock * kMoeDimension));
        quantized_ = device(static_cast<VkDeviceSize>(
            kDsparkBlock * kHeads * kHeadDimension / 4u +
            kDsparkBlock * kHeads * kHeadDimension / 128u) * sizeof(uint32_t));
        expert_quantized_ = device(static_cast<VkDeviceSize>(kDsparkBlock) *
                                   (1024u + 128u) * sizeof(uint32_t));
        intermediate_quantized_ = device(static_cast<VkDeviceSize>(kDsparkBlock) * kTopK *
                                          (512u + 64u) * sizeof(uint32_t));
        hc_mixes_ = device(floats(kDsparkBlock * 24u));
        hc_split_ = device(floats(kDsparkBlock * 24u));
        compressor_kv_batch_ = device(floats(kDsparkBlock * 1024u));
        compressor_score_batch_ = device(floats(kDsparkBlock * 1024u));
        compressed_tmp_ = device(floats(kHeadDimension));
        compressed_rope_ = device(floats(kHeadDimension));
        tokens_ = host(kDsparkBlock * sizeof(uint32_t));
        output_ids_ = host(kDsparkBlock * 4u * sizeof(uint32_t));
        indices_ = host(kDsparkBlock * 160u * sizeof(uint32_t));
        ones_ = host(1024u * sizeof(float));
        hc_head_params_ = host(5u * sizeof(float));
        rope_cos_plain_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        rope_sin_plain_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        rope_cos_compressed_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        rope_sin_compressed_ = host((kShortContext + kDsparkBlock + 1u) * 32u * sizeof(float));
        logits_ = device(floats(kDsparkBlock * kVocabulary));
        argmax_workspace_ = device(kDsparkBlock * 256u * 2u * sizeof(uint32_t));
        target_captures_ = device(floats(kDsparkBlock * 3u * kHcMultiplicity * kDimension));
        if (unique_expert_verify_)
            unique_occurrences_ = host(static_cast<VkDeviceSize>(kDsparkBlock) *
                kTopK * dsv4::kDsparkBlock * sizeof(uint32_t));
        initialize_constants();
        build_sets();
    }
    ~MainBatchVerifier() {
        for (Buffer* buffer : buffers()) destroy_buffer(runtime_, *buffer);
    }

    std::array<uint32_t, kDsparkBlock> verify(
            const std::array<uint32_t, kDsparkBlock>& tokens, uint32_t start_position) {
        if (start_position + kDsparkBlock > kShortContext)
            throw std::runtime_error("Batched verification exceeds short-context cap");
        const auto started = std::chrono::steady_clock::now();
        std::memcpy(tokens_.mapped, tokens.data(), sizeof(tokens));
        flush_buffer(runtime_, tokens_);
        std::fill(static_cast<uint32_t*>(output_ids_.mapped),
                  static_cast<uint32_t*>(output_ids_.mapped) + kDsparkBlock * 4u, 0u);
        flush_buffer(runtime_, output_ids_);
        const uint64_t embedded = compute_.submit([&](VkCommandBuffer command) {
            const HcApplyPush push{kVocabulary, kDimension, kDimension / 4u, kHcMultiplicity};
            kernels_.dispatch(command, kernels_.p().embedding_batch, embedding_, &push,
                divide_up(kDimension, 64u), kDsparkBlock * kHcMultiplicity);
            compute_barrier(command);
        });
        compute_.wait(embedded);
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            update_indices(start_position, index_.compression_ratio(layer) == 4u);
            const auto pre_started = std::chrono::steady_clock::now();
            const uint64_t pre = compute_.submit([&](VkCommandBuffer command) {
                record_pre(command, layer, start_position);
                host_read_barrier(command);
            });
            compute_.wait(pre);
            pre_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - pre_started).count();
            // The shared expert depends only on the normalized FFN input, not
            // on the routed-expert records.  Launch its gate/up/SwiGLU/Q8 work
            // before touching storage so GPU work overlaps the bounded CPU/NVMe
            // acquisition below.  FiniteQueue owns one finite command buffer;
            // the later submit waits this segment before recording the residual
            // down projection, preserving ordinary Vulkan ordering.
            compute_.submit([&](VkCommandBuffer command) {
                record_shared_pre(command, layer);
            });
            invalidate_buffer(runtime_, routing_);
            const auto acquire_started = std::chrono::steady_clock::now();
            const auto resolved = executor_.acquire_batch_experts(
                layer, static_cast<const uint32_t*>(routing_.mapped), kDsparkBlock);
            slots_ = resolved.slots;
            cache_hits_ += resolved.hits;
            cache_misses_ += resolved.misses;
            if (unique_expert_verify_)
                build_unique_expert_plan(
                    static_cast<const uint32_t*>(routing_.mapped), resolved.misses);
            update_expert_sets(layer);
            acquire_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - acquire_started).count();
            const auto post_started = std::chrono::steady_clock::now();
            const uint64_t post = compute_.submit([&](VkCommandBuffer command) {
                record_routed_and_finish(command, layer);
            }, resolved.ready ? transfer_semaphore() : VK_NULL_HANDLE, resolved.ready);
            compute_.wait(post);
            post_seconds_ += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - post_started).count();
        }
        const auto final_started = std::chrono::steady_clock::now();
        const uint64_t final = compute_.submit([&](VkCommandBuffer command) {
            record_final(command);
            host_read_barrier(command);
        });
        compute_.wait(final);
        final_seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - final_started).count();
        invalidate_buffer(runtime_, output_ids_);
        std::array<uint32_t, kDsparkBlock> result{};
        const auto* words = static_cast<const uint32_t*>(output_ids_.mapped);
        for (uint32_t i = 0; i < kDsparkBlock; ++i) {
            result[i] = words[i * 4u];
            if (result[i] >= kVocabulary)
                throw std::runtime_error("Batched main verifier emitted invalid token");
        }
        ++passes_;
        seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        return result;
    }
    double seconds() const { return seconds_; }
    double pre_seconds() const { return pre_seconds_; }
    double acquire_seconds() const { return acquire_seconds_; }
    double post_seconds() const { return post_seconds_; }
    double final_seconds() const { return final_seconds_; }
    uint64_t passes() const { return passes_; }
    uint64_t cache_hits() const { return cache_hits_; }
    uint64_t cache_misses() const { return cache_misses_; }
    const Buffer& target_captures() const { return target_captures_; }
    bool unique_expert_verify() const { return unique_expert_verify_; }
    UniqueExpertMetrics unique_expert_metrics() const { return unique_metrics_; }

private:
    struct Linear { VkDescriptorSet set{}; TensorFormat format{}; };
    struct Layer {
        bool ratio4 = false, compressed_rope = false, hash = false;
        VkDescriptorSet hc_attn_mix{}, hc_attn_split{}, hc_attn_pre{}, attn_norm{};
        Linear wq_a{}, wq_b{}, wkv{}, wo_a{}, wo_b{}, compressor_wkv{}, compressor_gate{};
        VkDescriptorSet q_norm{}, quant_q_rank{}, q_head_norm{}, kv_norm{};
        VkDescriptorSet query_rope{}, kv_rope{}, attention{}, inverse_rope{};
        VkDescriptorSet quant_normalized{}, quant_context{}, quant_o_rank{};
        VkDescriptorSet compress{};
        std::array<VkDescriptorSet, 32> compressed_norm{};
        VkDescriptorSet compressed_rope_set{}, hc_attn_post{};
        VkDescriptorSet hc_ffn_mix{}, hc_ffn_split{}, hc_ffn_pre{}, ffn_norm{};
        Linear router{}, shared_w1{}, shared_w3{}, shared_w2{};
        VkDescriptorSet router_select{}, reduce{}, shared_swiglu{}, quant_shared{}, hc_ffn_post{};
        std::array<VkDescriptorSet, kDsparkBlock> quant_expert{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> quant_intermediate{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> expert_gate{};
        std::array<std::array<VkDescriptorSet, kTopK>, kDsparkBlock> expert_down{};
    };
    struct UniqueExpertGroup {
        uint32_t expert = UINT32_MAX;
        uint32_t slot = UINT32_MAX;
        uint32_t occurrences = 0;
    };

    static VkDeviceSize floats(uint64_t count) { return count * sizeof(float); }
    static uint32_t words(uint32_t count) { return divide_up(count, 4u); }
    Buffer device(VkDeviceSize bytes) { return create_device_buffer(runtime_, bytes); }
    Buffer host(VkDeviceSize bytes) { return create_buffer(runtime_, bytes); }
    std::vector<Buffer*> buffers() {
        return {&unique_occurrences_,&target_captures_,&argmax_workspace_,&logits_,&rope_sin_compressed_,&rope_cos_compressed_,
            &rope_sin_plain_,&rope_cos_plain_,&hc_head_params_,&ones_,&indices_,&output_ids_,
            &tokens_,&compressed_rope_,&compressed_tmp_,&compressor_score_batch_,
            &compressor_kv_batch_,&hc_split_,&hc_mixes_,&intermediate_quantized_,
            &expert_quantized_,&quantized_,&shared_intermediate_,&dense_up_,&dense_gate_,
            &routed_reduced_,&accumulator_,&ffn_,&routing_,&router_logits_,&o_rank_,&context_,
            &kv_,&kv_raw_,&query_,&q_rank_norm_,&q_rank_,&normalized_,&reduced_,&hidden_alt_,&hidden_};
    }
    VkDescriptorSet set(std::initializer_list<DescriptorRange> ranges) {
        return kernels_.set(ranges);
    }
    Linear linear(const DescriptorRange& activation, const TensorView& matrix,
                  const Buffer& arena, uint64_t base, const DescriptorRange& output,
                  const DescriptorRange* residual = nullptr) {
        require_linear_matrix(matrix, static_cast<uint32_t>(matrix.shape[0]),
                              static_cast<uint32_t>(matrix.shape[1]), "batch matrix");
        std::array<DescriptorRange,6> ranges{}; ranges.fill(kernels_.dummy());
        ranges[0]=activation; ranges[1]=tensor_data_range(arena,base,matrix);
        ranges[2]=tensor_auxiliary_range(arena,base,matrix);ranges[3]=output;
        if(residual)ranges[4]=*residual;
        return {kernels_.set(ranges),matrix.format};
    }
    VkPipeline pipeline(const Linear& item, bool residual=false, bool grouped=false) const {
        if(grouped)return item.format==TensorFormat::q4g64t?kernels_.p().q4_grouped_batch:kernels_.p().q8_grouped_batch;
        if(residual)return item.format==TensorFormat::q4g64t?kernels_.p().q4_residual_batch:kernels_.p().q8_residual_batch;
        return item.format==TensorFormat::q4g64t?kernels_.p().q4_batch:kernels_.p().q8_batch;
    }
    void quantize(VkCommandBuffer c,VkDescriptorSet s,uint32_t count,uint32_t group,uint32_t packed,uint32_t scale){QuantizePush p{count,group,packed,scale};kernels_.dispatch(c,kernels_.p().quantize,s,&p,divide_up(count,group));}
    void gemv(VkCommandBuffer c,const Linear& l,uint32_t rows,uint32_t inner,uint32_t batch,uint32_t scale,bool residual=false){GemvPush p{rows,inner,scale,batch};kernels_.dispatch(c,pipeline(l,residual,false),l.set,&p,divide_up(rows,8u),batch);}
    void grouped(VkCommandBuffer c,const Linear& l,uint32_t groups,uint32_t rows,uint32_t inner,uint32_t batch,uint32_t scale){GroupedGemvPush p{groups,rows,inner,scale};kernels_.dispatch(c,pipeline(l,false,true),l.set,&p,divide_up(rows,8u),groups*batch);}
    uint32_t rms_bits()const{return float_word(index_.header().rms_epsilon);}
    uint32_t swiglu_bits()const{return float_word(index_.header().swiglu_limit);}
    VkSemaphore transfer_semaphore() const { return executor_.expert_transfer_semaphore(); }

    void host_read_barrier(VkCommandBuffer command) {
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkfn::CmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    }

    void fill_rope(Buffer& cosine, Buffer& sine, float base, bool yarn) {
        auto* c=static_cast<float*>(cosine.mapped);auto* s=static_cast<float*>(sine.mapped);
        const float pi=3.14159265358979323846f;float low=0.0f,high=0.0f;
        if(yarn){const float original=index_.header().original_max_position;
            low=std::floor(kRopeDimension*std::log(original/(index_.header().beta_fast*2.0f*pi))/(2.0f*std::log(base)));
            high=std::ceil(kRopeDimension*std::log(original/(index_.header().beta_slow*2.0f*pi))/(2.0f*std::log(base)));
            low=std::max(low,0.0f);high=std::min(high,static_cast<float>(kRopeDimension-1));if(low==high)high+=0.001f;}
        for(uint32_t f=0;f<32u;++f){float value=1.0f/std::pow(base,static_cast<float>(f*2u)/kRopeDimension);
            if(yarn){float ramp=std::clamp((static_cast<float>(f)-low)/(high-low),0.0f,1.0f),smooth=1.0f-ramp;
                value=value/index_.header().yarn_factor*(1.0f-smooth)+value*smooth;}
            for(uint32_t p=0;p<=kShortContext+kDsparkBlock;++p){float angle=static_cast<float>(p)*value;c[p*32u+f]=std::cos(angle);s[p*32u+f]=std::sin(angle);}}
        flush_buffer(runtime_,cosine);flush_buffer(runtime_,sine);
    }
    void initialize_constants() {
        auto* one=static_cast<float*>(ones_.mapped);std::fill(one,one+1024u,1.0f);flush_buffer(runtime_,ones_);
        fill_rope(rope_cos_plain_,rope_sin_plain_,index_.header().rope_theta,false);
        fill_rope(rope_cos_compressed_,rope_sin_compressed_,index_.header().compress_rope_theta,true);
        const TensorView& scale=index_.require("hc_head_scale");const TensorView& base=index_.require("hc_head_base");
        auto* p=static_cast<float*>(hc_head_params_.mapped);
        std::memcpy(p,shared_file_.data()+scale.offset,sizeof(float));
        std::memcpy(p+1,shared_file_.data()+base.offset,4u*sizeof(float));flush_buffer(runtime_,hc_head_params_);
        ratio_slot_.fill(-1);for(uint32_t layer=0;layer<kLayers;++layer)if(index_.compression_ratio(layer)==4)ratio_slot_[layer]=static_cast<int32_t>(ratio_layers_++);
    }

    void build_sets() {
        const Buffer& global=executor_.global_arena();const uint64_t global_base=executor_.global_file_base();
        const TensorView& embedding=index_.require("embed.weight");
        embedding_=set({tensor_data_range(global,global_base,embedding),tensor_auxiliary_range(global,global_base,embedding),whole(tokens_),whole(hidden_)});
        quant_expert_batch_=set({whole(normalized_),whole(expert_quantized_)});
        quant_intermediate_batch_=set({whole(ffn_),whole(intermediate_quantized_)});
        layers_.resize(kLayers);
        for(uint32_t layer=0;layer<kLayers;++layer){Layer& z=layers_[layer];z.ratio4=index_.compression_ratio(layer)==4;z.compressed_rope=index_.compression_ratio(layer)!=0;z.hash=layer<kHashLayers;
            const uint32_t parity=layer&1u;const Buffer& arena=executor_.shared_arena(layer,parity);const uint64_t base=executor_.shared_file_base(layer,parity);
            const std::string prefix="layers."+std::to_string(layer)+".";
            const auto t=[&](const char* suffix)->const TensorView&{return index_.require(prefix+suffix);};
            const auto d=[&](const TensorView& x){return tensor_data_range(arena,base,x);};
            z.hc_attn_mix=set({whole(hidden_),d(t("hc_attn_fn")),whole(hc_mixes_)});
            z.hc_attn_split=set({whole(hc_mixes_),d(t("hc_attn_scale")),d(t("hc_attn_base")),whole(hc_split_)});
            z.hc_attn_pre=set({whole(hidden_),whole(hc_split_),whole(reduced_)});
            z.attn_norm=set({whole(reduced_),d(t("attn_norm.weight")),whole(normalized_)});
            z.quant_normalized=set({whole(normalized_),whole(quantized_)});
            z.wq_a=linear(whole(quantized_),t("attn.wq_a.weight"),arena,base,whole(q_rank_));
            z.wkv=linear(whole(quantized_),t("attn.wkv.weight"),arena,base,whole(kv_raw_));
            z.q_norm=set({whole(q_rank_),d(t("attn.q_norm.weight")),whole(q_rank_norm_)});
            z.quant_q_rank=set({whole(q_rank_norm_),whole(quantized_)});
            z.wq_b=linear(whole(quantized_),t("attn.wq_b.weight"),arena,base,whole(query_));
            z.q_head_norm=set({whole(query_),whole(ones_),whole(context_)});
            Buffer& rc=z.compressed_rope?rope_cos_compressed_:rope_cos_plain_;Buffer& rs=z.compressed_rope?rope_sin_compressed_:rope_sin_plain_;
            z.query_rope=set({whole(context_),whole(rc),whole(rs),whole(query_)});
            z.kv_norm=set({whole(kv_raw_),d(t("attn.kv_norm.weight")),whole(kv_)});
            z.kv_rope=set({whole(kv_),whole(rc),whole(rs),whole(kv_raw_)});
            const VkDeviceSize kv_layer=floats((kWindow+32u)*kHeadDimension);
            z.attention=set({whole(query_),arena_range(executor_.kv_cache(),static_cast<VkDeviceSize>(layer)*kv_layer,kv_layer),whole(indices_),d(t("attn.attn_sink")),whole(context_)});
            z.inverse_rope=set({whole(context_),whole(rc),whole(rs),whole(query_)});
            z.quant_context=set({whole(query_),whole(quantized_)});
            z.wo_a=linear(whole(quantized_),t("attn.wo_a.weight"),arena,base,whole(o_rank_));
            z.quant_o_rank=set({whole(o_rank_),whole(quantized_)});
            z.wo_b=linear(whole(quantized_),t("attn.wo_b.weight"),arena,base,whole(reduced_));
            if(z.ratio4){uint32_t slot=static_cast<uint32_t>(ratio_slot_[layer]);VkDeviceSize history=floats(128u*1024u),raw=floats(32u*512u);
                z.compressor_wkv=linear(whole(quantized_),t("attn.compressor.wkv.weight"),arena,base,whole(compressor_kv_batch_));
                z.compressor_gate=linear(whole(quantized_),t("attn.compressor.wgate.weight"),arena,base,whole(compressor_score_batch_));
                z.compress=set({arena_range(main_program_.compressor_kv_history(),static_cast<VkDeviceSize>(slot)*history,history),arena_range(main_program_.compressor_score_history(),static_cast<VkDeviceSize>(slot)*history,history),d(t("attn.compressor.ape")),arena_range(main_program_.compressor_raw_history(),static_cast<VkDeviceSize>(slot)*raw,raw)});
                for(uint32_t g=0;g<32u;++g)z.compressed_norm[g]=set({arena_range(main_program_.compressor_raw_history(),static_cast<VkDeviceSize>(slot)*raw+floats(g*512u),floats(512u)),d(t("attn.compressor.norm.weight")),whole(compressed_tmp_)});
                z.compressed_rope_set=set({whole(compressed_tmp_),whole(rope_cos_compressed_),whole(rope_sin_compressed_),whole(compressed_rope_)});
            }
            z.hc_attn_post=set({whole(reduced_),whole(hidden_),whole(hc_split_),whole(hidden_alt_)});
            z.hc_ffn_mix=set({whole(hidden_alt_),d(t("hc_ffn_fn")),whole(hc_mixes_)});
            z.hc_ffn_split=set({whole(hc_mixes_),d(t("hc_ffn_scale")),d(t("hc_ffn_base")),whole(hc_split_)});
            z.hc_ffn_pre=set({whole(hidden_alt_),whole(hc_split_),whole(reduced_)});
            z.ffn_norm=set({whole(reduced_),d(t("ffn_norm.weight")),whole(normalized_)});
            z.router=linear(whole(quantized_),t("ffn.gate.weight"),arena,base,whole(router_logits_));
            z.router_select=z.hash?set({whole(router_logits_),d(t("ffn.gate.tid2eid")),whole(tokens_),whole(routing_)}):set({whole(router_logits_),d(t("ffn.gate.bias")),kernels_.dummy(),whole(routing_)});
            for(uint32_t token=0;token<kDsparkBlock;++token){DescriptorRange norm=arena_range(normalized_,floats(static_cast<uint64_t>(token)*kDimension),floats(kDimension));DescriptorRange eq=arena_range(expert_quantized_,static_cast<VkDeviceSize>(token)*(1024u+128u)*4u,(1024u+128u)*4u);z.quant_expert[token]=set({norm,eq});DescriptorRange route=arena_range(routing_,token*64u*4u,64u*4u);DescriptorRange tf=arena_range(ffn_,floats(static_cast<uint64_t>(token)*kTopK*kMoeDimension),floats(kTopK*kMoeDimension));DescriptorRange ta=arena_range(accumulator_,floats(static_cast<uint64_t>(token)*kTopK*kDimension),floats(kTopK*kDimension));
                for(uint32_t rank=0;rank<kTopK;++rank){
                    DescriptorRange iq=arena_range(intermediate_quantized_,static_cast<VkDeviceSize>(token*kTopK+rank)*(512u+64u)*4u,(512u+64u)*4u);
                    z.quant_intermediate[token][rank]=set({arena_range(ffn_,floats((static_cast<uint64_t>(token)*kTopK+rank)*kMoeDimension),floats(kMoeDimension)),iq});
                    if(unique_expert_verify_){
                        // The first N token/rank sets become unique-expert
                        // descriptor slots at run time.  Whole verifier
                        // buffers plus an occurrence table let one packed-Q4
                        // dispatch process every position selecting a record.
                        z.expert_gate[token][rank]=set({whole(expert_quantized_),kernels_.dummy(),whole(routing_),whole(ffn_),whole(unique_occurrences_)});
                        z.expert_down[token][rank]=set({whole(intermediate_quantized_),kernels_.dummy(),whole(routing_),whole(accumulator_),whole(unique_occurrences_)});
                    }else{
                        z.expert_gate[token][rank]=set({eq,kernels_.dummy(),route,tf});
                        z.expert_down[token][rank]=set({iq,kernels_.dummy(),route,ta});
                    }
                }}
            z.reduce=set({whole(accumulator_),whole(routed_reduced_)});
            z.shared_w1=linear(whole(quantized_),t("ffn.shared_experts.w1.weight"),arena,base,whole(dense_gate_));
            z.shared_w3=linear(whole(quantized_),t("ffn.shared_experts.w3.weight"),arena,base,whole(dense_up_));
            z.shared_swiglu=set({whole(dense_gate_),whole(dense_up_),kernels_.dummy(),whole(shared_intermediate_)});z.quant_shared=set({whole(shared_intermediate_),whole(quantized_)});DescriptorRange residual=whole(routed_reduced_);
            z.shared_w2=linear(whole(quantized_),t("ffn.shared_experts.w2.weight"),arena,base,whole(reduced_),&residual);
            z.hc_ffn_post=set({whole(reduced_),whole(hidden_alt_),whole(hc_split_),whole(hidden_)});
        }
        const TensorView& head_fn=index_.require("hc_head_fn");hc_head_=set({whole(hidden_),tensor_data_range(global,global_base,head_fn),whole(hc_head_params_),whole(reduced_)});
        norm_=set({whole(reduced_),tensor_data_range(global,global_base,index_.require("norm.weight")),whole(normalized_)});quant_final_=set({whole(normalized_),whole(quantized_)});
        const TensorView& head=index_.require("head.weight");head_=linear(whole(quantized_),head,global,global_base,whole(logits_));
        for(uint32_t i=0;i<kDsparkBlock;++i)argmax_[i]=set({arena_range(logits_,floats(static_cast<uint64_t>(i)*kVocabulary),floats(kVocabulary)),arena_range(output_ids_,static_cast<VkDeviceSize>(i)*4u*sizeof(uint32_t),4u*sizeof(uint32_t)),arena_range(argmax_workspace_,static_cast<VkDeviceSize>(i)*256u*2u*sizeof(uint32_t),256u*2u*sizeof(uint32_t))});
    }

    void hc_pre(VkCommandBuffer c,VkDescriptorSet mix,VkDescriptorSet split,VkDescriptorSet pre,VkDescriptorSet norm){HcMixPush m{kDimension,kHcMultiplicity,24u,rms_bits()};kernels_.dispatch(c,kernels_.p().hc_mix,mix,&m,6u,kDsparkBlock);compute_barrier(c);HcSplitPush s{kDsparkBlock,kHcMultiplicity,20u,float_word(index_.header().hc_epsilon)};kernels_.dispatch(c,kernels_.p().hc_sinkhorn,split,&s,kDsparkBlock);compute_barrier(c);HcApplyPush a{kDimension,kHcMultiplicity,24u,0u};kernels_.dispatch(c,kernels_.p().hc_pre,pre,&a,divide_up(kDimension,64u),kDsparkBlock);compute_barrier(c);RmsPush r{kDsparkBlock,kDimension,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,norm,&r,kDsparkBlock);compute_barrier(c);}

    void record_pre(VkCommandBuffer c,uint32_t layer,uint32_t start){Layer& z=layers_[layer];hc_pre(c,z.hc_attn_mix,z.hc_attn_split,z.hc_attn_pre,z.attn_norm);uint32_t nc=kDsparkBlock*kDimension;quantize(c,z.quant_normalized,nc,128u,words(nc),words(nc));compute_barrier(c);gemv(c,z.wq_a,1024u,kDimension,kDsparkBlock,words(nc));gemv(c,z.wkv,kHeadDimension,kDimension,kDsparkBlock,words(nc));if(z.ratio4){gemv(c,z.compressor_wkv,1024u,kDimension,kDsparkBlock,words(nc));gemv(c,z.compressor_gate,1024u,kDimension,kDsparkBlock,words(nc));}compute_barrier(c);
        if(z.ratio4){uint32_t slot=static_cast<uint32_t>(ratio_slot_[layer]);VkDeviceSize base=floats((static_cast<uint64_t>(slot)*128u+start)*1024u);for(uint32_t t=0;t<kDsparkBlock;++t){copy_compute_result(c,compressor_kv_batch_,floats(static_cast<uint64_t>(t)*1024u),main_program_.compressor_kv_history(),base+floats(static_cast<uint64_t>(t)*1024u),floats(1024u));copy_compute_result(c,compressor_score_batch_,floats(static_cast<uint64_t>(t)*1024u),main_program_.compressor_score_history(),base+floats(static_cast<uint64_t>(t)*1024u),floats(1024u));}}
        RmsPush qn{kDsparkBlock,1024u,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,z.q_norm,&qn,kDsparkBlock);RmsPush kn{kDsparkBlock,kHeadDimension,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,z.kv_norm,&kn,kDsparkBlock);compute_barrier(c);uint32_t qc=kDsparkBlock*1024u;quantize(c,z.quant_q_rank,qc,128u,words(qc),words(qc));compute_barrier(c);gemv(c,z.wq_b,kHeads*kHeadDimension,1024u,kDsparkBlock,words(qc));compute_barrier(c);RmsPush qh{kDsparkBlock*kHeads,kHeadDimension,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,z.q_head_norm,&qh,kDsparkBlock*kHeads);compute_barrier(c);RopePush qr{kHeads,kHeadDimension,kRopeDimension,start};kernels_.dispatch(c,kernels_.p().rope_batch,z.query_rope,&qr,divide_up(kHeads*kHeadDimension,64u),kDsparkBlock);RopePush kr{1u,kHeadDimension,kRopeDimension,start};kernels_.dispatch(c,kernels_.p().rope_batch,z.kv_rope,&kr,divide_up(kHeadDimension,64u),kDsparkBlock);compute_barrier(c);
        VkDeviceSize layer_base=floats(static_cast<uint64_t>(layer)*(kWindow+32u)*kHeadDimension);for(uint32_t t=0;t<kDsparkBlock;++t)copy_compute_result(c,kv_raw_,floats(static_cast<uint64_t>(t)*kHeadDimension),executor_.kv_cache(),layer_base+floats(static_cast<uint64_t>((start+t)%kWindow)*kHeadDimension),floats(kHeadDimension));
        if(z.ratio4){uint32_t groups=(start+kDsparkBlock)/4u;if(groups){CompressPush cp{groups,start+kDsparkBlock,kHeadDimension,4u};kernels_.dispatch(c,kernels_.p().compress_ratio4,z.compress,&cp,groups);compute_barrier(c);for(uint32_t g=0;g<groups;++g){RmsPush rn{1u,kHeadDimension,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,z.compressed_norm[g],&rn,1u);compute_barrier(c);RopePush rr{1u,kHeadDimension,kRopeDimension,g*4u};kernels_.dispatch(c,kernels_.p().rope_batch,z.compressed_rope_set,&rr,divide_up(kHeadDimension,64u),1u);compute_barrier(c);copy_compute_result(c,compressed_rope_,0,executor_.kv_cache(),layer_base+floats(static_cast<uint64_t>(kWindow+g)*kHeadDimension),floats(kHeadDimension));}}}
        AttentionPush ap{kHeads,kHeadDimension,160u,kDsparkBlock};kernels_.dispatch(c,kernels_.p().main_attention_batch,z.attention,&ap,kHeads,kDsparkBlock);compute_barrier(c);RopePush inv{kHeads,kHeadDimension,kRopeDimension,start|0x80000000u};kernels_.dispatch(c,kernels_.p().rope_batch,z.inverse_rope,&inv,divide_up(kHeads*kHeadDimension,64u),kDsparkBlock);compute_barrier(c);uint32_t cc=kDsparkBlock*kHeads*kHeadDimension;quantize(c,z.quant_context,cc,128u,words(cc),words(cc));compute_barrier(c);grouped(c,z.wo_a,8u,1024u,4096u,kDsparkBlock,words(cc));compute_barrier(c);uint32_t oc=kDsparkBlock*8192u;quantize(c,z.quant_o_rank,oc,128u,words(oc),words(oc));compute_barrier(c);gemv(c,z.wo_b,kDimension,8192u,kDsparkBlock,words(oc));compute_barrier(c);HcApplyPush ha{kDimension,kHcMultiplicity,24u,0u};kernels_.dispatch(c,kernels_.p().hc_post,z.hc_attn_post,&ha,divide_up(kHcMultiplicity*kDimension,64u),kDsparkBlock);compute_barrier(c);hc_pre(c,z.hc_ffn_mix,z.hc_ffn_split,z.hc_ffn_pre,z.ffn_norm);quantize(c,z.quant_normalized,nc,128u,words(nc),words(nc));compute_barrier(c);gemv(c,z.router,kExperts,kDimension,kDsparkBlock,words(nc));compute_barrier(c);RouterPush rp{kExperts,kTopK,float_word(index_.header().route_scale),kDsparkBlock};kernels_.dispatch(c,z.hash?kernels_.p().router_hash_batch:kernels_.p().router_batch,z.router_select,&rp,kDsparkBlock);compute_barrier(c);
    }

    void build_unique_expert_plan(const uint32_t* route_words,
                                  uint32_t acquired_unique){
        std::array<uint32_t,kExperts> group_for{};
        group_for.fill(UINT32_MAX);
        for(auto& group:unique_groups_)group={};
        unique_group_count_=0;
        auto* occurrences=static_cast<uint32_t*>(unique_occurrences_.mapped);
        std::fill(occurrences,occurrences+kDsparkBlock*kTopK*dsv4::kDsparkBlock,
                  UINT32_MAX);
        for(uint32_t token=0;token<kDsparkBlock;++token)
            for(uint32_t rank=0;rank<kTopK;++rank){
                const uint32_t expert=route_words[token*64u+rank];
                if(expert>=kExperts)
                    throw std::runtime_error("Unique verifier saw invalid authoritative route");
                uint32_t group=group_for[expert];
                if(group==UINT32_MAX){
                    group=unique_group_count_++;
                    group_for[expert]=group;
                    unique_groups_[group].expert=expert;
                    unique_groups_[group].slot=slots_[token][rank];
                }else if(unique_groups_[group].slot!=slots_[token][rank]){
                    throw std::runtime_error(
                        "Unique verifier acquisition duplicated one expert across slots");
                }
                UniqueExpertGroup& item=unique_groups_[group];
                if(item.occurrences>=dsv4::kDsparkBlock)
                    throw std::runtime_error("Unique verifier occurrence bound exceeded");
                occurrences[group*dsv4::kDsparkBlock+item.occurrences++]=
                    token|(rank<<8u);
            }
        flush_buffer(runtime_,unique_occurrences_);
        const uint64_t routed=static_cast<uint64_t>(kDsparkBlock)*kTopK;
        if(acquired_unique>unique_group_count_)
            throw std::runtime_error(
                "Unique verifier acquired more records than authoritative routes");
        ++unique_metrics_.authoritative_layers;
        unique_metrics_.routed_occurrences+=routed;
        unique_metrics_.unique_experts+=unique_group_count_;
        unique_metrics_.initial_resident_unique+=unique_group_count_-acquired_unique;
        unique_metrics_.acquired_unique+=acquired_unique;
        unique_metrics_.reused_occurrences+=routed-unique_group_count_;
        unique_metrics_.layers_with_reuse+=unique_group_count_<routed;
        unique_metrics_.maximum_unique_experts=std::max<uint64_t>(
            unique_metrics_.maximum_unique_experts,unique_group_count_);
    }
    VkDescriptorSet& unique_gate_set(Layer& layer,uint32_t group){
        return layer.expert_gate[group/kTopK][group%kTopK];
    }
    VkDescriptorSet& unique_down_set(Layer& layer,uint32_t group){
        return layer.expert_down[group/kTopK][group%kTopK];
    }
    void update_expert_sets(uint32_t layer){
        Layer& z=layers_[layer];
        if(unique_expert_verify_){
            for(uint32_t group=0;group<unique_group_count_;++group){
                DescriptorRange record=executor_.device_expert_record(
                    unique_groups_[group].slot);
                kernels_.update(unique_gate_set(z,group),1u,record);
                kernels_.update(unique_down_set(z,group),1u,record);
            }
            return;
        }
        for(uint32_t t=0;t<kDsparkBlock;++t)
            for(uint32_t r=0;r<kTopK;++r){
                DescriptorRange record=executor_.device_expert_record(slots_[t][r]);
                kernels_.update(z.expert_gate[t][r],1u,record);
                kernels_.update(z.expert_down[t][r],1u,record);
            }
    }
    void record_shared_pre(VkCommandBuffer c,uint32_t layer){Layer& z=layers_[layer];uint32_t nc=kDsparkBlock*kDimension;gemv(c,z.shared_w1,kMoeDimension,kDimension,kDsparkBlock,words(nc));gemv(c,z.shared_w3,kMoeDimension,kDimension,kDsparkBlock,words(nc));compute_barrier(c);SwigluPush sw{kMoeDimension,swiglu_bits(),kDsparkBlock,0u};kernels_.dispatch(c,kernels_.p().swiglu_batch,z.shared_swiglu,&sw,divide_up(kMoeDimension,64u),kDsparkBlock);compute_barrier(c);uint32_t sc=kDsparkBlock*kMoeDimension;quantize(c,z.quant_shared,sc,128u,words(sc),words(sc));compute_barrier(c);}
    void record_unique_routed_and_finish(VkCommandBuffer c,uint32_t layer){
        Layer& z=layers_[layer];
        if(batch_quant_fused_){
            QuantizePush q{kDimension,32u,1024u,1024u};
            kernels_.dispatch(c,kernels_.p().quantize_strided_batch,
                quant_expert_batch_,&q,divide_up(kDimension,32u),kDsparkBlock);
        }else for(uint32_t t=0;t<kDsparkBlock;++t){
            QuantizePush q{kDimension,32u,1024u,1024u};
            kernels_.dispatch(c,kernels_.p().quantize,z.quant_expert[t],&q,
                              divide_up(kDimension,32u));
        }
        compute_barrier(c);
        for(uint32_t group=0;group<unique_group_count_;++group){
            const ExpertPush p{group,1024u,swiglu_bits(),
                               unique_groups_[group].occurrences};
            kernels_.dispatch(c,kernels_.p().expert_gate_q4_unique,
                unique_gate_set(z,group),&p,divide_up(kMoeDimension,4u),
                unique_groups_[group].occurrences);
            ++unique_metrics_.gate_dispatches;
        }
        compute_barrier(c);
        if(batch_quant_fused_){
            QuantizePush q{kMoeDimension,32u,512u,512u};
            kernels_.dispatch(c,kernels_.p().quantize_strided_batch,
                quant_intermediate_batch_,&q,divide_up(kMoeDimension,32u),
                kDsparkBlock*kTopK);
        }else for(uint32_t t=0;t<kDsparkBlock;++t)
            for(uint32_t r=0;r<kTopK;++r){
                QuantizePush q{kMoeDimension,32u,512u,512u};
                kernels_.dispatch(c,kernels_.p().quantize,
                    z.quant_intermediate[t][r],&q,divide_up(kMoeDimension,32u));
            }
        compute_barrier(c);
        for(uint32_t group=0;group<unique_group_count_;++group){
            const ExpertPush p{group,512u,0u,
                               unique_groups_[group].occurrences};
            kernels_.dispatch(c,kernels_.p().expert_down_q4_unique,
                unique_down_set(z,group),&p,divide_up(kDimension,4u),
                unique_groups_[group].occurrences);
            ++unique_metrics_.down_dispatches;
        }
        compute_barrier(c);
        QuantizePush red{kDimension,kTopK,kDsparkBlock,0u};
        kernels_.dispatch(c,kernels_.p().reduce_batch,z.reduce,&red,
                          divide_up(kDimension,64u),kDsparkBlock);
        compute_barrier(c);
        uint32_t sc=kDsparkBlock*kMoeDimension;
        gemv(c,z.shared_w2,kDimension,kMoeDimension,kDsparkBlock,words(sc),true);
        compute_barrier(c);
        HcApplyPush hp{kDimension,kHcMultiplicity,24u,0u};
        kernels_.dispatch(c,kernels_.p().hc_post,z.hc_ffn_post,&hp,
                          divide_up(kHcMultiplicity*kDimension,64u),kDsparkBlock);
        compute_barrier(c);
        if(layer>=40u&&layer<=42u){
            VkDeviceSize block=floats(kHcMultiplicity*kDimension);
            for(uint32_t t=0;t<kDsparkBlock;++t)
                copy_compute_result(c,hidden_,static_cast<VkDeviceSize>(t)*block,
                    target_captures_,static_cast<VkDeviceSize>(t*3u+layer-40u)*block,
                    block);
        }
    }
    void record_routed_and_finish(VkCommandBuffer c,uint32_t layer){if(unique_expert_verify_){record_unique_routed_and_finish(c,layer);return;}Layer& z=layers_[layer];const VkPipeline gate=q4_experts_?(q4_swar_?kernels_.p().expert_gate_q4_swar:kernels_.p().expert_gate_q4):kernels_.p().expert_gate;const VkPipeline down=q4_experts_?(q4_swar_?kernels_.p().expert_down_q4_swar:kernels_.p().expert_down_q4):kernels_.p().expert_down;const uint32_t rows=q4_experts_?4u:8u;if(batch_quant_fused_){QuantizePush q{kDimension,32u,1024u,1024u};kernels_.dispatch(c,kernels_.p().quantize_strided_batch,quant_expert_batch_,&q,divide_up(kDimension,32u),kDsparkBlock);}else for(uint32_t t=0;t<kDsparkBlock;++t){QuantizePush q{kDimension,32u,1024u,1024u};kernels_.dispatch(c,kernels_.p().quantize,z.quant_expert[t],&q,divide_up(kDimension,32u));}compute_barrier(c);for(uint32_t t=0;t<kDsparkBlock;++t)for(uint32_t r=0;r<kTopK;++r){ExpertPush p{r,1024u,swiglu_bits(),0u};kernels_.dispatch(c,gate,z.expert_gate[t][r],&p,divide_up(kMoeDimension,rows));}compute_barrier(c);if(batch_quant_fused_){QuantizePush q{kMoeDimension,32u,512u,512u};kernels_.dispatch(c,kernels_.p().quantize_strided_batch,quant_intermediate_batch_,&q,divide_up(kMoeDimension,32u),kDsparkBlock*kTopK);}else for(uint32_t t=0;t<kDsparkBlock;++t)for(uint32_t r=0;r<kTopK;++r){QuantizePush q{kMoeDimension,32u,512u,512u};kernels_.dispatch(c,kernels_.p().quantize,z.quant_intermediate[t][r],&q,divide_up(kMoeDimension,32u));}compute_barrier(c);for(uint32_t t=0;t<kDsparkBlock;++t)for(uint32_t r=0;r<kTopK;++r){ExpertPush p{r,512u,0u,0u};kernels_.dispatch(c,down,z.expert_down[t][r],&p,divide_up(kDimension,rows));}compute_barrier(c);QuantizePush red{kDimension,kTopK,kDsparkBlock,0u};kernels_.dispatch(c,kernels_.p().reduce_batch,z.reduce,&red,divide_up(kDimension,64u),kDsparkBlock);compute_barrier(c);uint32_t sc=kDsparkBlock*kMoeDimension;gemv(c,z.shared_w2,kDimension,kMoeDimension,kDsparkBlock,words(sc),true);compute_barrier(c);HcApplyPush hp{kDimension,kHcMultiplicity,24u,0u};kernels_.dispatch(c,kernels_.p().hc_post,z.hc_ffn_post,&hp,divide_up(kHcMultiplicity*kDimension,64u),kDsparkBlock);compute_barrier(c);if(layer>=40u&&layer<=42u){VkDeviceSize block=floats(kHcMultiplicity*kDimension);for(uint32_t t=0;t<kDsparkBlock;++t)copy_compute_result(c,hidden_,static_cast<VkDeviceSize>(t)*block,target_captures_,static_cast<VkDeviceSize>(t*3u+layer-40u)*block,block);}}
    void record_final(VkCommandBuffer c){HcApplyPush h{kDimension,kHcMultiplicity,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().hc_head,hc_head_,&h,kDsparkBlock);compute_barrier(c);RmsPush n{kDsparkBlock,kDimension,rms_bits(),0u};kernels_.dispatch(c,kernels_.p().rmsnorm,norm_,&n,kDsparkBlock);compute_barrier(c);uint32_t count=kDsparkBlock*kDimension;quantize(c,quant_final_,count,128u,words(count),words(count));compute_barrier(c);gemv(c,head_,kVocabulary,kDimension,kDsparkBlock,words(count));compute_barrier(c);uint32_t groups=divide_up(kVocabulary,64u*8u);for(uint32_t i=0;i<kDsparkBlock;++i){ArgmaxPush a{kVocabulary,groups,0u,0u};kernels_.dispatch(c,kernels_.p().greedy_argmax,argmax_[i],&a,groups);compute_barrier(c);ArgmaxPush b{kVocabulary,groups,1u,0u};kernels_.dispatch(c,kernels_.p().greedy_argmax,argmax_[i],&b,1u);compute_barrier(c);}}
    void update_indices(uint32_t start,bool ratio4){auto* out=static_cast<uint32_t*>(indices_.mapped);for(uint32_t t=0;t<kDsparkBlock;++t){uint32_t pos=start+t,count=0,base=t*160u;uint32_t history=std::min<uint32_t>(pos+1u,kWindow),first=pos+1u-history;for(uint32_t i=0;i<history;++i)out[base+count++]=(first+i)%kWindow;if(ratio4)for(uint32_t g=0;g<(pos+1u)/4u;++g)out[base+count++]=kWindow+g;uint32_t sources=count;while(count<159u)out[base+count++]=UINT32_MAX;out[base+159u]=sources;}flush_buffer(runtime_,indices_);}

    const Runtime& runtime_;ExecutorScaffold& executor_;DeepSeekProgram& main_program_;const SharedIndex& index_;const ReadOnlyMapping& shared_file_;FiniteQueue compute_;DsparkKernels kernels_;
    Buffer hidden_{},hidden_alt_{},reduced_{},normalized_{},q_rank_{},q_rank_norm_{},query_{},kv_raw_{},kv_{},context_{},o_rank_{},router_logits_{},routing_{},ffn_{},accumulator_{},routed_reduced_{},dense_gate_{},dense_up_{},shared_intermediate_{},quantized_{},expert_quantized_{},intermediate_quantized_{},hc_mixes_{},hc_split_{},compressor_kv_batch_{},compressor_score_batch_{},compressed_tmp_{},compressed_rope_{},tokens_{},output_ids_{},indices_{},ones_{},hc_head_params_{},rope_cos_plain_{},rope_sin_plain_{},rope_cos_compressed_{},rope_sin_compressed_{},logits_{},argmax_workspace_{},target_captures_{},unique_occurrences_{};
    VkDescriptorSet embedding_{},hc_head_{},norm_{},quant_final_{},quant_expert_batch_{},quant_intermediate_batch_{};Linear head_{};std::array<VkDescriptorSet,kDsparkBlock> argmax_{};std::vector<Layer> layers_;std::array<int32_t,kLayers> ratio_slot_{};uint32_t ratio_layers_=0;GlobalExpertCache::BatchSlots slots_{};std::array<UniqueExpertGroup,kDsparkBlock*kTopK> unique_groups_{};uint32_t unique_group_count_=0;UniqueExpertMetrics unique_metrics_{};uint64_t passes_=0,cache_hits_=0,cache_misses_=0;double seconds_=0,pre_seconds_=0,acquire_seconds_=0,post_seconds_=0,final_seconds_=0;bool q4_experts_=false,q4_swar_=false,batch_quant_fused_=false,unique_expert_verify_=false;
};

} // namespace dsv4

int main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: amd_deepseek_v4_dspark.exe <runtime-directory> "
                         "[--init | <prompt> [new-tokens]]\n";
            return 2;
        }
        _putenv_s("DSV4_DSPARK_CAPTURE", "1");
        if (!std::getenv("DSV4_CACHE_SLOTS")) _putenv_s("DSV4_CACHE_SLOTS", "10");
        if (!std::getenv("DSV4_GLOBAL_DEVICE_CACHE")) _putenv_s("DSV4_GLOBAL_DEVICE_CACHE", "1");
        if (!std::getenv("DSV4_FORCE_CACHE_ADMISSION")) _putenv_s("DSV4_FORCE_CACHE_ADMISSION", "1");
        uint64_t requested_ram_gib = 0, main_ram_gib = 0;
        if (const char* configured = std::getenv("DSV4_RAM_GIB")) {
            requested_ram_gib = std::stoull(configured);
            if (requested_ram_gib < 3u || requested_ram_gib > 60u)
                throw std::runtime_error("DSpark DSV4_RAM_GIB must be in [3,60]");
            // The main streamer guarantees its own explicit allocations plus
            // cache do not exceed this reduced integral-GiB allowance.  The
            // remaining GiB covers DSpark's six 12.75-MiB direct-I/O staging
            // records and small verifier buffers, keeping the original user
            // budget honest without changing the retained m13 implementation.
            main_ram_gib = requested_ram_gib - 1u;
            const std::string reduced = std::to_string(main_ram_gib);
            _putenv_s("DSV4_RAM_GIB", reduced.c_str());
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file(
            dsv4::runtime_path(directory, "tokenizer.ovb").string());
        dsv4::Tokenizer tokenizer(tokenizer_file);
        const bool q8_shared = std::getenv("DSV4_Q8_SHARED") != nullptr;
        const char* shared_name = !q8_shared &&
            std::filesystem::exists(directory / "model-q4g64.ovs") ?
            "model-q4g64.ovs" : "model.ovs";
        dsv4::ReadOnlyMapping shared(dsv4::runtime_path(directory, shared_name).string());
        const bool q4_experts = std::getenv("DSV4_Q4_EXPERTS") != nullptr;
        const char* expert_name = q4_experts ? "experts-q4g64.ovx" : "experts.ovx";
        if (q4_experts && !std::filesystem::exists(directory / expert_name))
            throw std::runtime_error("DSV4_Q4_EXPERTS requested but experts-q4g64.ovx is absent");
        dsv4::ReadOnlyMapping experts(dsv4::runtime_path(directory, expert_name).string());
        dsv4::ReadOnlyMapping dspark_shared(dsv4::runtime_path(directory, "dspark.ovs").string());
        dsv4::ReadOnlyMapping dspark_experts(
            dsv4::runtime_path(directory, "dspark-experts.ovx").string());
        dsv4::SharedIndex shared_index(shared);
        dsv4::ExpertIndex expert_index(experts);
        if (expert_index.q4g64t() != q4_experts)
            throw std::runtime_error("DSpark main expert container/precision mismatch");
        dsv4::DsparkIndex dspark_index(dspark_shared);
        dsv4::DsparkExpertIndex dspark_expert_index(dspark_experts);

        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "main precision: Q4G64T shared / Q8 global+router / "
                  << (q4_experts ? "Q4G64T K64/BF16 experts" :
                                   "native E2M1 FP4 experts") << "\n"
                  << "DSpark: 3 stages, block 5, native E2M1 FP4 experts\n";
        const std::filesystem::path shader_directory =
            std::filesystem::absolute(argv[0]).parent_path();
        uint64_t draft_hits = 0, draft_misses = 0, draft_bytes = 0;
        double draft_acquire = 0.0, draft_seconds = 0.0, verification_seconds = 0.0;
        uint64_t draft_passes = 0, verification_passes = 0;
        uint64_t accepted_drafts = 0, compared_drafts = 0, accepted_run_total = 0;
        uint64_t timed_output_tokens = 0, speculative_output_tokens = 0;
        uint64_t final_cycle_outputs = 0, final_cycle_accepted_drafts = 0;
        double batch_pre = 0.0, batch_acquire = 0.0, batch_post = 0.0, batch_final = 0.0;
        uint64_t batch_passes = 0;
        uint64_t batch_hits = 0, batch_misses = 0, batch_direct_disk_bytes = 0;
        uint64_t main_decode_ssd_bytes = 0, main_decode_ram_bytes = 0;
        uint64_t main_decode_h2d_bytes = 0;
        double main_decode_acquisition = 0.0;
        bool unique_expert_verify = false;
        dsv4::MainBatchVerifier::UniqueExpertMetrics unique_expert_metrics{};
        std::vector<uint32_t> generated;
        double decode_seconds = 0.0;
        {
            dsv4::ExecutorScaffold executor(runtime, shared, shared_index, expert_index);
            dsv4::DeepSeekProgram main_program(runtime, executor, shared_index, shared,
                                                shader_directory);
            dsv4::DsparkProgram dspark(runtime, executor, shared_index, dspark_index,
                                        dspark_shared, dspark_expert_index, shader_directory);
            dsv4::MainBatchVerifier batch_verifier(runtime, executor, main_program,
                                                    shared_index, shared, shader_directory);
            std::cout << "main expert cache slots: " << executor.expert_cache_slots()
                      << ", DSpark expert cache slots: "
                      << dsv4::kDsparkStages * dsv4::kDsparkCacheSlots << "\n";
            if (std::strcmp(argv[2], "--init") != 0) {
                const uint32_t maximum = argc >= 4 ?
                    static_cast<uint32_t>(std::stoul(argv[3])) : 8u;
                const std::vector<uint32_t> prompt = tokenizer.chat_prompt(argv[2]);
                if (prompt.empty() || prompt.size() + maximum > dsv4::kShortContext)
                    throw std::runtime_error("DSpark prompt plus generation exceeds short-context cap");
                const auto run_main = [&](uint32_t token, uint32_t position) {
                    return executor.run_token(token, position,
                        [&](VkCommandBuffer command, uint32_t value, uint32_t item) {
                            main_program.record_embedding(command, value, item);
                        },
                        [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                            main_program.record_pre(command, layer, parity);
                        },
                        [&](VkCommandBuffer command, uint32_t layer, uint32_t parity) {
                            main_program.record_post(command, layer, parity);
                        },
                        [&](VkCommandBuffer command, uint32_t item) {
                            main_program.record_final(command, item);
                        });
                };
                uint32_t prediction = 0;
                for (uint32_t position = 0; position < prompt.size(); ++position) {
                    prediction = run_main(prompt[position], position);
                    dspark.prefill_main(position);
                }
                const uint64_t transfer_base = executor.expert_transfer_bytes();
                const uint64_t host_hits_base = executor.host_cache_hits();
                const uint64_t host_misses_base = executor.host_cache_misses();
                const uint64_t batch_disk_base = executor.batch_direct_disk_bytes();
                const double scalar_acquire_base = executor.acquire_seconds();
                uint32_t current = prediction;
                uint32_t current_position = static_cast<uint32_t>(prompt.size());
                const bool bridge_free = std::getenv("DSV4_DSPARK_BRIDGE_FREE") != nullptr;
                const bool adaptive_bridge =
                    std::getenv("DSV4_DSPARK_ADAPTIVE_BRIDGE") != nullptr;
                bool bridge_free_next = false;
                if (current != tokenizer.eos() && maximum != 0u) {
                    generated.push_back(current);
                    std::cout << tokenizer.decode_piece(current) << std::flush;
                }
                // The first displayed token was computed by the final prompt
                // pass.  Start decode timing only after emitting it, and count
                // only outputs whose main work occurs inside this interval.
                const auto decode_started = std::chrono::steady_clock::now();
                while (generated.size() < maximum && current != tokenizer.eos()) {
                    const size_t cycle_begin = generated.size();
                    bool cycle_drafted = false;
                    const bool bridge_free_cycle = bridge_free ||
                        (adaptive_bridge && bridge_free_next);
                    uint32_t known = current;
                    uint32_t processed_position = current_position - 1u;
                    if (!bridge_free_cycle) {
                        processed_position = current_position;
                        const auto verify_start = std::chrono::steady_clock::now();
                        known = run_main(current, current_position);
                        verification_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - verify_start).count();
                        ++verification_passes;
                        current = known;
                        ++current_position;
                        if (current == tokenizer.eos()) break;
                        generated.push_back(current);
                        ++timed_output_tokens;
                        std::cout << tokenizer.decode_piece(current) << std::flush;
                        if (generated.size() == maximum) break;
                    } else if (generated.size() + 1u == maximum) {
                        // A full draft plus two-wide verifier cannot amortize the
                        // final token.  Keep this bounded tail on the retained
                        // scalar path; all complete speculative cycles remain
                        // bridge-free.
                        const auto verify_start = std::chrono::steady_clock::now();
                        current = run_main(current, current_position);
                        verification_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - verify_start).count();
                        ++verification_passes;
                        ++current_position;
                        if (current != tokenizer.eos()) {
                            generated.push_back(current);
                            ++timed_output_tokens;
                            std::cout << tokenizer.decode_piece(current) << std::flush;
                        }
                        break;
                    }
                    const auto draft_tokens = dspark.draft(known, processed_position);
                    cycle_drafted = true;
                    if (draft_tokens[0] != known)
                        throw std::runtime_error("DSpark known-token contract failed");

                    uint64_t accepted_run = 0;
                    if (std::getenv("DSV4_SEQUENTIAL_VERIFY")) {
                        for (uint32_t candidate = 1; candidate <= dsv4::kDsparkBlock; ++candidate) {
                            const auto item_start = std::chrono::steady_clock::now();
                            const uint32_t exact = run_main(current, current_position);
                            verification_seconds += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - item_start).count();
                            ++verification_passes;
                            dspark.prefill_main(current_position);
                            ++compared_drafts;
                            const bool accepted = exact == draft_tokens[candidate];
                            if (accepted) { ++accepted_drafts; ++accepted_run; }
                            current = exact;
                            ++current_position;
                            if (current == tokenizer.eos()) break;
                            generated.push_back(current);
                            ++timed_output_tokens;
                            std::cout << tokenizer.decode_piece(current) << std::flush;
                            if (!accepted || generated.size() == maximum) break;
                        }
                    } else {
                        std::array<uint32_t, dsv4::MainBatchVerifier::kDsparkBlock> inputs{};
                        inputs[0] = known;
                        for (uint32_t i = 1; i < dsv4::MainBatchVerifier::kDsparkBlock; ++i)
                            inputs[i] = draft_tokens[i];
                        const uint32_t batch_start = current_position;
                        const auto item_start = std::chrono::steady_clock::now();
                        const auto exact = batch_verifier.verify(inputs, batch_start);
                        verification_seconds += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - item_start).count();
                        ++verification_passes;
                        for (uint32_t candidate = 1;
                             candidate <= dsv4::MainBatchVerifier::kDsparkBlock; ++candidate) {
                            ++compared_drafts;
                            const bool accepted = exact[candidate - 1u] == draft_tokens[candidate];
                            if (accepted) { ++accepted_drafts; ++accepted_run; }
                            current = exact[candidate - 1u];
                            ++current_position;
                            if (current == tokenizer.eos()) break;
                            generated.push_back(current);
                            ++timed_output_tokens;
                            std::cout << tokenizer.decode_piece(current) << std::flush;
                            if (!accepted || generated.size() == maximum) break;
                        }
                        const uint32_t correct_inputs = std::min<uint32_t>(
                            static_cast<uint32_t>(accepted_run + 1u),
                            dsv4::MainBatchVerifier::kDsparkBlock);
                        for (uint32_t i = 0; i < correct_inputs; ++i)
                            dspark.prefill_captured(batch_verifier.target_captures(),
                                                   i, batch_start + i);
                    }
                    accepted_run_total += accepted_run;
                    if (cycle_drafted) {
                        const uint64_t outputs = static_cast<uint64_t>(
                            generated.size() - cycle_begin);
                        speculative_output_tokens += outputs;
                        final_cycle_outputs = outputs;
                        final_cycle_accepted_drafts = accepted_run;
                    }
                    // A complete verifier-block acceptance proves that all
                    // captured target states follow the greedy path.  Only then may the
                    // next cycle consume the final verified token without a
                    // scalar re-anchor; any rejection restores the retained
                    // scalar bridge immediately.
                    bridge_free_next = accepted_run ==
                        dsv4::MainBatchVerifier::kDsparkBlock;
                }
                decode_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - decode_started).count();
                batch_direct_disk_bytes =
                    executor.batch_direct_disk_bytes() - batch_disk_base;
                main_decode_h2d_bytes =
                    executor.expert_transfer_bytes() - transfer_base;
                main_decode_ram_bytes =
                    (executor.host_cache_hits() - host_hits_base) *
                    dsv4::kExpertRecordBytes;
                main_decode_ssd_bytes =
                    (executor.host_cache_misses() - host_misses_base) *
                        dsv4::kExpertRecordBytes + batch_direct_disk_bytes;
                main_decode_acquisition =
                    executor.acquire_seconds() - scalar_acquire_base +
                    batch_verifier.acquire_seconds();
            } else {
                std::cout << "DSpark Vulkan resources initialized\n";
            }
            draft_hits = dspark.cache().hits();
            draft_misses = dspark.cache().misses();
            draft_bytes = dspark.cache().transfer_bytes();
            draft_acquire = dspark.cache().acquisition_seconds();
            draft_seconds = dspark.draft_seconds();
            draft_passes = dspark.draft_passes();
            batch_pre = batch_verifier.pre_seconds();
            batch_acquire = batch_verifier.acquire_seconds();
            batch_post = batch_verifier.post_seconds();
            batch_final = batch_verifier.final_seconds();
            batch_passes = batch_verifier.passes();
            batch_hits = batch_verifier.cache_hits();
            batch_misses = batch_verifier.cache_misses();
            unique_expert_verify = batch_verifier.unique_expert_verify();
            unique_expert_metrics = batch_verifier.unique_expert_metrics();
        }
        std::cout << "\ntoken ids:";
        for (uint32_t token : generated) std::cout << ' ' << token;
        const double acceptance = compared_drafts ?
            static_cast<double>(accepted_drafts) / compared_drafts : 0.0;
        const double accepted_per_cycle = draft_passes ?
            static_cast<double>(speculative_output_tokens) / draft_passes : 0.0;
        const double accepted_run_length = draft_passes ?
            static_cast<double>(accepted_run_total) / draft_passes : 0.0;
        const double accepted_per_main_pass = verification_passes ?
            static_cast<double>(timed_output_tokens) / verification_passes : 0.0;
        const double output_divisor = timed_output_tokens ?
            static_cast<double>(timed_output_tokens) : 1.0;
        const double bytes_to_gib = 1.0 / (1024.0 * 1024.0 * 1024.0);
        const uint64_t dspark_ssd_bytes = draft_misses * dsv4::kExpertRecordBytes;
        const uint64_t dspark_ram_bytes = 0;
        std::cout << "\neffective decode: " <<
            (decode_seconds > 0.0 ? timed_output_tokens / decode_seconds : 0.0) << " tok/s\n"
                  << "accepted timed output tokens: " << timed_output_tokens << "\n"
                  << "draft passes: " << draft_passes << "\n"
                  << "draft acceptance: " << accepted_drafts << '/' << compared_drafts
                  << " (" << acceptance * 100.0 << "%)\n"
                  << "accepted tokens per speculative cycle: " << accepted_per_cycle << "\n"
                  << "accepted draft run length: " << accepted_run_length << "\n"
                  << "final speculative cycle (outputs/accepted drafts): "
                  << final_cycle_outputs << '/' << final_cycle_accepted_drafts << "\n"
                  << "accepted tokens per main-model pass: " << accepted_per_main_pass << "\n"
                  << "main verifier passes: " << verification_passes << "\n"
                  << "draft / verification: " << draft_seconds << " / "
                  << verification_seconds << " s\n"
                  << "DSpark acquisition: " << draft_acquire << " s\n"
                  << "batched verification passes: " << batch_passes << "\n"
                  << "batched cache hits/misses: " << batch_hits << '/' << batch_misses << "\n"
                  << "batched direct-I/O disk traffic: "
                  << static_cast<double>(batch_direct_disk_bytes) /
                        (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "batched pre / acquire / post / final: " << batch_pre << " / "
                  << batch_acquire << " / " << batch_post << " / " << batch_final << " s\n"
                  << "DSpark cache hits/misses: " << draft_hits << '/' << draft_misses << "\n"
                  << "DSpark transfer: " << static_cast<double>(draft_bytes) /
                        (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "main verifier decode traffic (SSD/RAM/H2D): "
                  << main_decode_ssd_bytes * bytes_to_gib << " / "
                  << main_decode_ram_bytes * bytes_to_gib << " / "
                  << main_decode_h2d_bytes * bytes_to_gib << " GiB\n"
                  << "main verifier traffic per accepted output (SSD/RAM/H2D): "
                  << main_decode_ssd_bytes * bytes_to_gib / output_divisor << " / "
                  << main_decode_ram_bytes * bytes_to_gib / output_divisor << " / "
                  << main_decode_h2d_bytes * bytes_to_gib / output_divisor << " GiB\n"
                  << "main verifier decode acquisition: "
                  << main_decode_acquisition << " s ("
                  << main_decode_acquisition / output_divisor << " s/output)\n"
                  << "DSpark decode traffic (SSD/RAM/H2D): "
                  << dspark_ssd_bytes * bytes_to_gib << " / "
                  << dspark_ram_bytes * bytes_to_gib << " / "
                  << draft_bytes * bytes_to_gib << " GiB\n"
                  << "DSpark traffic per accepted output (SSD/RAM/H2D): "
                  << dspark_ssd_bytes * bytes_to_gib / output_divisor << " / "
                  << dspark_ram_bytes * bytes_to_gib / output_divisor << " / "
                  << draft_bytes * bytes_to_gib / output_divisor << " GiB\n"
                  << "DSpark decode acquisition: " << draft_acquire << " s ("
                  << draft_acquire / output_divisor << " s/output)\n"
                  << "strict host budget / main-tier allowance / DSpark staging: "
                  << requested_ram_gib << " / " << main_ram_gib << " / "
                  << static_cast<double>(dsv4::kDsparkHostStagingBytes) /
                        (1024.0 * 1024.0 * 1024.0) << " GiB\n"
                  << "DSpark expert direct I/O: "
                  << (requested_ram_gib ? "unbuffered+overlapped" : "mapped") << "\n"
                  << "baseline comparison: exact by greedy main-model verification\n"
                  << "peak Vulkan allocations: " << static_cast<double>(peak_vulkan_buffer_bytes) /
                        (1024.0 * 1024.0 * 1024.0) << " GiB\n";
        if (unique_expert_verify) {
            const auto& metrics = unique_expert_metrics;
            std::cout
                << "unique verifier routes (unique/occurrences/reused/layers): "
                << metrics.unique_experts << '/' << metrics.routed_occurrences << '/'
                << metrics.reused_occurrences << '/'
                << metrics.authoritative_layers << "\n"
                << "unique verifier initial-resident / acquired unique: "
                << metrics.initial_resident_unique << '/'
                << metrics.acquired_unique << "\n"
                << "unique verifier reuse layers / maximum unique: "
                << metrics.layers_with_reuse << '/'
                << metrics.maximum_unique_experts << "\n"
                << "unique verifier expert dispatches (gate/down): "
                << metrics.gate_dispatches << '/'
                << metrics.down_dispatches << "\n";
        }
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        runtime = {};
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "DeepSeek DSpark runtime error: " << error.what()
                  << " (active Vulkan allocations "
                  << static_cast<double>(active_vulkan_buffer_bytes) /
                         (1024.0 * 1024.0 * 1024.0) << " GiB)\n";
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}
