#define OVLLM_QWEN35_RUNTIME_ONLY
#include "m16_qwen35.cpp"

// Isolated Qwen3.5 MTP experiment.  The ordinary m16/ovllm path is unchanged.
// This first establishes the official draft model's real acceptance and cost;
// it is intentionally not presented as a retained speculative verifier.
namespace qwen35_mtp {

using namespace qwen35;

class MtpOneEngine {
public:
    MtpOneEngine(const Runtime& runtime, const SharedIndex& index,
                 const TensorDevice& embedding, DescriptorRange main_hidden,
                 const std::filesystem::path& expert_path,
                 const std::filesystem::path& shaders, QwenEngine& main,
                 uint32_t device_slots = 16)
        : runtime_(runtime), weights_(runtime, index), kernels_(runtime, shaders),
          compute_(runtime, runtime.queue), transfer_(runtime, runtime.secondary_queue),
          main_hidden_(main_hidden), main_(main), slots_(device_slots) {
        if (slots_ < kTopK || slots_ > 32)
            throw std::runtime_error("Qwen MTP device slots must be 8..32");
        load_experts(expert_path);
        allocate();
        make_rope();
        build_sets(embedding);
    }

    ~MtpOneEngine() {
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        if (expert_file_ != INVALID_HANDLE_VALUE) CloseHandle(expert_file_);
        for (Buffer* buffer : {&rope_, &kv_, &expert_arena_, &expert_outputs_,
                               &expert_quant_, &expert_intermediate_,
                               &router_logits_, &shared_expert_gate_,
                               &shared_output_, &shared_intermediate_,
                               &shared_up_, &shared_gate_, &quant_, &context_,
                               &value_, &key_, &qgate_, &normalized_, &hidden_,
                               &concat_, &embedding_, &routing_, &token_})
            destroy_buffer(runtime_, *buffer);
    }

    uint32_t process(uint32_t token, uint32_t position, bool project = true) {
        if (position >= kMaximumContext)
            throw std::runtime_error("Qwen MTP context cap reached");
        const auto started = std::chrono::steady_clock::now();
        *static_cast<uint32_t*>(token_.mapped) = token;
        flush_buffer(runtime_, token_);
        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            Push push{kVocabulary, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().embedding, embedding_set_,
                              &push, kDim / 64);
            compute_barrier(command);
            push = {kDim, float_bits(1e-6f), 0, 0};
            kernels_.dispatch(command, kernels_.p().mtp_fuse, fuse_set_, &push, 1);
            compute_barrier(command);
            push = {2 * kDim, 128, 2 * kDim / 4, 2 * kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, concat_quant_set_,
                              &push, (2 * kDim) / 128);
            compute_barrier(command);
            push = {kDim, 2 * kDim, 2 * kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, fc_set_, &push, kDim / 8);
            compute_barrier(command);

            push = {1, kDim, float_bits(1e-6f), 0};
            kernels_.dispatch(command, kernels_.p().rms, input_norm_set_, &push, 1);
            compute_barrier(command);
            push = {kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, hidden_quant_set_,
                              &push, kDim / 128);
            compute_barrier(command);
            push = {16384, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, qgate_set_, &push, 2048);
            push = {512, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, key_set_, &push, 64);
            kernels_.dispatch(command, kernels_.p().q4, value_set_, &push, 64);
            compute_barrier(command);
            push = {0, position, kAttentionHeads, kRopeDim / 2};
            kernels_.dispatch(command, kernels_.p().qk, qk_set_, &push,
                              kAttentionHeads);
            push = {0, position, 0, 0};
            kernels_.dispatch(command, kernels_.p().store_value, store_value_set_,
                              &push, (kKvHeads * kHeadDim + 63) / 64);
            compute_barrier(command);
            push = {0, position, kAttentionHeads, 0};
            kernels_.dispatch(command, kernels_.p().attention, attention_set_,
                              &push, kAttentionHeads);
            compute_barrier(command);
            push = {kLinearValue, kAttentionHeads, 0, 0};
            kernels_.dispatch(command, kernels_.p().head_gate, head_gate_set_,
                              &push, kLinearValue / 64);
            compute_barrier(command);
            push = {kLinearValue, 128, kLinearValue / 4, kLinearValue / 4};
            kernels_.dispatch(command, kernels_.p().quant, context_quant_set_,
                              &push, kLinearValue / 128);
            compute_barrier(command);
            push = {kDim, kLinearValue, kLinearValue / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4_residual,
                              attention_out_set_, &push, kDim / 8);
            compute_barrier(command);

            push = {1, kDim, float_bits(1e-6f), 0};
            kernels_.dispatch(command, kernels_.p().rms, post_norm_set_, &push, 1);
            compute_barrier(command);
            push = {kDim, 128, kDim / 4, kDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, hidden_quant_set_,
                              &push, kDim / 128);
            compute_barrier(command);
            push = {kExperts, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q8, router_gemv_set_, &push,
                              kExperts / 8);
            compute_barrier(command);
            push = {kExperts, kTopK, 0, 0};
            kernels_.dispatch(command, kernels_.p().router, router_set_, &push, 1);
            compute_barrier(command);

            push = {kMoeDim, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, shared_gate_set_, &push,
                              kMoeDim / 8);
            kernels_.dispatch(command, kernels_.p().q4, shared_up_set_, &push,
                              kMoeDim / 8);
            push = {1, kDim, kDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4,
                              shared_expert_gate_set_, &push, 1);
            compute_barrier(command);
            push = {kMoeDim, float_bits(3.402823466e+38f), 0, 0};
            kernels_.dispatch(command, kernels_.p().swiglu, shared_swiglu_set_,
                              &push, kMoeDim / 64);
            compute_barrier(command);
            push = {kMoeDim, 128, kMoeDim / 4, kMoeDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, shared_quant_set_,
                              &push, kMoeDim / 128);
            compute_barrier(command);
            push = {kDim, kMoeDim, kMoeDim / 4, 0};
            kernels_.dispatch(command, kernels_.p().q4, shared_down_set_, &push,
                              kDim / 8);
            compute_barrier(command);
        });
        compute_.wait(signal);

        invalidate_buffer(runtime_, routing_);
        std::array<uint32_t, kTopK> experts{};
        const uint32_t* routes = static_cast<const uint32_t*>(routing_.mapped);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            experts[rank] = routes[rank];
            if (experts[rank] >= kExperts)
                throw std::runtime_error("Qwen MTP router returned invalid expert");
        }
        const std::array<bool, kTopK> misses = resolve(experts);
        std::vector<uint32_t> copied;
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (!misses[rank]) continue;
            copied.push_back(rank);
        }
        read_missing(experts, copied);
        for (uint32_t rank : copied)
            dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                     kExpertRecordBytes);
        uint64_t ready = 0;
        if (!copied.empty()) {
            ready = transfer_.submit([&](VkCommandBuffer command) {
                for (uint32_t rank : copied) {
                    const VkBufferCopy copy{0,
                        uint64_t(selected_slots_[rank]) * kExpertRecordBytes,
                        kExpertRecordBytes};
                    vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                                        expert_arena_.handle, 1, &copy);
                }
                dsv4::transfer_barrier(command, expert_arena_);
            });
            transfer_bytes_ += uint64_t(copied.size()) * kExpertRecordBytes;
        }
        signal = compute_.submit([&](VkCommandBuffer command) {
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                Push push{rank, kDim / 4, 0, 0};
                kernels_.dispatch(command, kernels_.p().expert_gate,
                                  expert_gate_sets_[selected_slots_[rank]], &push,
                                  kMoeDim / 8);
            }
            compute_barrier(command);
            Push push{kTopK * kMoeDim, 128, kTopK * kMoeDim / 4,
                      kTopK * kMoeDim / 4};
            kernels_.dispatch(command, kernels_.p().quant, expert_quant_set_,
                              &push, kTopK * kMoeDim / 128);
            compute_barrier(command);
            for (uint32_t rank = 0; rank < kTopK; ++rank) {
                push = {rank, kTopK * kMoeDim / 4, 0, 0};
                kernels_.dispatch(command, kernels_.p().expert_down,
                                  expert_down_sets_[selected_slots_[rank]], &push,
                                  kDim / 8);
            }
            compute_barrier(command);
            push = {kDim, kTopK, 0, 0};
            kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                              kDim / 64);
            compute_barrier(command);
            if (project) {
                push = {1, kDim, float_bits(1e-6f), 0};
                kernels_.dispatch(command, kernels_.p().rms, final_norm_set_,
                                  &push, 1);
                compute_barrier(command);
            }
        }, ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        compute_.wait(signal);
        seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        ++passes_;
        if (!project) return 0;
        return main_.project_experiment_normalized(whole(normalized_));
    }

    void reset_metrics() {
        seconds_ = 0;
        passes_ = hits_ = misses_ = transfer_bytes_ = disk_bytes_ = 0;
    }
    double seconds() const { return seconds_; }
    uint64_t passes() const { return passes_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
    uint64_t disk_bytes() const { return disk_bytes_; }
    uint64_t host_bytes() const {
        return uint64_t(kTopK) * kExpertRecordBytes;
    }
    uint64_t device_bytes() const {
        return weights_.device_bytes() + activation_device_bytes_;
    }

private:
    struct Entry { int32_t expert = -1; uint64_t age = 0; };

    Buffer device(uint64_t bytes) {
        Buffer buffer = create_device_buffer(runtime_, bytes);
        activation_device_bytes_ += buffer.allocation_size;
        return buffer;
    }

    TensorDevice tensor(const std::string& name, TensorFormat format,
                        uint64_t first, uint64_t second = 0) const {
        TensorDevice value = weights_.tensor(name);
        const uint32_t rank = second ? 2u : 1u;
        if (value.format != format || value.rank != rank ||
            value.shape[0] != first || (second && value.shape[1] != second))
            throw std::runtime_error("Unexpected Qwen MTP tensor ABI: " + name);
        return value;
    }

    VkDescriptorSet q4(DescriptorRange activation, const TensorDevice& weight,
                       DescriptorRange output, DescriptorRange residual = {}) {
        return kernels_.set({activation, weight.data, weight.auxiliary, output,
                             residual.buffer ? residual : kernels_.dummy()});
    }

    void load_experts(const std::filesystem::path& path) {
        const uint64_t bytes = uint64_t(kExperts) * kExpertRecordBytes;
        if (std::filesystem::file_size(path) != kHeaderBytes + bytes)
            throw std::runtime_error("Invalid Qwen MTP expert file size");
        std::ifstream check(path, std::ios::binary);
        ExpertHeader header{};
        read_at(check, 0, &header, sizeof(header), "MTP expert header");
        if (std::memcmp(header.magic, "OQ35MEX\0", 8) ||
            header.layers != 1 || header.experts != kExperts ||
            header.record_bytes != kExpertRecordBytes)
            throw std::runtime_error("Invalid Qwen MTP expert header");
        expert_file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
            nullptr, OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS |
                FILE_FLAG_OVERLAPPED,
            nullptr);
        if (expert_file_ == INVALID_HANDLE_VALUE)
            throw std::runtime_error("Could not open Qwen MTP expert file");
    }

    void read_missing(const std::array<uint32_t, kTopK>& experts,
                      const std::vector<uint32_t>& ranks) {
        if (ranks.empty()) return;
        std::array<OVERLAPPED, kTopK> operations{};
        std::array<HANDLE, kTopK> events{};
        events.fill(nullptr);
        try {
            for (uint32_t index = 0; index < ranks.size(); ++index) {
                const uint32_t rank = ranks[index];
                const uint64_t offset = kHeaderBytes +
                    uint64_t(experts[rank]) * kExpertRecordBytes;
                events[index] = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (!events[index])
                    throw std::runtime_error("Qwen MTP I/O event failed");
                operations[index].Offset = DWORD(offset);
                operations[index].OffsetHigh = DWORD(offset >> 32);
                operations[index].hEvent = events[index];
                const BOOL started = ReadFile(expert_file_, staging_[rank].mapped,
                    DWORD(kExpertRecordBytes), nullptr, &operations[index]);
                if (!started && GetLastError() != ERROR_IO_PENDING)
                    throw std::runtime_error("Qwen MTP expert read submit failed");
            }
            const DWORD waited = WaitForMultipleObjects(
                DWORD(ranks.size()), events.data(), TRUE, 10000);
            if (waited != WAIT_OBJECT_0)
                throw std::runtime_error("Qwen MTP expert read timed out");
            for (uint32_t index = 0; index < ranks.size(); ++index) {
                DWORD transferred = 0;
                if (!GetOverlappedResult(expert_file_, &operations[index],
                                         &transferred, FALSE) ||
                    transferred != kExpertRecordBytes)
                    throw std::runtime_error("Qwen MTP expert read failed");
            }
        } catch (...) {
            for (uint32_t index = 0; index < ranks.size(); ++index) {
                if (events[index]) {
                    CancelIoEx(expert_file_, &operations[index]);
                    CloseHandle(events[index]);
                }
            }
            throw;
        }
        for (uint32_t index = 0; index < ranks.size(); ++index)
            CloseHandle(events[index]);
        disk_bytes_ += uint64_t(ranks.size()) * kExpertRecordBytes;
    }

    void allocate() {
        token_ = create_buffer(runtime_, 4);
        routing_ = create_buffer(runtime_, 16u * sizeof(uint32_t));
        embedding_ = device(kDim * 4ull);
        concat_ = device(2ull * kDim * 4);
        hidden_ = device(kDim * 4ull);
        normalized_ = device(kDim * 4ull);
        qgate_ = device(16384ull * 4);
        key_ = device(512ull * 4);
        value_ = device(512ull * 4);
        context_ = device(kLinearValue * 4ull);
        quant_ = device(4224ull * 4);
        shared_gate_ = device(kMoeDim * 4ull);
        shared_up_ = device(kMoeDim * 4ull);
        shared_intermediate_ = device(kMoeDim * 4ull);
        shared_output_ = device(kDim * 4ull);
        shared_expert_gate_ = device(4);
        router_logits_ = device(kExperts * 4ull);
        expert_intermediate_ = device(uint64_t(kTopK) * kMoeDim * 4);
        expert_quant_ = device(2112ull * 4);
        expert_outputs_ = device(uint64_t(kTopK) * kDim * 4);
        kv_ = device(uint64_t(2) * kMaximumContext * kKvHeads * kHeadDim * 4);
        rope_ = device(uint64_t(kMaximumContext) * kRopeDim * 4);
        expert_arena_ = device(uint64_t(slots_) * kExpertRecordBytes);
        entries_.resize(slots_);
        staging_.resize(kTopK);
        for (Buffer& buffer : staging_)
            buffer = dsv4::create_host_buffer_uninitialized(runtime_,
                                                             kExpertRecordBytes);
    }

    void make_rope() {
        std::vector<float> table(uint64_t(kMaximumContext) * kRopeDim);
        for (uint32_t position = 0; position < kMaximumContext; ++position) {
            for (uint32_t pair = 0; pair < kRopeDim / 2; ++pair) {
                const double inverse = std::pow(10000000.0,
                    -2.0 * double(pair) / double(kRopeDim));
                const double angle = double(position) * inverse;
                table[uint64_t(position) * kRopeDim + pair] = float(std::cos(angle));
                table[uint64_t(position) * kRopeDim + kRopeDim / 2 + pair] =
                    float(std::sin(angle));
            }
        }
        Buffer staging = dsv4::create_host_buffer_uninitialized(
            runtime_, table.size() * sizeof(float));
        std::memcpy(staging.mapped, table.data(), table.size() * sizeof(float));
        dsv4::flush_buffer_range(runtime_, staging, 0,
                                 table.size() * sizeof(float));
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            const VkBufferCopy copy{0, 0, table.size() * sizeof(float)};
            vkfn::CmdCopyBuffer(command, staging.handle, rope_.handle, 1, &copy);
            dsv4::transfer_barrier(command, rope_);
        });
        compute_.wait(signal);
        destroy_buffer(runtime_, staging);
    }

    void build_sets(const TensorDevice& embedding) {
        embedding_set_ = kernels_.set({embedding.data, embedding.auxiliary,
                                       whole(token_), whole(embedding_)});
        const TensorDevice enorm = tensor("pre_embedding_norm", TensorFormat::f32,
                                          kDim);
        const TensorDevice hnorm = tensor("pre_hidden_norm", TensorFormat::f32,
                                          kDim);
        fuse_set_ = kernels_.set({whole(embedding_), main_hidden_, enorm.data,
                                  hnorm.data, whole(concat_)});
        concat_quant_set_ = kernels_.set({whole(concat_), whole(quant_)});
        fc_set_ = q4(whole(quant_), tensor("fc", TensorFormat::q4g64t,
                                          kDim, 2 * kDim), whole(hidden_));
        input_norm_set_ = kernels_.set({whole(hidden_),
            tensor("layers.0.input_norm", TensorFormat::f32, kDim).data,
            whole(normalized_)});
        hidden_quant_set_ = kernels_.set({whole(normalized_), whole(quant_)});
        qgate_set_ = q4(whole(quant_), tensor("layers.0.q_proj",
            TensorFormat::q4g64t, 16384, kDim), whole(qgate_));
        key_set_ = q4(whole(quant_), tensor("layers.0.k_proj",
            TensorFormat::q4g64t, 512, kDim), whole(key_));
        value_set_ = q4(whole(quant_), tensor("layers.0.v_proj",
            TensorFormat::q4g64t, 512, kDim), whole(value_));
        qk_set_ = kernels_.set({whole(qgate_), whole(key_),
            tensor("layers.0.q_norm", TensorFormat::f32, kHeadDim).data,
            tensor("layers.0.k_norm", TensorFormat::f32, kHeadDim).data,
            whole(kv_), whole(rope_)});
        store_value_set_ = kernels_.set({whole(value_), whole(kv_)});
        attention_set_ = kernels_.set({whole(qgate_), whole(kv_), whole(context_)});
        head_gate_set_ = kernels_.set({whole(context_), whole(qgate_)});
        context_quant_set_ = kernels_.set({whole(context_), whole(quant_)});
        attention_out_set_ = q4(whole(quant_), tensor("layers.0.o_proj",
            TensorFormat::q4g64t, kDim, kLinearValue), whole(hidden_),
            whole(hidden_));
        post_norm_set_ = kernels_.set({whole(hidden_),
            tensor("layers.0.post_norm", TensorFormat::f32, kDim).data,
            whole(normalized_)});
        router_gemv_set_ = kernels_.set({whole(quant_),
            tensor("layers.0.router", TensorFormat::q8_row, kExperts, kDim).data,
            tensor("layers.0.router", TensorFormat::q8_row, kExperts, kDim).auxiliary,
            whole(router_logits_)});
        router_set_ = kernels_.set({whole(router_logits_), whole(routing_)});
        shared_gate_set_ = q4(whole(quant_), tensor("layers.0.shared_gate_proj",
            TensorFormat::q4g64t, kMoeDim, kDim), whole(shared_gate_));
        shared_up_set_ = q4(whole(quant_), tensor("layers.0.shared_up_proj",
            TensorFormat::q4g64t, kMoeDim, kDim), whole(shared_up_));
        shared_swiglu_set_ = kernels_.set({whole(shared_gate_), whole(shared_up_),
                                          kernels_.dummy(),
                                          whole(shared_intermediate_)});
        shared_quant_set_ = kernels_.set({whole(shared_intermediate_),
                                          whole(expert_quant_)});
        shared_down_set_ = q4(whole(expert_quant_),
            tensor("layers.0.shared_down_proj", TensorFormat::q4g64t,
                   kDim, kMoeDim), whole(shared_output_));
        shared_expert_gate_set_ = q4(whole(quant_),
            tensor("layers.0.shared_expert_gate", TensorFormat::q4g64t,
                   1, kDim), whole(shared_expert_gate_));
        expert_quant_set_ = kernels_.set({whole(expert_intermediate_),
                                          whole(expert_quant_)});
        reduce_set_ = kernels_.set({whole(expert_outputs_), whole(shared_output_),
                                    whole(shared_expert_gate_), whole(hidden_)});
        final_norm_set_ = kernels_.set({whole(hidden_),
            tensor("final_norm", TensorFormat::f32, kDim).data,
            whole(normalized_)});
        expert_gate_sets_.resize(slots_);
        expert_down_sets_.resize(slots_);
        for (uint32_t slot = 0; slot < slots_; ++slot) {
            const DescriptorRange record = arena_range(
                expert_arena_, uint64_t(slot) * kExpertRecordBytes,
                kExpertRecordBytes);
            expert_gate_sets_[slot] = kernels_.set({whole(quant_), record,
                whole(routing_), whole(expert_intermediate_)});
            expert_down_sets_[slot] = kernels_.set({whole(expert_quant_), record,
                whole(routing_), whole(expert_outputs_)});
        }
    }

    std::array<bool, kTopK> resolve(
        const std::array<uint32_t, kTopK>& experts) {
        std::array<bool, kTopK> misses{};
        std::vector<bool> reserved(slots_);
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            ++frequency_[experts[rank]];
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (entries_[slot].expert != int32_t(experts[rank])) continue;
                selected_slots_[rank] = slot;
                reserved[slot] = true;
                entries_[slot].age = ++clock_;
                ++hits_;
                goto hit;
            }
            misses[rank] = true;
            ++misses_;
        hit:;
        }
        for (uint32_t rank = 0; rank < kTopK; ++rank) {
            if (!misses[rank]) continue;
            uint32_t victim = UINT32_MAX;
            for (uint32_t slot = 0; slot < slots_; ++slot) {
                if (reserved[slot]) continue;
                bool replace = victim == UINT32_MAX || entries_[slot].expert < 0;
                if (!replace && entries_[victim].expert >= 0) {
                    replace = frequency_[entries_[slot].expert] <
                                  frequency_[entries_[victim].expert] ||
                              (frequency_[entries_[slot].expert] ==
                                   frequency_[entries_[victim].expert] &&
                               entries_[slot].age < entries_[victim].age);
                }
                if (replace) { victim = slot; if (entries_[slot].expert < 0) break; }
            }
            if (victim == UINT32_MAX)
                throw std::runtime_error("No Qwen MTP expert victim");
            entries_[victim].expert = int32_t(experts[rank]);
            entries_[victim].age = ++clock_;
            reserved[victim] = true;
            selected_slots_[rank] = victim;
        }
        return misses;
    }

    const Runtime& runtime_;
    DeviceWeights weights_;
    Kernels kernels_;
    dsv4::FiniteQueue compute_, transfer_;
    DescriptorRange main_hidden_{};
    QwenEngine& main_;
    uint32_t slots_ = 0;
    HANDLE expert_file_ = INVALID_HANDLE_VALUE;
    uint64_t activation_device_bytes_ = 0, clock_ = 0;
    uint64_t hits_ = 0, misses_ = 0, transfer_bytes_ = 0, disk_bytes_ = 0;
    uint64_t passes_ = 0;
    double seconds_ = 0;
    std::array<uint32_t, kExperts> frequency_{};
    std::vector<Entry> entries_;
    std::array<uint32_t, kTopK> selected_slots_{};
    std::vector<Buffer> staging_;
    Buffer token_{}, routing_{}, embedding_{}, concat_{}, hidden_{}, normalized_{};
    Buffer qgate_{}, key_{}, value_{}, context_{}, quant_{};
    Buffer shared_gate_{}, shared_up_{}, shared_intermediate_{}, shared_output_{};
    Buffer shared_expert_gate_{}, router_logits_{}, expert_intermediate_{};
    Buffer expert_quant_{}, expert_outputs_{}, kv_{}, rope_{}, expert_arena_{};
    VkDescriptorSet embedding_set_{}, fuse_set_{}, concat_quant_set_{}, fc_set_{};
    VkDescriptorSet input_norm_set_{}, hidden_quant_set_{}, qgate_set_{};
    VkDescriptorSet key_set_{}, value_set_{}, qk_set_{}, store_value_set_{};
    VkDescriptorSet attention_set_{}, head_gate_set_{}, context_quant_set_{};
    VkDescriptorSet attention_out_set_{}, post_norm_set_{}, router_gemv_set_{};
    VkDescriptorSet router_set_{}, shared_gate_set_{}, shared_up_set_{};
    VkDescriptorSet shared_swiglu_set_{}, shared_quant_set_{}, shared_down_set_{};
    VkDescriptorSet shared_expert_gate_set_{}, expert_quant_set_{}, reduce_set_{};
    VkDescriptorSet final_norm_set_{};
    std::vector<VkDescriptorSet> expert_gate_sets_, expert_down_sets_;
};

} // namespace qwen35_mtp

int main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: amd_qwen35_mtp.exe <runtime-dir> <prompt> [tokens]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file((directory / "tokenizer.ovb").string());
        qwen35::Tokenizer tokenizer(tokenizer_file);
        qwen35::SharedIndex main_index(directory / "model-q4g64.ovs");
        qwen35::SharedIndex mtp_index(directory / "mtp-q4g64.ovs", true);
        runtime = create_runtime();
        const uint32_t count = argc >= 4 ? uint32_t(std::stoul(argv[3])) : 8u;
        constexpr double kMainRamGiB = 16.0;
        const uint64_t main_budget = uint64_t(kMainRamGiB * double(1ull << 30));
        const uint32_t slots = qwen35::device_slots();
        _putenv_s("QWEN_VERIFY2_EXPERIMENT", "1");
        const std::filesystem::path shaders =
            std::filesystem::absolute(argv[0]).parent_path();
        std::vector<uint32_t> result;
        uint64_t accepted = 0, compared = 0;
        double decode = 0, draft_seconds = 0;
        uint64_t main_disk = 0, main_h2d = 0, main_host_copy = 0;
        double main_acquire = 0;
        uint64_t mtp_h2d = 0, mtp_disk = 0, mtp_hits = 0, mtp_misses = 0;
        uint64_t controlled_host = 0, device_bytes = 0;
        {
            qwen35::QwenEngine main(runtime, main_index,
                directory / "experts-q4g64.ovx", shaders, main_budget, slots);
            qwen35_mtp::MtpOneEngine mtp(runtime, mtp_index,
                main.embedding_experiment_tensor(), main.hidden_experiment_range(),
                directory / "mtp-experts-q4g64.ovx", shaders, main, 16);
            const std::vector<uint32_t> prompt =
                tokenizer.chat_prompt(argv[2], false);
            uint32_t position = 0, next = 0;
            for (uint32_t i = 0; i < prompt.size(); ++i) {
                next = main.process_experiment_token(prompt[i], position++);
                if (i + 1 < prompt.size())
                    mtp.process(prompt[i + 1], position, false);
            }
            main.reset_experiment_metrics();
            mtp.reset_metrics();
            const auto started = std::chrono::steady_clock::now();
            if (count && !tokenizer.is_eos(next)) {
                result.push_back(next);
                std::cout << tokenizer.decode_piece(next) << std::flush;
            }
            uint64_t main_passes = 0;
            while (result.size() < count && !tokenizer.is_eos(next)) {
                const uint32_t draft = mtp.process(next, position, true);
                const std::array<uint32_t, 2> exact =
                    main.verify2_experiment({next, draft}, position);
                ++main_passes;
                ++compared;
                const bool matched = draft == exact[0];
                if (matched) ++accepted;
                main.accept_verify2_experiment(matched);
                ++position;
                next = exact[0];
                if (tokenizer.is_eos(next)) break;
                result.push_back(next);
                std::cout << tokenizer.decode_piece(next) << std::flush;
                if (result.size() >= count) break;
                if (matched) {
                    // Advance the draft KV with the accepted candidate.  Its
                    // prediction is not needed because exact[1] is already
                    // authoritative and becomes the next cycle's input.
                    mtp.process(next, position, false);
                    ++position;
                    next = exact[1];
                    if (tokenizer.is_eos(next)) break;
                    result.push_back(next);
                    std::cout << tokenizer.decode_piece(next) << std::flush;
                }
            }
            decode = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            draft_seconds = mtp.seconds();
            main_disk = main.disk_bytes();
            main_h2d = main.transfer_bytes();
            main_host_copy = main.host_copy_bytes();
            main_acquire = main.acquisition_seconds();
            mtp_h2d = mtp.transfer_bytes();
            mtp_disk = mtp.disk_bytes();
            mtp_hits = mtp.hits();
            mtp_misses = mtp.misses();
            controlled_host = main.ram_bytes() + mtp.host_bytes();
            device_bytes = main.vram_bytes() + mtp.device_bytes();
        }
        const uint64_t timed = result.size() > 1 ? result.size() - 1 : 0;
        const double divisor = timed ? double(timed) : 1.0;
        const double accepted_per_main = compared ? double(timed) / compared : 0.0;
        std::cout << "\ntoken ids:";
        for (uint32_t token : result) std::cout << ' ' << token;
        std::cout << "\neffective decode throughput: "
                  << (decode > 0 ? timed / decode : 0.0) << " tok/s\n"
                  << "MTP draft acceptance: " << accepted << '/' << compared
                  << " (" << (compared ? 100.0 * accepted / compared : 0.0)
                  << "%)\naccepted tokens per main-model pass: "
                  << accepted_per_main << "\n"
                  << "draft / main verification seconds: " << draft_seconds
                  << " / " << (decode - draft_seconds) << "\n"
                  << "main acquisition ms/output: "
                  << 1000.0 * main_acquire / divisor << "\n"
                  << "main SSD / host-copy / H2D bytes/output: "
                  << main_disk / divisor << " / " << main_host_copy / divisor
                  << " / " << main_h2d / divisor << "\n"
                  << "MTP cache hits/misses and H2D bytes/output: "
                  << mtp_hits << '/' << mtp_misses << " / "
                  << mtp_h2d / divisor << " (SSD " << mtp_disk / divisor
                  << ")\n"
                  << "controlled host / device allocation GiB: "
                  << double(controlled_host) / double(1ull << 30) << " / "
                  << double(device_bytes) / double(1ull << 30) << '\n';
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen MTP experiment error: " << error.what() << '\n';
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}
