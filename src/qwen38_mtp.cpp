#define OVLLM_QWEN38_RUNTIME_ONLY
#include "qwen38_flash_next.cpp"

// Native one-layer Qwen3.8-Flash-Next MTP experiment.  This is isolated from
// the ordinary executor until its acceptance and exact verifier are proven.
namespace qwen38_mtp {

using namespace qwen38;

constexpr uint64_t kMtpExpertRecordBytes = 1'998'848;

class MtpOneEngine {
public:
    enum class HiddenSource { zero, main, chained };

    MtpOneEngine(const Runtime& runtime, const SharedIndex& index,
                 const TensorDevice& embedding, DescriptorRange main_hyper,
                 const std::filesystem::path& expert_path,
                 const std::filesystem::path& shaders, QwenEngine& main,
                 uint32_t device_slots)
        : runtime_(runtime), weights_(runtime, index),
          expert_file_(expert_path.string()), kernels_(runtime, shaders),
          compute_(runtime, runtime.queue),
          transfer_(runtime, runtime.secondary_queue),
          main_hyper_(main_hyper), main_(main), slots_(device_slots) {
        if (slots_ < kTopK || slots_ > 64)
            throw std::runtime_error("Qwen3.8 MTP device slots must be 10..64");
        const char* active_text = std::getenv("QWEN38_MTP_ACTIVE_TOPK");
        if (!active_text)
            active_text = std::getenv("QWEN38_VERIFY4_ACTIVE_TOPK");
        if (active_text) {
            active_topk_ = static_cast<uint32_t>(std::stoul(active_text));
            if (active_topk_ < 1 || active_topk_ > kTopK)
                throw std::runtime_error(
                    "QWEN38_MTP_ACTIVE_TOPK must be between 1 and 10");
        }
        validate_experts();
        allocate();
        initialize_zero();
        make_rope();
        build_sets(embedding);
    }

    ~MtpOneEngine() {
        for (Buffer& buffer : staging_) destroy_buffer(runtime_, buffer);
        for (Buffer* buffer : {
                 &rope_, &kv_, &expert_arena_, &expert_outputs_,
                 &expert_quant_, &expert_intermediate_, &router_logits_,
                 &shared_expert_gate_, &shared_output_, &shared_intermediate_,
                 &shared_up_, &shared_gate_, &context_, &value_, &key_,
                 &qgate_, &block_output_, &hc_injection_, &hc_low_quant_,
                 &hc_low_, &hc_mix_weights_, &hc_normed_, &hidden_quant_,
                 &hidden_, &hyper_quant_, &embedding_quant_,
                 &fusion_embedding_, &hyper_norm_, &embedding_norm_,
                 &embedding_, &zero_hyper_, &hyper_, &routing_, &token_})
            destroy_buffer(runtime_, *buffer);
    }

    uint32_t process(uint32_t token, uint32_t position, HiddenSource source,
                     bool project = true) {
        if (position >= kMaximumContext)
            throw std::runtime_error("Qwen3.8 MTP context cap reached");
        const auto started = std::chrono::steady_clock::now();
        *static_cast<uint32_t*>(token_.mapped) = token;
        flush_buffer(runtime_, token_);

        const VkDescriptorSet hnorm = source == HiddenSource::zero
            ? pre_hidden_zero_set_
            : source == HiddenSource::main
                ? pre_hidden_main_set_ : pre_hidden_self_set_;
        uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            record_fusion(command, hnorm);
            record_hc_start(command, attn_hc_, true);
            record_attention(command, position);
            record_hc_apply(command, attn_hc_);
            record_hc_start(command, mlp_hc_, true);
            record_router_shared(command);
        });
        compute_.wait(signal);

        invalidate_buffer(runtime_, routing_);
        std::array<uint32_t, kTopK> experts{};
        const uint32_t* routes = static_cast<const uint32_t*>(routing_.mapped);
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
            experts[rank] = routes[rank];
            if (experts[rank] >= kExperts)
                throw std::runtime_error("Qwen3.8 MTP router returned invalid expert");
        }
        const std::array<bool, kTopK> misses = resolve(experts);
        std::vector<uint32_t> copied;
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
            if (!misses[rank]) continue;
            const uint64_t offset = kHeaderBytes +
                uint64_t(experts[rank]) * kMtpExpertRecordBytes;
            std::memcpy(staging_[rank].mapped,
                        expert_file_.data() + offset,
                        kMtpExpertRecordBytes);
            dsv4::flush_buffer_range(runtime_, staging_[rank], 0,
                                     kMtpExpertRecordBytes);
            copied.push_back(rank);
        }
        uint64_t ready = 0;
        if (!copied.empty()) {
            ready = transfer_.submit([&](VkCommandBuffer command) {
                for (uint32_t rank : copied) {
                    const VkBufferCopy copy{
                        0, uint64_t(selected_slots_[rank]) *
                               kMtpExpertRecordBytes,
                        kMtpExpertRecordBytes};
                    vkfn::CmdCopyBuffer(command, staging_[rank].handle,
                                        expert_arena_.handle, 1, &copy);
                }
                dsv4::transfer_barrier(command, expert_arena_);
            });
            transfer_bytes_ += uint64_t(copied.size()) *
                kMtpExpertRecordBytes;
            host_copy_bytes_ += uint64_t(copied.size()) *
                kMtpExpertRecordBytes;
        }
        signal = compute_.submit([&](VkCommandBuffer command) {
            record_experts(command);
            record_hc_apply(command, mlp_hc_);
            if (project) record_hc_start(command, final_hc_, false);
        }, ready ? transfer_.semaphore() : VK_NULL_HANDLE, ready,
           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        compute_.wait(signal);

        uint32_t result = 0;
        if (project)
            result = main_.project_experiment_hidden(whole(hidden_));
        seconds_ += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        ++passes_;
        return result;
    }

    void reset_metrics() {
        seconds_ = 0;
        passes_ = hits_ = misses_ = transfer_bytes_ = host_copy_bytes_ = 0;
    }
    double seconds() const { return seconds_; }
    uint64_t passes() const { return passes_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t transfer_bytes() const { return transfer_bytes_; }
    uint64_t host_copy_bytes() const { return host_copy_bytes_; }
    uint64_t device_bytes() const {
        return weights_.device_bytes() + activation_device_bytes_;
    }
    uint64_t host_bytes() const {
        return uint64_t(kTopK) * kMtpExpertRecordBytes;
    }

private:
    struct Entry { int32_t expert = -1; uint64_t age = 0; };
    struct HcSets {
        VkDescriptorSet norm{}, quant{}, down{}, inject{}, act{};
        VkDescriptorSet low_quant{}, up{}, mix{}, apply{};
    };

    Buffer device(uint64_t bytes) {
        Buffer result = create_device_buffer(runtime_, bytes);
        activation_device_bytes_ += result.allocation_size;
        return result;
    }

    TensorDevice tensor(const std::string& name, TensorFormat format,
                        uint64_t first, uint64_t second = 0) const {
        TensorDevice value = weights_.tensor(name);
        const uint32_t rank = second ? 2u : 1u;
        if (value.format != format || value.rank != rank ||
            value.shape[0] != first || (second && value.shape[1] != second))
            throw std::runtime_error("Unexpected Qwen3.8 MTP tensor ABI: " + name);
        return value;
    }

    VkPipeline q4_pipeline(uint32_t inner) const {
        if (std::getenv("QWEN38_Q4_ONE_LANE") && inner > kMoeDim)
            return kernels_.p().q4_one;
        return inner <= kMoeDim ? kernels_.p().q4_small : kernels_.p().q4;
    }

    VkDescriptorSet q4_set(DescriptorRange activation,
                           const TensorDevice& weight,
                           DescriptorRange output,
                           DescriptorRange residual = {}) {
        return kernels_.set({activation, weight.data, weight.auxiliary, output,
                             residual.buffer ? residual : kernels_.dummy()});
    }

    void validate_experts() {
        const uint64_t expected = kHeaderBytes +
            uint64_t(kExperts) * kMtpExpertRecordBytes;
        if (expert_file_.size() != expected)
            throw std::runtime_error("Invalid Qwen3.8 MTP Q3 expert file size");
        ExpertHeader header{};
        std::memcpy(&header, expert_file_.data(), sizeof(header));
        if (std::memcmp(header.magic, "OQ38MEX\0", 8) != 0 ||
            header.version != 1 || header.header_bytes != kHeaderBytes ||
            header.dimension != kDim || header.moe_dimension != kMoeDim ||
            header.layers != 1 || header.experts != kExperts ||
            header.record_bytes != kMtpExpertRecordBytes ||
            header.file_bytes != expected)
            throw std::runtime_error("Invalid Qwen3.8 MTP Q3 expert header");
    }

    void allocate() {
        token_ = create_buffer(runtime_, sizeof(uint32_t));
        routing_ = create_buffer(runtime_, (16u + 2u * kTopK) * sizeof(uint32_t));
        hyper_ = device(uint64_t(kHcDim) * 4);
        zero_hyper_ = device(uint64_t(kHcDim) * 4);
        embedding_ = device(uint64_t(kDim) * 4);
        embedding_norm_ = device(uint64_t(kDim) * 4);
        hyper_norm_ = device(uint64_t(kHcDim) * 4);
        fusion_embedding_ = device(uint64_t(kDim) * 4);
        embedding_quant_ = device(4096);
        hyper_quant_ = device(4u * 4096u);
        hidden_ = device(uint64_t(kDim) * 4);
        hidden_quant_ = device(4096);
        hc_normed_ = device(uint64_t(kHcDim) * 4);
        hc_mix_weights_ = device(uint64_t(kHcDim) * 4);
        hc_low_ = device(uint64_t(kHcLowrank) * 4);
        hc_low_quant_ = device(4096);
        hc_injection_ = device(uint64_t(kHcCount) * 4);
        block_output_ = device(uint64_t(kDim) * 4);
        qgate_ = device(12288ull * 4);
        key_ = device(512ull * 4);
        value_ = device(512ull * 4);
        context_ = device(kLinearValue * 4ull);
        shared_gate_ = device(kMoeDim * 4ull);
        shared_up_ = device(kMoeDim * 4ull);
        shared_intermediate_ = device(kMoeDim * 4ull);
        shared_output_ = device(kDim * 4ull);
        shared_expert_gate_ = device(4);
        router_logits_ = device(kExperts * 4ull);
        expert_intermediate_ = device(uint64_t(kTopK) * kMoeDim * 4);
        expert_quant_ = device(1664ull * 4);
        expert_outputs_ = device(uint64_t(kTopK) * kDim * 4);
        kv_ = device(uint64_t(2) * kMaximumContext * kKvHeads * kHeadDim * 4);
        rope_ = device(uint64_t(kMaximumContext) * kRopeDim * 4);
        expert_arena_ = device(uint64_t(slots_) * kMtpExpertRecordBytes);
        entries_.resize(slots_);
        staging_.resize(kTopK);
        for (Buffer& buffer : staging_)
            buffer = dsv4::create_host_buffer_uninitialized(
                runtime_, kMtpExpertRecordBytes);
    }

    void initialize_zero() {
        Buffer staging = dsv4::create_host_buffer_uninitialized(
            runtime_, zero_hyper_.size + kv_.size);
        std::memset(staging.mapped, 0, static_cast<size_t>(staging.size));
        dsv4::flush_buffer_range(runtime_, staging, 0, staging.size);
        const uint64_t signal = compute_.submit([&](VkCommandBuffer command) {
            VkBufferCopy copies[2] = {
                {0, 0, zero_hyper_.size},
                {zero_hyper_.size, 0, kv_.size},
            };
            vkfn::CmdCopyBuffer(command, staging.handle, zero_hyper_.handle,
                                1, &copies[0]);
            vkfn::CmdCopyBuffer(command, staging.handle, kv_.handle,
                                1, &copies[1]);
            dsv4::transfer_barrier(command, zero_hyper_);
            dsv4::transfer_barrier(command, kv_);
        });
        compute_.wait(signal);
        destroy_buffer(runtime_, staging);
    }

    void make_rope() {
        std::vector<float> table(uint64_t(kMaximumContext) * kRopeDim);
        for (uint32_t position = 0; position < kMaximumContext; ++position) {
            for (uint32_t pair = 0; pair < kRopeDim / 2; ++pair) {
                const double inverse = std::pow(
                    10000000.0, -2.0 * double(pair) / double(kRopeDim));
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
            const VkBufferCopy copy{0, 0, table.size() * sizeof(float)};
            vkfn::CmdCopyBuffer(command, staging.handle, rope_.handle, 1, &copy);
            dsv4::transfer_barrier(command, rope_);
        });
        compute_.wait(signal);
        destroy_buffer(runtime_, staging);
    }

    HcSets build_hc_sets(const std::string& prefix, bool injection) {
        HcSets sets{};
        const TensorDevice norm = tensor(prefix + "norm", TensorFormat::f32,
                                          kHcDim);
        const TensorDevice down = tensor(prefix + "down", TensorFormat::q4g64t,
                                          kHcLowrank, kHcDim);
        const TensorDevice up = tensor(prefix + "up", TensorFormat::q4g64t,
                                        kHcDim, kHcLowrank);
        sets.norm = kernels_.set({whole(hyper_), norm.data, whole(hc_normed_)});
        sets.quant = kernels_.set({whole(hc_normed_), whole(hyper_quant_)});
        sets.down = q4_set(whole(hyper_quant_), down, whole(hc_low_));
        sets.act = kernels_.set({whole(hc_low_)});
        sets.low_quant = kernels_.set({whole(hc_low_), whole(hc_low_quant_)});
        sets.up = q4_set(whole(hc_low_quant_), up, whole(hc_mix_weights_));
        sets.mix = kernels_.set({whole(hc_normed_), whole(hc_mix_weights_),
                                 whole(hidden_)});
        if (injection) {
            const TensorDevice inject = tensor(prefix + "inject",
                TensorFormat::q4g64t, kHcCount, kHcDim);
            sets.inject = q4_set(whole(hyper_quant_), inject,
                                 whole(hc_injection_));
            sets.apply = kernels_.set({whole(block_output_),
                                       whole(hc_injection_), whole(hyper_)});
        }
        return sets;
    }

    void build_sets(const TensorDevice& embedding) {
        embedding_set_ = kernels_.set({embedding.data, embedding.auxiliary,
                                       whole(token_), whole(embedding_)});
        const TensorDevice enorm = tensor("pre_embedding_norm",
                                           TensorFormat::f32, kDim);
        const TensorDevice hnorm = tensor("pre_hidden_norm",
                                           TensorFormat::f32, kHcDim);
        pre_embedding_norm_set_ = kernels_.set(
            {whole(embedding_), enorm.data, whole(embedding_norm_)});
        pre_hidden_zero_set_ = kernels_.set(
            {whole(zero_hyper_), hnorm.data, whole(hyper_norm_)});
        pre_hidden_main_set_ = kernels_.set(
            {main_hyper_, hnorm.data, whole(hyper_norm_)});
        pre_hidden_self_set_ = kernels_.set(
            {whole(hyper_), hnorm.data, whole(hyper_norm_)});
        embedding_quant_set_ = kernels_.set(
            {whole(embedding_norm_), whole(embedding_quant_)});
        const TensorDevice fc_embedding = tensor(
            "fc_embedding", TensorFormat::q4g64t, kDim, kDim);
        const TensorDevice fc_hidden = tensor(
            "fc_hidden", TensorFormat::q4g64t, kDim, kDim);
        fc_embedding_set_ = q4_set(whole(embedding_quant_), fc_embedding,
                                   whole(fusion_embedding_));
        for (uint32_t branch = 0; branch < kHcCount; ++branch) {
            const DescriptorRange input = arena_range(
                hyper_norm_, uint64_t(branch) * kDim * 4, uint64_t(kDim) * 4);
            const DescriptorRange quant = arena_range(
                hyper_quant_, uint64_t(branch) * 4096, 4096);
            const DescriptorRange output = arena_range(
                hyper_, uint64_t(branch) * kDim * 4, uint64_t(kDim) * 4);
            hyper_branch_quant_sets_[branch] = kernels_.set({input, quant});
            fc_hidden_sets_[branch] = q4_set(
                quant, fc_hidden, output, whole(fusion_embedding_));
        }

        attn_hc_ = build_hc_sets("layers.0.attn_hc_", true);
        mlp_hc_ = build_hc_sets("layers.0.mlp_hc_", true);
        final_hc_ = build_hc_sets("final_hc_", false);
        hidden_quant_set_ = kernels_.set({whole(hidden_), whole(hidden_quant_)});

        const TensorDevice query = tensor("layers.0.q_proj",
            TensorFormat::q4g64t, 12288, kDim);
        const TensorDevice key = tensor("layers.0.k_proj",
            TensorFormat::q4g64t, 512, kDim);
        const TensorDevice value = tensor("layers.0.v_proj",
            TensorFormat::q4g64t, 512, kDim);
        const TensorDevice output = tensor("layers.0.o_proj",
            TensorFormat::q4g64t, kDim, kLinearValue);
        qgate_set_ = q4_set(whole(hidden_quant_), query, whole(qgate_));
        key_set_ = q4_set(whole(hidden_quant_), key, whole(key_));
        value_set_ = q4_set(whole(hidden_quant_), value, whole(value_));
        qk_set_ = kernels_.set({whole(qgate_), whole(key_),
            tensor("layers.0.q_norm", TensorFormat::f32, kHeadDim).data,
            tensor("layers.0.k_norm", TensorFormat::f32, kHeadDim).data,
            whole(kv_), whole(rope_)});
        store_value_set_ = kernels_.set({whole(value_), whole(kv_)});
        attention_set_ = kernels_.set({whole(qgate_), whole(kv_), whole(context_)});
        head_gate_set_ = kernels_.set({whole(context_), whole(qgate_)});
        context_quant_set_ = kernels_.set({whole(context_), whole(expert_quant_)});
        attention_out_set_ = q4_set(whole(expert_quant_), output,
                                     whole(block_output_));

        const TensorDevice router = tensor("layers.0.router",
            TensorFormat::q8_row, kExperts, kDim);
        router_gemv_set_ = kernels_.set({whole(hidden_quant_), router.data,
                                         router.auxiliary,
                                         whole(router_logits_)});
        router_set_ = kernels_.set({whole(router_logits_), whole(routing_)});
        shared_gate_set_ = q4_set(whole(hidden_quant_), tensor(
            "layers.0.shared_gate_proj", TensorFormat::q4g64t,
            kMoeDim, kDim), whole(shared_gate_));
        shared_up_set_ = q4_set(whole(hidden_quant_), tensor(
            "layers.0.shared_up_proj", TensorFormat::q4g64t,
            kMoeDim, kDim), whole(shared_up_));
        shared_swiglu_set_ = kernels_.set({whole(shared_gate_), whole(shared_up_),
                                           kernels_.dummy(),
                                           whole(shared_intermediate_)});
        shared_quant_set_ = kernels_.set({whole(shared_intermediate_),
                                           whole(expert_quant_)});
        shared_down_set_ = q4_set(whole(expert_quant_), tensor(
            "layers.0.shared_down_proj", TensorFormat::q4g64t,
            kDim, kMoeDim), whole(shared_output_));
        shared_expert_gate_set_ = q4_set(whole(hidden_quant_), tensor(
            "layers.0.shared_expert_gate", TensorFormat::q4g64t,
            1, kDim), whole(shared_expert_gate_));
        expert_quant_set_ = kernels_.set({whole(expert_intermediate_),
                                           whole(expert_quant_)});
        reduce_set_ = kernels_.set({whole(expert_outputs_), whole(shared_output_),
                                    whole(shared_expert_gate_),
                                    whole(block_output_)});
        expert_gate_sets_.resize(slots_);
        expert_down_sets_.resize(slots_);
        for (uint32_t slot = 0; slot < slots_; ++slot) {
            const DescriptorRange record = arena_range(
                expert_arena_, uint64_t(slot) * kMtpExpertRecordBytes,
                kMtpExpertRecordBytes);
            expert_gate_sets_[slot] = kernels_.set(
                {whole(hidden_quant_), record, whole(routing_),
                 whole(expert_intermediate_)});
            expert_down_sets_[slot] = kernels_.set(
                {whole(expert_quant_), record, whole(routing_),
                 whole(expert_outputs_)});
        }
    }

    void record_fusion(VkCommandBuffer command, VkDescriptorSet hnorm) {
        Push push{kVocabulary, kDim, kDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().embedding, embedding_set_,
                          &push, (kDim + 63) / 64);
        compute_barrier(command);
        push = {1, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().rms, pre_embedding_norm_set_,
                          &push, 1);
        push = {1, kHcDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().rms, hnorm, &push, 1);
        compute_barrier(command);
        push = {kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, embedding_quant_set_,
                          &push, kDim / 128);
        for (uint32_t branch = 0; branch < kHcCount; ++branch)
            kernels_.dispatch(command, kernels_.p().quant,
                              hyper_branch_quant_sets_[branch], &push,
                              kDim / 128);
        compute_barrier(command);
        push = {kDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kDim), fc_embedding_set_,
                          &push, kDim / 8);
        compute_barrier(command);
        for (uint32_t branch = 0; branch < kHcCount; ++branch)
            kernels_.dispatch(command, kernels_.p().q4_residual,
                              fc_hidden_sets_[branch], &push, kDim / 8);
        compute_barrier(command);
    }

    void record_hc_start(VkCommandBuffer command, HcSets& sets,
                         bool injection) {
        Push push{kHcCount, kDim, float_bits(1e-6f), 0};
        kernels_.dispatch(command, kernels_.p().group_rms, sets.norm, &push,
                          kHcCount);
        compute_barrier(command);
        push = {kHcDim, 128, kHcDim / 4, kHcDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.quant, &push,
                          kHcDim / 128);
        compute_barrier(command);
        push = {kHcLowrank, kHcDim, kHcDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kHcDim), sets.down, &push,
                          kHcLowrank / 8);
        if (injection) {
            push = {kHcCount, kHcDim, kHcDim / 4, 0};
            kernels_.dispatch(command, q4_pipeline(kHcDim), sets.inject,
                              &push, 1);
        }
        compute_barrier(command);
        push = {kHcLowrank, float_bits(float(kHcCount)), 0, 0};
        kernels_.dispatch(command, kernels_.p().hc_act, sets.act, &push,
                          (kHcLowrank + 63) / 64);
        compute_barrier(command);
        push = {kHcLowrank, 128, kHcLowrank / 4, kHcLowrank / 4};
        kernels_.dispatch(command, kernels_.p().quant, sets.low_quant, &push,
                          kHcLowrank / 128);
        compute_barrier(command);
        push = {kHcDim, kHcLowrank, kHcLowrank / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kHcLowrank), sets.up, &push,
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
                          kHcDim / 64);
        compute_barrier(command);
    }

    void record_attention(VkCommandBuffer command, uint32_t position) {
        Push push{kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, hidden_quant_set_, &push,
                          kDim / 128);
        compute_barrier(command);
        push = {12288, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kDim), qgate_set_, &push, 1536);
        push = {512, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kDim), key_set_, &push, 64);
        kernels_.dispatch(command, q4_pipeline(kDim), value_set_, &push, 64);
        compute_barrier(command);
        push = {0, position, kAttentionHeads, kRopeDim / 2};
        kernels_.dispatch(command, kernels_.p().qk, qk_set_, &push,
                          kAttentionHeads);
        push = {0, position, 0, 0};
        kernels_.dispatch(command, kernels_.p().store_value, store_value_set_,
                          &push, (kKvHeads * kHeadDim + 63) / 64);
        compute_barrier(command);
        push = {0, position, kAttentionHeads, 0};
        kernels_.dispatch(command, kernels_.p().attention, attention_set_, &push,
                          kAttentionHeads);
        compute_barrier(command);
        push = {kLinearValue, kAttentionHeads, 0, 0};
        kernels_.dispatch(command, kernels_.p().head_gate, head_gate_set_, &push,
                          kLinearValue / 64);
        compute_barrier(command);
        push = {kLinearValue, 128, kLinearValue / 4, kLinearValue / 4};
        kernels_.dispatch(command, kernels_.p().quant, context_quant_set_, &push,
                          kLinearValue / 128);
        compute_barrier(command);
        push = {kDim, kLinearValue, kLinearValue / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kLinearValue), attention_out_set_,
                          &push, kDim / 8);
        compute_barrier(command);
    }

    void record_router_shared(VkCommandBuffer command) {
        Push push{kDim, 128, kDim / 4, kDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, hidden_quant_set_, &push,
                          kDim / 128);
        compute_barrier(command);
        push = {kExperts, kDim, kDim / 4, 0};
        kernels_.dispatch(command, kernels_.p().q8, router_gemv_set_, &push,
                          kExperts / 8);
        compute_barrier(command);
        push = {kExperts, active_topk_, 0, 0};
        kernels_.dispatch(command, kernels_.p().router, router_set_, &push, 1);
        compute_barrier(command);
        push = {kMoeDim, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kDim), shared_gate_set_, &push,
                          kMoeDim / 8);
        kernels_.dispatch(command, q4_pipeline(kDim), shared_up_set_, &push,
                          kMoeDim / 8);
        push = {1, kDim, kDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kDim), shared_expert_gate_set_,
                          &push, 1);
        compute_barrier(command);
        push = {kMoeDim, float_bits(3.402823466e+38f), 0, 0};
        kernels_.dispatch(command, kernels_.p().swiglu, shared_swiglu_set_,
                          &push, kMoeDim / 64);
        compute_barrier(command);
        push = {kMoeDim, 128, kMoeDim / 4, kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, shared_quant_set_, &push,
                          kMoeDim / 128);
        compute_barrier(command);
        push = {kDim, kMoeDim, kMoeDim / 4, 0};
        kernels_.dispatch(command, q4_pipeline(kMoeDim), shared_down_set_, &push,
                          kDim / 8);
        compute_barrier(command);
    }

    void record_experts(VkCommandBuffer command) {
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
            Push push{rank, kDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_gate,
                              expert_gate_sets_[selected_slots_[rank]], &push,
                              kMoeDim / 8);
        }
        compute_barrier(command);
        Push push{active_topk_ * kMoeDim, 128,
                  active_topk_ * kMoeDim / 4,
                  active_topk_ * kMoeDim / 4};
        kernels_.dispatch(command, kernels_.p().quant, expert_quant_set_, &push,
                          active_topk_ * kMoeDim / 128);
        compute_barrier(command);
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
            push = {rank, active_topk_ * kMoeDim / 4, 0, 0};
            kernels_.dispatch(command, kernels_.p().expert_down,
                              expert_down_sets_[selected_slots_[rank]], &push,
                              kDim / 8);
        }
        compute_barrier(command);
        push = {kDim, active_topk_, 0, 0};
        kernels_.dispatch(command, kernels_.p().reduce, reduce_set_, &push,
                          kDim / 64);
        compute_barrier(command);
    }

    std::array<bool, kTopK> resolve(
        const std::array<uint32_t, kTopK>& experts) {
        std::array<bool, kTopK> misses{};
        std::vector<bool> reserved(slots_);
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
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
        for (uint32_t rank = 0; rank < active_topk_; ++rank) {
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
                if (replace) {
                    victim = slot;
                    if (entries_[slot].expert < 0) break;
                }
            }
            if (victim == UINT32_MAX)
                throw std::runtime_error("No Qwen3.8 MTP expert victim");
            entries_[victim].expert = int32_t(experts[rank]);
            entries_[victim].age = ++clock_;
            reserved[victim] = true;
            selected_slots_[rank] = victim;
        }
        return misses;
    }

    const Runtime& runtime_;
    DeviceWeights weights_;
    dsv4::ReadOnlyMapping expert_file_;
    Kernels kernels_;
    dsv4::FiniteQueue compute_, transfer_;
    DescriptorRange main_hyper_{};
    QwenEngine& main_;
    uint32_t slots_ = 0;
    uint32_t active_topk_ = kTopK;
    uint64_t activation_device_bytes_ = 0, clock_ = 0;
    uint64_t hits_ = 0, misses_ = 0, transfer_bytes_ = 0;
    uint64_t host_copy_bytes_ = 0, passes_ = 0;
    double seconds_ = 0;
    std::array<uint32_t, kExperts> frequency_{};
    std::vector<Entry> entries_;
    std::array<uint32_t, kTopK> selected_slots_{};
    std::vector<Buffer> staging_;

    Buffer token_{}, routing_{}, hyper_{}, zero_hyper_{};
    Buffer embedding_{}, embedding_norm_{}, hyper_norm_{}, fusion_embedding_{};
    Buffer embedding_quant_{}, hyper_quant_{}, hidden_{}, hidden_quant_{};
    Buffer hc_normed_{}, hc_mix_weights_{}, hc_low_{}, hc_low_quant_{};
    Buffer hc_injection_{}, block_output_{}, qgate_{}, key_{}, value_{};
    Buffer context_{}, shared_gate_{}, shared_up_{}, shared_intermediate_{};
    Buffer shared_output_{}, shared_expert_gate_{}, router_logits_{};
    Buffer expert_intermediate_{}, expert_quant_{}, expert_outputs_{};
    Buffer kv_{}, rope_{}, expert_arena_{};

    HcSets attn_hc_{}, mlp_hc_{}, final_hc_{};
    VkDescriptorSet embedding_set_{}, pre_embedding_norm_set_{};
    VkDescriptorSet pre_hidden_zero_set_{}, pre_hidden_main_set_{};
    VkDescriptorSet pre_hidden_self_set_{}, embedding_quant_set_{};
    VkDescriptorSet fc_embedding_set_{};
    std::array<VkDescriptorSet, kHcCount> hyper_branch_quant_sets_{};
    std::array<VkDescriptorSet, kHcCount> fc_hidden_sets_{};
    VkDescriptorSet hidden_quant_set_{}, qgate_set_{}, key_set_{}, value_set_{};
    VkDescriptorSet qk_set_{}, store_value_set_{}, attention_set_{};
    VkDescriptorSet head_gate_set_{}, context_quant_set_{}, attention_out_set_{};
    VkDescriptorSet router_gemv_set_{}, router_set_{}, shared_gate_set_{};
    VkDescriptorSet shared_up_set_{}, shared_swiglu_set_{}, shared_quant_set_{};
    VkDescriptorSet shared_down_set_{}, shared_expert_gate_set_{};
    VkDescriptorSet expert_quant_set_{}, reduce_set_{};
    std::vector<VkDescriptorSet> expert_gate_sets_, expert_down_sets_;
};

static uint32_t mtp_slots() {
    const char* text = std::getenv("QWEN38_MTP_SLOTS");
    return text ? static_cast<uint32_t>(std::stoul(text)) : 16u;
}

} // namespace qwen38_mtp

int main(int argc, char** argv) {
    Runtime runtime{};
    try {
        if (argc < 3) {
            std::cerr << "usage: xtllm_qwen38_mtp.exe <runtime-dir> "
                         "<prompt> [tokens]\n";
            return 2;
        }
        const std::filesystem::path directory = argv[1];
        dsv4::ReadOnlyMapping tokenizer_file(
            (directory / "tokenizer.ovb").string());
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
        qwen38::SharedIndex main_index(directory / "model-q4g64.ovs");
        qwen38::SharedIndex mtp_index(directory / "mtp-q4g64.ovs", true);
        runtime = create_runtime();
        std::cout << "Vulkan device: " << runtime.properties.deviceName << '\n';
        const uint32_t count = argc >= 4
            ? static_cast<uint32_t>(std::stoul(argv[3])) : 16u;
        const uint64_t budget = qwen38::ram_budget();
        const uint32_t slots = qwen38::device_slots();
        const uint32_t draft_slots = qwen38_mtp::mtp_slots();
        const bool thinking = std::getenv("QWEN38_NO_THINK") == nullptr;
        const auto shaders = std::filesystem::absolute(argv[0]).parent_path();
        std::vector<uint32_t> result;
        uint64_t accepted = 0, compared = 0;
        double decode = 0, draft_seconds = 0;
        uint64_t draft_hits = 0, draft_misses = 0, draft_h2d = 0;
        uint64_t main_h2d = 0, main_host_copy = 0, main_disk = 0;
        std::array<uint64_t, qwen38::kLayers> main_layer_misses{};
        uint64_t main_unique = 0, main_occurrences = 0, main_reused = 0;
        double main_acquire = 0;
        double main_pre = 0, main_expert = 0;
        uint64_t device_bytes = 0, controlled_host = 0;
        {
            qwen38::QwenEngine main(runtime, main_index,
                directory / "experts-q3g64.ovx", directory / "ple-fp8.ovp",
                shaders, budget, slots);
            qwen38_mtp::MtpOneEngine mtp(
                runtime, mtp_index, main.embedding_experiment_tensor(),
                main.hyper_experiment_range(),
                directory / "mtp-experts-q3g64.ovx", shaders, main,
                draft_slots);
            const std::vector<uint32_t> prompt =
                tokenizer.chat_prompt(argv[2], thinking);
            uint32_t position = 0, next = 0;
            mtp.process(prompt[0], 0,
                qwen38_mtp::MtpOneEngine::HiddenSource::zero, false);
            for (uint32_t index = 0; index < prompt.size(); ++index) {
                next = main.process_experiment_token(prompt[index], position++);
                if (index + 1 < prompt.size())
                    mtp.process(prompt[index + 1], position,
                        qwen38_mtp::MtpOneEngine::HiddenSource::main, false);
            }
            if (std::getenv("QWEN38_FILL_RAM_CACHE")) {
                double fill_seconds = 0;
                const uint32_t filled =
                    main.fill_experiment_ram_cache(fill_seconds);
                std::cout << "RAM cache top-off: " << filled << " records, "
                          << fill_seconds << " s\n";
            }
            main.reset_experiment_metrics();
            mtp.reset_metrics();
            const auto started = std::chrono::steady_clock::now();
            if (std::getenv("QWEN38_VERIFY4_EXPERIMENT")) {
                uint32_t relaxed_drafts =
                    std::getenv("QWEN38_RELAXED_ACCEPT_ALL") ? 3u : 0u;
                if (const char* text =
                        std::getenv("QWEN38_RELAXED_DRAFTS")) {
                    relaxed_drafts = static_cast<uint32_t>(std::stoul(text));
                    if (relaxed_drafts < 1 || relaxed_drafts > 3)
                        throw std::runtime_error(
                            "QWEN38_RELAXED_DRAFTS must be between 1 and 3");
                }
                const uint32_t fast_active_topk = main.verify_active_topk();
                const bool repeat_guard =
                    std::getenv("QWEN38_RELAXED_REPEAT_GUARD") != nullptr;
                uint32_t mismatch_budget = 3;
                if (const char* text =
                        std::getenv("QWEN38_RELAXED_MISMATCH_BUDGET")) {
                    mismatch_budget =
                        static_cast<uint32_t>(std::stoul(text));
                    if (mismatch_budget > 3)
                        throw std::runtime_error(
                            "QWEN38_RELAXED_MISMATCH_BUDGET must be 0..3");
                }
                uint32_t recovery_burst = 0;
                if (const char* text =
                        std::getenv("QWEN38_RELAXED_RECOVERY_CYCLES"))
                    recovery_burst = static_cast<uint32_t>(std::stoul(text));
                uint32_t recovery_cycles = 0;
                while (result.size() < count && !tokenizer.is_eos(next)) {
                    const uint32_t cycle_relaxed_drafts =
                        recovery_cycles ? 0u : relaxed_drafts;
                    result.push_back(next);
                    std::cout << tokenizer.decode_piece(next) << std::flush;
                    if (result.size() >= count) break;
                    const uint32_t d1 = mtp.process(next, position,
                        qwen38_mtp::MtpOneEngine::HiddenSource::main, true);
                    uint32_t d2 = d1, d3 = d1;
                    if (!cycle_relaxed_drafts || cycle_relaxed_drafts >= 2)
                        d2 = mtp.process(d1, position + 1,
                            qwen38_mtp::MtpOneEngine::HiddenSource::chained,
                            true);
                    if (!cycle_relaxed_drafts || cycle_relaxed_drafts >= 3)
                        d3 = mtp.process(d2, position + 2,
                            qwen38_mtp::MtpOneEngine::HiddenSource::chained,
                            true);
                    else
                        d3 = d2;
                    const std::array<uint32_t, 4> exact =
                        main.verify4_experiment({next, d1, d2, d3}, position);
                    const std::array<uint32_t, 3> drafts{d1, d2, d3};
                    uint32_t matched_here = 0;
                    while (matched_here < drafts.size() &&
                           drafts[matched_here] == exact[matched_here])
                        ++matched_here;
                    uint32_t committed_here =
                        cycle_relaxed_drafts ? cycle_relaxed_drafts
                                             : matched_here;
                    if (cycle_relaxed_drafts)
                        committed_here = std::min<uint32_t>(
                            committed_here, matched_here + mismatch_budget);
                    bool guard_blocked = false;
                    if (cycle_relaxed_drafts && repeat_guard) {
                        std::vector<uint32_t> guarded = result;
                        committed_here = 0;
                        for (uint32_t index = 0;
                             index < cycle_relaxed_drafts; ++index) {
                            const uint32_t candidate = drafts[index];
                            bool unsafe = candidate >= 248000u;
                            if (!unsafe && guarded.size() >= 3) {
                                const size_t suffix = guarded.size() - 3;
                                for (size_t prior = 0;
                                     prior + 3 < guarded.size(); ++prior) {
                                    if (guarded[prior] == guarded[suffix] &&
                                        guarded[prior + 1] ==
                                            guarded[suffix + 1] &&
                                        guarded[prior + 2] ==
                                            guarded[suffix + 2] &&
                                        guarded[prior + 3] == candidate) {
                                        unsafe = true;
                                        break;
                                    }
                                }
                            }
                            if (unsafe) {
                                guard_blocked = true;
                                break;
                            }
                            guarded.push_back(candidate);
                            ++committed_here;
                        }
                    }
                    for (uint32_t index = 0; index < committed_here; ++index) {
                        if (result.size() >= count) break;
                        result.push_back(drafts[index]);
                        std::cout << tokenizer.decode_piece(drafts[index])
                                  << std::flush;
                    }
                    accepted += matched_here;
                    compared += matched_here < 3 ? matched_here + 1 : 3;
                    if (std::getenv("QWEN38_ACCEPT_TRACE")) {
                        std::cerr << " [match=" << matched_here
                                  << " commit=" << committed_here
                                  << " draft=" << d1 << ',' << d2 << ',' << d3
                                  << " exact=" << exact[0] << ',' << exact[1]
                                  << ',' << exact[2] << ',' << exact[3] << "] ";
                    }
                    const uint32_t consumed = 1 + committed_here;
                    main.accept_verify4_experiment(consumed);
                    position += consumed;
                    next = repeat_guard
                        ? main.safe_verify_token(committed_here, result)
                        : exact[committed_here];
                    if (guard_blocked && recovery_burst) {
                        recovery_cycles = recovery_burst;
                        main.set_verify_active_topk(qwen38::kTopK);
                    } else if (recovery_cycles) {
                        --recovery_cycles;
                        if (!recovery_cycles)
                            main.set_verify_active_topk(fast_active_topk);
                    }
                    if (committed_here > 0 &&
                        (cycle_relaxed_drafts || committed_here == 3)) {
                        // Catch the draft KV/hyper state up through the last
                        // committed draft without rereading the vocabulary
                        // head.  Exact mode only needs this after all three
                        // drafts were accepted.
                        mtp.process(drafts[committed_here - 1], position - 1,
                            qwen38_mtp::MtpOneEngine::HiddenSource::chained,
                            false);
                    }
                }
            } else if (std::getenv("QWEN38_MTP_CHAIN_ONLY")) {
                bool first = true;
                while (result.size() < count && !tokenizer.is_eos(next)) {
                    result.push_back(next);
                    std::cout << tokenizer.decode_piece(next) << std::flush;
                    if (result.size() >= count) break;
                    next = mtp.process(next, position++, first
                        ? qwen38_mtp::MtpOneEngine::HiddenSource::main
                        : qwen38_mtp::MtpOneEngine::HiddenSource::chained, true);
                    first = false;
                    ++compared;
                }
            } else {
                while (result.size() < count && !tokenizer.is_eos(next)) {
                    result.push_back(next);
                    std::cout << tokenizer.decode_piece(next) << std::flush;
                    if (result.size() >= count) break;
                    const uint32_t draft = mtp.process(next, position,
                        qwen38_mtp::MtpOneEngine::HiddenSource::main, true);
                    const uint32_t exact =
                        main.process_experiment_token(next, position++);
                    ++compared;
                    if (draft == exact) ++accepted;
                    next = exact;
                }
            }
            decode = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            draft_seconds = mtp.seconds();
            draft_hits = mtp.hits();
            draft_misses = mtp.misses();
            draft_h2d = mtp.transfer_bytes();
            main_h2d = main.transfer_bytes();
            main_layer_misses = main.device_layer_misses();
            main_unique = main.verify_unique_experts();
            main_occurrences = main.verify_occurrences();
            main_reused = main.verify_reused_occurrences();
            main_host_copy = main.host_copy_bytes();
            main_disk = main.disk_bytes();
            main_acquire = main.acquisition_seconds();
            main_pre = main.pre_seconds();
            main_expert = main.expert_seconds();
            device_bytes = main.vram_bytes() + mtp.device_bytes();
            controlled_host = main.ram_bytes() + mtp.host_bytes();
        }
        const uint64_t timed = result.size() > 1 ? result.size() - 1 : 0;
        const double divisor = timed ? double(timed) : 1.0;
        std::cout << "\ntoken ids:";
        for (uint32_t token : result) std::cout << ' ' << token;
        std::cout << "\nsequential acceptance-run throughput: "
                  << (decode > 0 ? timed / decode : 0.0) << " tok/s\n"
                  << "MTP one-step greedy acceptance: " << accepted << '/'
                  << compared << " ("
                  << (compared ? 100.0 * accepted / compared : 0.0) << "%)\n"
                  << "MTP seconds/pass: "
                  << (compared ? draft_seconds / compared : 0.0)
                  << " (total " << draft_seconds << ")\n"
                  << "MTP cache hits/misses and H2D bytes/output: "
                  << draft_hits << '/' << draft_misses << " / "
                  << draft_h2d / divisor << '\n'
                  << "main acquisition ms/output: "
                  << 1000.0 * main_acquire / divisor << '\n'
                  << "main pre / expert seconds total: " << main_pre
                  << " / " << main_expert << '\n'
                  << "main SSD / host-copy / H2D bytes/output: "
                  << main_disk / divisor << " / "
                  << main_host_copy / divisor << " / " << main_h2d / divisor
                  << '\n'
                  << "main verifier unique/reused/occurrences: "
                  << main_unique << '/' << main_reused << '/'
                  << main_occurrences << '\n'
                  << "main device misses by layer:";
        for (uint64_t misses : main_layer_misses)
            std::cout << ' ' << misses;
        std::cout << '\n'
                  << "controlled host / device allocation GiB: "
                  << double(controlled_host) / double(1ull << 30) << " / "
                  << double(device_bytes) / double(1ull << 30) << '\n';
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Qwen3.8 MTP experiment error: " << error.what() << '\n';
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        return 1;
    }
}
