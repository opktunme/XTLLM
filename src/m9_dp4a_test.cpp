#define OVLLM_RUNTIME_ONLY
#include "transformer_block.cpp"

struct QuantizePush9 { uint32_t count, group, unused0, unused1; };

static uint32_t pack4(const int values[4]) {
    return (static_cast<uint32_t>(values[0]) & 255u) |
           ((static_cast<uint32_t>(values[1]) & 255u) << 8u) |
           ((static_cast<uint32_t>(values[2]) & 255u) << 16u) |
           ((static_cast<uint32_t>(values[3]) & 255u) << 24u);
}

int main(int argc, char** argv) {
    try {
        constexpr uint32_t INNER = 64, COLUMNS = 16, GROUP = 4;
        std::vector<float> input(INNER);
        for (uint32_t k = 0; k < INNER; ++k) input[k] = 1.7f * std::sin(k * .31f) + .11f * k;
        std::vector<int8_t> qweights(COLUMNS * INNER);
        std::vector<float> weight_scales(COLUMNS);
        std::vector<uint32_t> packed_weights(COLUMNS * INNER / 4);
        for (uint32_t row = 0; row < COLUMNS; ++row) {
            weight_scales[row] = .003f * (row + 1);
            for (uint32_t k = 0; k < INNER; ++k) qweights[row * INNER + k] =
                static_cast<int8_t>((int(row * 19 + k * 7) % 255) - 127);
            for (uint32_t p = 0; p < INNER / 4; ++p) {
                int values[4];
                for (uint32_t b = 0; b < 4; ++b) values[b] = qweights[row * INNER + p * 4 + b];
                packed_weights[row * INNER / 4 + p] = pack4(values);
            }
        }
        std::vector<int8_t> qinput(INNER);
        std::vector<float> activation_scales(INNER / GROUP);
        for (uint32_t group = 0; group < INNER / GROUP; ++group) {
            float maximum = 0;
            for (uint32_t k = 0; k < GROUP; ++k) maximum = std::max(maximum, std::abs(input[group * GROUP + k]));
            activation_scales[group] = maximum / 127.0f;
            for (uint32_t k = 0; k < GROUP; ++k) qinput[group * GROUP + k] = static_cast<int8_t>(
                std::lround(input[group * GROUP + k] / activation_scales[group]));
        }
        std::vector<float> reference(COLUMNS);
        for (uint32_t row = 0; row < COLUMNS; ++row) {
            float sum = 0;
            for (uint32_t group = 0; group < INNER / GROUP; ++group) {
                int dot = 0;
                for (uint32_t k = 0; k < GROUP; ++k) dot += int(qinput[group * GROUP + k]) *
                    int(qweights[row * INNER + group * GROUP + k]);
                sum += dot * activation_scales[group];
            }
            reference[row] = sum * weight_scales[row];
        }

        Runtime runtime = create_runtime();
        Buffer dummy = create_buffer(runtime, 4);
        Buffer input_buffer = upload_vector(runtime, input);
        Buffer quantized = create_buffer(runtime, 2048 * sizeof(uint32_t));
        Buffer weights = upload_vector(runtime, packed_weights);
        Buffer scales = upload_vector(runtime, weight_scales);
        Buffer output = create_buffer(runtime, COLUMNS * sizeof(float));
        ComputeResources resources = create_compute_resources(runtime, 2);
        const std::string shader_dir = argc > 1 ? argv[1] : "A:\\amd-vulkan-llm-m1\\build";
        VkPipeline quant_pipeline = create_pipeline(runtime, resources, shader_dir + "\\quantize_q8.comp.spv");
        VkPipeline dot_pipeline = create_pipeline(runtime, resources, shader_dir + "\\qgemv_dp4a.comp.spv");
        VkDescriptorSet quant_set = create_descriptor_set(runtime, resources,
            {&input_buffer, &quantized, &dummy, &dummy});
        VkDescriptorSet dot_set = create_descriptor_set(runtime, resources,
            {&quantized, &weights, &scales, &output});
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = runtime.queue_family;
        VkCommandPool pool; VK_CHECK(vkfn::CreateCommandPool(runtime.device, &pi, nullptr, &pool));
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ai.commandPool=pool; ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount=1;
        VkCommandBuffer command; VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device,&ai,&command));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; VK_CHECK(vkfn::BeginCommandBuffer(command,&bi));
        QuantizePush9 qp{INNER,GROUP,0,0}; dispatch(command,resources,quant_pipeline,quant_set,&qp,1,1); compute_barrier(command);
        LinearPush lp{1,COLUMNS,INNER,INNER/4}; dispatch(command,resources,dot_pipeline,dot_set,&lp,(COLUMNS+7)/8,1);
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; mb.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkfn::CmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&mb,0,nullptr,0,nullptr);
        VK_CHECK(vkfn::EndCommandBuffer(command)); VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&command;
        VK_CHECK(vkfn::QueueSubmit(runtime.queue,1,&si,VK_NULL_HANDLE)); VK_CHECK(vkfn::QueueWaitIdle(runtime.queue)); invalidate_buffer(runtime,output);
        float maximum_error=0; const float* gpu=static_cast<const float*>(output.mapped);
        for(uint32_t i=0;i<COLUMNS;++i){ maximum_error=std::max(maximum_error,std::abs(gpu[i]-reference[i])); std::cout<<i<<" cpu="<<reference[i]<<" gpu="<<gpu[i]<<"\n"; }
        std::cout<<"max_error="<<maximum_error<<"\n";
        return maximum_error < 1e-3f ? 0 : 2;
    } catch(const std::exception& e){ std::cerr<<e.what()<<"\n"; return 1; }
}
