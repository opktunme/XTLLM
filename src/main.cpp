#define VK_NO_PROTOTYPES
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkfn {
PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
PFN_vkCreateInstance CreateInstance;
PFN_vkDestroyInstance DestroyInstance;
PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
PFN_vkCreateDevice CreateDevice;
PFN_vkGetDeviceProcAddr GetDeviceProcAddr;

PFN_vkDestroyDevice DestroyDevice;
PFN_vkGetDeviceQueue GetDeviceQueue;
PFN_vkCreateBuffer CreateBuffer;
PFN_vkDestroyBuffer DestroyBuffer;
PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
PFN_vkAllocateMemory AllocateMemory;
PFN_vkFreeMemory FreeMemory;
PFN_vkBindBufferMemory BindBufferMemory;
PFN_vkMapMemory MapMemory;
PFN_vkUnmapMemory UnmapMemory;
PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
PFN_vkCreateDescriptorPool CreateDescriptorPool;
PFN_vkDestroyDescriptorPool DestroyDescriptorPool;
PFN_vkAllocateDescriptorSets AllocateDescriptorSets;
PFN_vkUpdateDescriptorSets UpdateDescriptorSets;
PFN_vkCreateShaderModule CreateShaderModule;
PFN_vkDestroyShaderModule DestroyShaderModule;
PFN_vkCreatePipelineLayout CreatePipelineLayout;
PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
PFN_vkCreateComputePipelines CreateComputePipelines;
PFN_vkDestroyPipeline DestroyPipeline;
PFN_vkCreateCommandPool CreateCommandPool;
PFN_vkDestroyCommandPool DestroyCommandPool;
PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
PFN_vkBeginCommandBuffer BeginCommandBuffer;
PFN_vkEndCommandBuffer EndCommandBuffer;
PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
PFN_vkCmdBindPipeline CmdBindPipeline;
PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
PFN_vkCmdPushConstants CmdPushConstants;
PFN_vkCmdDispatch CmdDispatch;
PFN_vkQueueSubmit QueueSubmit;
PFN_vkQueueWaitIdle QueueWaitIdle;
}  // namespace vkfn

static void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}

#define VK_CHECK(call) check_vk((call), #call)

template <typename T>
static T load_instance(VkInstance instance, const char* name) {
    auto fn = reinterpret_cast<T>(vkfn::GetInstanceProcAddr(instance, name));
    if (!fn) throw std::runtime_error(std::string("Missing Vulkan instance function: ") + name);
    return fn;
}

template <typename T>
static T load_device(VkDevice device, const char* name) {
    auto fn = reinterpret_cast<T>(vkfn::GetDeviceProcAddr(device, name));
    if (!fn) throw std::runtime_error(std::string("Missing Vulkan device function: ") + name);
    return fn;
}

static void load_instance_functions(VkInstance instance) {
    vkfn::DestroyInstance = load_instance<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
    vkfn::EnumeratePhysicalDevices = load_instance<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices");
    vkfn::GetPhysicalDeviceProperties = load_instance<PFN_vkGetPhysicalDeviceProperties>(instance, "vkGetPhysicalDeviceProperties");
    vkfn::GetPhysicalDeviceQueueFamilyProperties = load_instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkfn::GetPhysicalDeviceMemoryProperties = load_instance<PFN_vkGetPhysicalDeviceMemoryProperties>(instance, "vkGetPhysicalDeviceMemoryProperties");
    vkfn::CreateDevice = load_instance<PFN_vkCreateDevice>(instance, "vkCreateDevice");
    vkfn::GetDeviceProcAddr = load_instance<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr");
}

static void load_device_functions(VkDevice device) {
#define LOAD_DEVICE(name) vkfn::name = load_device<PFN_vk##name>(device, "vk" #name)
    LOAD_DEVICE(DestroyDevice);
    LOAD_DEVICE(GetDeviceQueue);
    LOAD_DEVICE(CreateBuffer);
    LOAD_DEVICE(DestroyBuffer);
    LOAD_DEVICE(GetBufferMemoryRequirements);
    LOAD_DEVICE(AllocateMemory);
    LOAD_DEVICE(FreeMemory);
    LOAD_DEVICE(BindBufferMemory);
    LOAD_DEVICE(MapMemory);
    LOAD_DEVICE(UnmapMemory);
    LOAD_DEVICE(FlushMappedMemoryRanges);
    LOAD_DEVICE(InvalidateMappedMemoryRanges);
    LOAD_DEVICE(CreateDescriptorSetLayout);
    LOAD_DEVICE(DestroyDescriptorSetLayout);
    LOAD_DEVICE(CreateDescriptorPool);
    LOAD_DEVICE(DestroyDescriptorPool);
    LOAD_DEVICE(AllocateDescriptorSets);
    LOAD_DEVICE(UpdateDescriptorSets);
    LOAD_DEVICE(CreateShaderModule);
    LOAD_DEVICE(DestroyShaderModule);
    LOAD_DEVICE(CreatePipelineLayout);
    LOAD_DEVICE(DestroyPipelineLayout);
    LOAD_DEVICE(CreateComputePipelines);
    LOAD_DEVICE(DestroyPipeline);
    LOAD_DEVICE(CreateCommandPool);
    LOAD_DEVICE(DestroyCommandPool);
    LOAD_DEVICE(AllocateCommandBuffers);
    LOAD_DEVICE(BeginCommandBuffer);
    LOAD_DEVICE(EndCommandBuffer);
    LOAD_DEVICE(CmdPipelineBarrier);
    LOAD_DEVICE(CmdBindPipeline);
    LOAD_DEVICE(CmdBindDescriptorSets);
    LOAD_DEVICE(CmdPushConstants);
    LOAD_DEVICE(CmdDispatch);
    LOAD_DEVICE(QueueSubmit);
    LOAD_DEVICE(QueueWaitIdle);
#undef LOAD_DEVICE
}

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    bool coherent = false;
};

struct Runtime {
    HMODULE loader = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkPhysicalDeviceProperties properties{};
};

static std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Could not open shader: " + path);
    const auto length = stream.tellg();
    if (length <= 0 || (length % 4) != 0) throw std::runtime_error("Invalid SPIR-V file size");
    std::vector<uint32_t> code(static_cast<size_t>(length) / sizeof(uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(code.data()), length);
    if (!stream) throw std::runtime_error("Could not read complete SPIR-V shader");
    return code;
}

static Runtime create_runtime() {
    Runtime rt;
    rt.loader = LoadLibraryA("vulkan-1.dll");
    if (!rt.loader) throw std::runtime_error("Windows Vulkan loader vulkan-1.dll is unavailable");
    vkfn::GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(rt.loader, "vkGetInstanceProcAddr"));
    if (!vkfn::GetInstanceProcAddr) throw std::runtime_error("vkGetInstanceProcAddr is unavailable");
    vkfn::CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        vkfn::GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!vkfn::CreateInstance) throw std::runtime_error("vkCreateInstance is unavailable");

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "AMD Vulkan Quantized Matmul M1";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName = "Our AMD LLM Runtime";
    app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    VK_CHECK(vkfn::CreateInstance(&instance_info, nullptr, &rt.instance));
    load_instance_functions(rt.instance);

    uint32_t device_count = 0;
    VK_CHECK(vkfn::EnumeratePhysicalDevices(rt.instance, &device_count, nullptr));
    if (device_count == 0) throw std::runtime_error("Vulkan found no physical devices");
    std::vector<VkPhysicalDevice> devices(device_count);
    VK_CHECK(vkfn::EnumeratePhysicalDevices(rt.instance, &device_count, devices.data()));

    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkfn::GetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorID == 0x1002 &&
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            rt.physical = candidate;
            rt.properties = properties;
            break;
        }
    }
    if (!rt.physical) {
        throw std::runtime_error("No discrete AMD Vulkan GPU found; refusing any CPU/software fallback");
    }

    uint32_t family_count = 0;
    vkfn::GetPhysicalDeviceQueueFamilyProperties(rt.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkfn::GetPhysicalDeviceQueueFamilyProperties(rt.physical, &family_count, families.data());
    bool found_family = false;
    for (uint32_t i = 0; i < family_count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            rt.queue_family = i;
            found_family = true;
            break;
        }
    }
    if (!found_family) {
        for (uint32_t i = 0; i < family_count; ++i) {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                rt.queue_family = i;
                found_family = true;
                break;
            }
        }
    }
    if (!found_family) throw std::runtime_error("Selected AMD GPU exposes no compute queue");

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = rt.queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VK_CHECK(vkfn::CreateDevice(rt.physical, &device_info, nullptr, &rt.device));
    load_device_functions(rt.device);
    vkfn::GetDeviceQueue(rt.device, rt.queue_family, 0, &rt.queue);
    vkfn::GetPhysicalDeviceMemoryProperties(rt.physical, &rt.memory_properties);
    return rt;
}

static uint32_t choose_host_memory(const Runtime& rt, uint32_t allowed, bool& coherent) {
    for (int pass = 0; pass < 2; ++pass) {
        for (uint32_t i = 0; i < rt.memory_properties.memoryTypeCount; ++i) {
            if (!(allowed & (1u << i))) continue;
            const auto flags = rt.memory_properties.memoryTypes[i].propertyFlags;
            if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            const bool is_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            if ((pass == 0 && is_coherent) || pass == 1) {
                coherent = is_coherent;
                return i;
            }
        }
    }
    throw std::runtime_error("GPU has no host-visible Vulkan memory type");
}

static Buffer create_buffer(const Runtime& rt, VkDeviceSize size) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(rt.device, &info, nullptr, &buffer.handle));

    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(rt.device, buffer.handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_host_memory(rt, requirements.memoryTypeBits, buffer.coherent);
    VK_CHECK(vkfn::AllocateMemory(rt.device, &allocation, nullptr, &buffer.memory));
    VK_CHECK(vkfn::BindBufferMemory(rt.device, buffer.handle, buffer.memory, 0));
    VK_CHECK(vkfn::MapMemory(rt.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped));
    return buffer;
}

static void flush_buffer(const Runtime& rt, const Buffer& buffer) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkfn::FlushMappedMemoryRanges(rt.device, 1, &range));
}

static void invalidate_buffer(const Runtime& rt, const Buffer& buffer) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkfn::InvalidateMappedMemoryRanges(rt.device, 1, &range));
}

static void destroy_buffer(const Runtime& rt, Buffer& buffer) {
    if (buffer.mapped) vkfn::UnmapMemory(rt.device, buffer.memory);
    if (buffer.handle) vkfn::DestroyBuffer(rt.device, buffer.handle, nullptr);
    if (buffer.memory) vkfn::FreeMemory(rt.device, buffer.memory, nullptr);
    buffer = {};
}

struct PushConstants {
    uint32_t rows;
    uint32_t columns;
    uint32_t inner;
    uint32_t packed_stride;
};

int main(int argc, char** argv) {
    try {
        const std::string shader_path = argc > 1 ? argv[1] : "qgemm.comp.spv";
        constexpr uint32_t M = 37;
        constexpr uint32_t N = 53;
        constexpr uint32_t K = 127;
        constexpr uint32_t PACKED_STRIDE = (K + 3) / 4;

        std::vector<float> activations(M * K);
        std::vector<float> original_weights(N * K);
        for (size_t i = 0; i < activations.size(); ++i) {
            activations[i] = 0.8f * std::sin(static_cast<float>(i) * 0.017f) +
                             0.2f * std::cos(static_cast<float>(i) * 0.031f);
        }
        for (uint32_t n = 0; n < N; ++n) {
            for (uint32_t k = 0; k < K; ++k) {
                original_weights[n * K + k] =
                    0.7f * std::sin(static_cast<float>((n + 1) * (k + 3)) * 0.013f) +
                    0.3f * std::cos(static_cast<float>(n * 7 + k * 5) * 0.021f);
            }
        }

        std::vector<float> scales(N);
        std::vector<int8_t> quantized(N * K);
        std::vector<uint32_t> packed_weights(N * PACKED_STRIDE, 0);
        for (uint32_t n = 0; n < N; ++n) {
            float max_abs = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                max_abs = std::max(max_abs, std::abs(original_weights[n * K + k]));
            }
            scales[n] = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            for (uint32_t k = 0; k < K; ++k) {
                int q = static_cast<int>(std::lround(original_weights[n * K + k] / scales[n]));
                q = std::max(-127, std::min(127, q));
                quantized[n * K + k] = static_cast<int8_t>(q);
                const uint32_t byte_value = static_cast<uint8_t>(quantized[n * K + k]);
                packed_weights[n * PACKED_STRIDE + k / 4] |= byte_value << ((k % 4) * 8);
            }
        }

        std::vector<float> cpu_quantized(M * N, 0.0f);
        std::vector<float> cpu_fp32(M * N, 0.0f);
        for (uint32_t m = 0; m < M; ++m) {
            for (uint32_t n = 0; n < N; ++n) {
                float quant_sum = 0.0f;
                float fp32_sum = 0.0f;
                for (uint32_t k = 0; k < K; ++k) {
                    quant_sum += activations[m * K + k] *
                                 (static_cast<float>(quantized[n * K + k]) * scales[n]);
                    fp32_sum += activations[m * K + k] * original_weights[n * K + k];
                }
                cpu_quantized[m * N + n] = quant_sum;
                cpu_fp32[m * N + n] = fp32_sum;
            }
        }

        Runtime rt = create_runtime();
        std::cout << "Selected Vulkan device: " << rt.properties.deviceName << "\n"
                  << "Vendor/device ID: 0x" << std::hex << rt.properties.vendorID << ":0x"
                  << rt.properties.deviceID << std::dec << "\n"
                  << "Device type: discrete GPU\n"
                  << "Compute queue family: " << rt.queue_family << "\n";

        Buffer activation_buffer = create_buffer(rt, activations.size() * sizeof(float));
        Buffer weight_buffer = create_buffer(rt, packed_weights.size() * sizeof(uint32_t));
        Buffer scale_buffer = create_buffer(rt, scales.size() * sizeof(float));
        Buffer output_buffer = create_buffer(rt, M * N * sizeof(float));
        std::memcpy(activation_buffer.mapped, activations.data(), activation_buffer.size);
        std::memcpy(weight_buffer.mapped, packed_weights.data(), weight_buffer.size);
        std::memcpy(scale_buffer.mapped, scales.data(), scale_buffer.size);
        std::memset(output_buffer.mapped, 0, static_cast<size_t>(output_buffer.size));
        flush_buffer(rt, activation_buffer);
        flush_buffer(rt, weight_buffer);
        flush_buffer(rt, scale_buffer);
        flush_buffer(rt, output_buffer);

        VkDescriptorSetLayoutBinding bindings[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo set_layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout_info.bindingCount = 4;
        set_layout_info.pBindings = bindings;
        VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateDescriptorSetLayout(rt.device, &set_layout_info, nullptr, &set_layout));

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = sizeof(PushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreatePipelineLayout(rt.device, &pipeline_layout_info, nullptr, &pipeline_layout));

        const std::vector<uint32_t> shader_code = read_spirv(shader_path);
        VkShaderModuleCreateInfo shader_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        shader_info.codeSize = shader_code.size() * sizeof(uint32_t);
        shader_info.pCode = shader_code.data();
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateShaderModule(rt.device, &shader_info, nullptr, &shader_module));
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader_module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline_info.stage = stage;
        pipeline_info.layout = pipeline_layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateComputePipelines(rt.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline));

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 4;
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateDescriptorPool(rt.device, &pool_info, nullptr, &descriptor_pool));
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = descriptor_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &set_layout;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateDescriptorSets(rt.device, &set_info, &descriptor_set));

        Buffer* buffers[4] = {&activation_buffer, &weight_buffer, &scale_buffer, &output_buffer};
        VkDescriptorBufferInfo buffer_infos[4]{};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t i = 0; i < 4; ++i) {
            buffer_infos[i].buffer = buffers[i]->handle;
            buffer_infos[i].range = buffers[i]->size;
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = descriptor_set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[i];
        }
        vkfn::UpdateDescriptorSets(rt.device, 4, writes, 0, nullptr);

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = rt.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(rt.device, &command_pool_info, nullptr, &command_pool));
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = command_pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateCommandBuffers(rt.device, &command_info, &command_buffer));
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkfn::BeginCommandBuffer(command_buffer, &begin_info));

        VkMemoryBarrier upload_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        upload_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        upload_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &upload_barrier, 0, nullptr, 0, nullptr);
        vkfn::CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkfn::CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
        const PushConstants params{M, N, K, PACKED_STRIDE};
        vkfn::CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                               0, sizeof(params), &params);
        vkfn::CmdDispatch(command_buffer, (N + 15) / 16, (M + 15) / 16, 1);

        VkMemoryBarrier download_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        download_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        download_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, 0,
                                 1, &download_barrier, 0, nullptr, 0, nullptr);
        VK_CHECK(vkfn::EndCommandBuffer(command_buffer));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command_buffer;
        VK_CHECK(vkfn::QueueSubmit(rt.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkfn::QueueWaitIdle(rt.queue));

        invalidate_buffer(rt, output_buffer);
        std::vector<float> gpu_output(M * N);
        std::memcpy(gpu_output.data(), output_buffer.mapped, output_buffer.size);

        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f;
        float quantization_max_abs_error = 0.0f;
        size_t failures = 0;
        for (size_t i = 0; i < gpu_output.size(); ++i) {
            const float abs_error = std::abs(gpu_output[i] - cpu_quantized[i]);
            const float rel_error = abs_error / std::max(1.0e-6f, std::abs(cpu_quantized[i]));
            max_abs_error = std::max(max_abs_error, abs_error);
            max_rel_error = std::max(max_rel_error, rel_error);
            quantization_max_abs_error = std::max(
                quantization_max_abs_error, std::abs(cpu_quantized[i] - cpu_fp32[i]));
            const float tolerance = 5.0e-4f + 5.0e-5f * std::abs(cpu_quantized[i]);
            if (!std::isfinite(gpu_output[i]) || abs_error > tolerance) ++failures;
        }

        std::cout << std::fixed << std::setprecision(8)
                  << "Matrix: " << M << "x" << K << " times " << N << "x" << K
                  << "^T -> " << M << "x" << N << "\n"
                  << "Quantization: symmetric signed INT8, per-output-channel FP32 scale, 4/uint32\n"
                  << "GPU dispatch: " << ((N + 15) / 16) << "x" << ((M + 15) / 16)
                  << " workgroups (16x16 local size)\n"
                  << "GPU vs CPU quantized max abs error: " << max_abs_error << "\n"
                  << "GPU vs CPU quantized max rel error: " << max_rel_error << "\n"
                  << "Quantized CPU vs original FP32 max abs error: " << quantization_max_abs_error << "\n"
                  << "Tolerance failures: " << failures << " / " << gpu_output.size() << "\n"
                  << (failures == 0 ? "RESULT: PASS - Vulkan quantized matmul is correct\n"
                                    : "RESULT: FAIL - GPU output exceeded tolerance\n");

        vkfn::DestroyCommandPool(rt.device, command_pool, nullptr);
        vkfn::DestroyDescriptorPool(rt.device, descriptor_pool, nullptr);
        vkfn::DestroyPipeline(rt.device, pipeline, nullptr);
        vkfn::DestroyShaderModule(rt.device, shader_module, nullptr);
        vkfn::DestroyPipelineLayout(rt.device, pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(rt.device, set_layout, nullptr);
        destroy_buffer(rt, output_buffer);
        destroy_buffer(rt, scale_buffer);
        destroy_buffer(rt, weight_buffer);
        destroy_buffer(rt, activation_buffer);
        vkfn::DestroyDevice(rt.device, nullptr);
        vkfn::DestroyInstance(rt.instance, nullptr);
        FreeLibrary(rt.loader);
        return failures == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
