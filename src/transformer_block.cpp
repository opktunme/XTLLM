#define VK_NO_PROTOTYPES
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace vkfn {
PFN_vkGetInstanceProcAddr GetInstanceProcAddr;
PFN_vkCreateInstance CreateInstance;
PFN_vkDestroyInstance DestroyInstance;
PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
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
PFN_vkResetCommandBuffer ResetCommandBuffer;
PFN_vkBeginCommandBuffer BeginCommandBuffer;
PFN_vkEndCommandBuffer EndCommandBuffer;
PFN_vkCmdPipelineBarrier CmdPipelineBarrier;
PFN_vkCmdBindPipeline CmdBindPipeline;
PFN_vkCmdBindDescriptorSets CmdBindDescriptorSets;
PFN_vkCmdPushConstants CmdPushConstants;
PFN_vkCmdDispatch CmdDispatch;
PFN_vkCmdDispatchIndirect CmdDispatchIndirect;
PFN_vkCmdCopyBuffer CmdCopyBuffer;
PFN_vkQueueSubmit QueueSubmit;
PFN_vkQueueWaitIdle QueueWaitIdle;
PFN_vkCreateFence CreateFence;
PFN_vkDestroyFence DestroyFence;
PFN_vkWaitForFences WaitForFences;
PFN_vkGetFenceStatus GetFenceStatus;
PFN_vkResetFences ResetFences;
PFN_vkCreateSemaphore CreateSemaphore;
PFN_vkDestroySemaphore DestroySemaphore;
PFN_vkGetSemaphoreCounterValue GetSemaphoreCounterValue;
PFN_vkSignalSemaphore SignalSemaphore;
PFN_vkWaitSemaphores WaitSemaphores;
PFN_vkGetBufferDeviceAddress GetBufferDeviceAddress;
PFN_vkCreateQueryPool CreateQueryPool;
PFN_vkDestroyQueryPool DestroyQueryPool;
PFN_vkCmdResetQueryPool CmdResetQueryPool;
PFN_vkCmdWriteTimestamp CmdWriteTimestamp;
PFN_vkGetQueryPoolResults GetQueryPoolResults;
PFN_vkGetMemoryHostPointerPropertiesEXT GetMemoryHostPointerPropertiesEXT;
}  // namespace vkfn

static bool external_host_memory_enabled = false;

static void check_vk(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with VkResult " +
                                 std::to_string(static_cast<int>(result)));
    }
}
#define VK_CHECK(call) check_vk((call), #call)

template <typename T>
static T load_instance(VkInstance instance, const char* name) {
    auto function = reinterpret_cast<T>(vkfn::GetInstanceProcAddr(instance, name));
    if (!function) throw std::runtime_error(std::string("Missing Vulkan function: ") + name);
    return function;
}

template <typename T>
static T load_device(VkDevice device, const char* name) {
    auto function = reinterpret_cast<T>(vkfn::GetDeviceProcAddr(device, name));
    if (!function) throw std::runtime_error(std::string("Missing Vulkan function: ") + name);
    return function;
}

static void load_instance_functions(VkInstance instance) {
    vkfn::DestroyInstance = load_instance<PFN_vkDestroyInstance>(instance, "vkDestroyInstance");
    vkfn::EnumeratePhysicalDevices = load_instance<PFN_vkEnumeratePhysicalDevices>(instance, "vkEnumeratePhysicalDevices");
    vkfn::EnumerateDeviceExtensionProperties = load_instance<PFN_vkEnumerateDeviceExtensionProperties>(instance, "vkEnumerateDeviceExtensionProperties");
    vkfn::GetPhysicalDeviceProperties = load_instance<PFN_vkGetPhysicalDeviceProperties>(instance, "vkGetPhysicalDeviceProperties");
    vkfn::GetPhysicalDeviceQueueFamilyProperties = load_instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    vkfn::GetPhysicalDeviceMemoryProperties = load_instance<PFN_vkGetPhysicalDeviceMemoryProperties>(instance, "vkGetPhysicalDeviceMemoryProperties");
    vkfn::GetPhysicalDeviceFeatures2 = load_instance<PFN_vkGetPhysicalDeviceFeatures2>(instance, "vkGetPhysicalDeviceFeatures2");
    vkfn::GetPhysicalDeviceProperties2 = load_instance<PFN_vkGetPhysicalDeviceProperties2>(instance, "vkGetPhysicalDeviceProperties2");
    vkfn::CreateDevice = load_instance<PFN_vkCreateDevice>(instance, "vkCreateDevice");
    vkfn::GetDeviceProcAddr = load_instance<PFN_vkGetDeviceProcAddr>(instance, "vkGetDeviceProcAddr");
}

static void load_device_functions(VkDevice device) {
#define LOAD(name) vkfn::name = load_device<PFN_vk##name>(device, "vk" #name)
    LOAD(DestroyDevice);
    LOAD(GetDeviceQueue);
    LOAD(CreateBuffer);
    LOAD(DestroyBuffer);
    LOAD(GetBufferMemoryRequirements);
    LOAD(AllocateMemory);
    LOAD(FreeMemory);
    LOAD(BindBufferMemory);
    LOAD(MapMemory);
    LOAD(UnmapMemory);
    LOAD(FlushMappedMemoryRanges);
    LOAD(InvalidateMappedMemoryRanges);
    LOAD(CreateDescriptorSetLayout);
    LOAD(DestroyDescriptorSetLayout);
    LOAD(CreateDescriptorPool);
    LOAD(DestroyDescriptorPool);
    LOAD(AllocateDescriptorSets);
    LOAD(UpdateDescriptorSets);
    LOAD(CreateShaderModule);
    LOAD(DestroyShaderModule);
    LOAD(CreatePipelineLayout);
    LOAD(DestroyPipelineLayout);
    LOAD(CreateComputePipelines);
    LOAD(DestroyPipeline);
    LOAD(CreateCommandPool);
    LOAD(DestroyCommandPool);
    LOAD(AllocateCommandBuffers);
    LOAD(ResetCommandBuffer);
    LOAD(BeginCommandBuffer);
    LOAD(EndCommandBuffer);
    LOAD(CmdPipelineBarrier);
    LOAD(CmdBindPipeline);
    LOAD(CmdBindDescriptorSets);
    LOAD(CmdPushConstants);
    LOAD(CmdDispatch);
    LOAD(CmdDispatchIndirect);
    LOAD(CmdCopyBuffer);
    LOAD(QueueSubmit);
    LOAD(QueueWaitIdle);
    LOAD(CreateFence);
    LOAD(DestroyFence);
    LOAD(WaitForFences);
    LOAD(GetFenceStatus);
    LOAD(ResetFences);
    LOAD(CreateSemaphore);
    LOAD(DestroySemaphore);
    LOAD(GetSemaphoreCounterValue);
    LOAD(SignalSemaphore);
    LOAD(WaitSemaphores);
    LOAD(GetBufferDeviceAddress);
    LOAD(CreateQueryPool);
    LOAD(DestroyQueryPool);
    LOAD(CmdResetQueryPool);
    LOAD(CmdWriteTimestamp);
    LOAD(GetQueryPoolResults);
#undef LOAD
}

struct Runtime {
    HMODULE loader = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkQueue secondary_queue = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
};

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
    VkDeviceSize allocation_size = 0;
    bool coherent = false;
};

static uint64_t active_vulkan_buffer_bytes = 0;
static uint64_t peak_vulkan_buffer_bytes = 0;
static VkQueryPool dispatch_profile_pool = VK_NULL_HANDLE;
static bool dispatch_profile_enabled = false;
static uint32_t dispatch_profile_next = 0;
static std::vector<VkPipeline> dispatch_profile_pipelines;

static Runtime create_runtime() {
    Runtime runtime;
    runtime.loader = LoadLibraryA("vulkan-1.dll");
    if (!runtime.loader) throw std::runtime_error("vulkan-1.dll is unavailable");
    vkfn::GetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(runtime.loader, "vkGetInstanceProcAddr"));
    if (!vkfn::GetInstanceProcAddr) throw std::runtime_error("vkGetInstanceProcAddr is unavailable");
    vkfn::CreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
        vkfn::GetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!vkfn::CreateInstance) throw std::runtime_error("vkCreateInstance is unavailable");

    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.pApplicationName = "Our Vulkan Transformer Block";
    application.applicationVersion = VK_MAKE_VERSION(0, 2, 0);
    application.pEngineName = "Our AMD LLM Runtime";
    application.engineVersion = VK_MAKE_VERSION(0, 2, 0);
    application.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    VK_CHECK(vkfn::CreateInstance(&instance_info, nullptr, &runtime.instance));
    load_instance_functions(runtime.instance);

    uint32_t device_count = 0;
    VK_CHECK(vkfn::EnumeratePhysicalDevices(runtime.instance, &device_count, nullptr));
    std::vector<VkPhysicalDevice> devices(device_count);
    VK_CHECK(vkfn::EnumeratePhysicalDevices(runtime.instance, &device_count, devices.data()));
    for (VkPhysicalDevice candidate : devices) {
        VkPhysicalDeviceProperties properties{};
        vkfn::GetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorID == 0x1002 &&
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            runtime.physical = candidate;
            runtime.properties = properties;
            break;
        }
    }
    if (!runtime.physical) {
        throw std::runtime_error("No discrete AMD Vulkan GPU found; refusing CPU/software fallback");
    }

    uint32_t family_count = 0;
    vkfn::GetPhysicalDeviceQueueFamilyProperties(runtime.physical, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkfn::GetPhysicalDeviceQueueFamilyProperties(runtime.physical, &family_count, families.data());
    bool found = false;
    for (uint32_t family = 0; family < family_count; ++family) {
        if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            !(families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            runtime.queue_family = family;
            found = true;
            break;
        }
    }
    if (!found) {
        for (uint32_t family = 0; family < family_count; ++family) {
            if (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                runtime.queue_family = family;
                found = true;
                break;
            }
        }
    }
    if (!found) throw std::runtime_error("AMD GPU exposes no compute queue");

    const float priorities[2]{1.0f, 0.8f};
    const uint32_t requested_queue_count =
        std::min(2u, families[runtime.queue_family].queueCount);
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = runtime.queue_family;
    queue_info.queueCount = requested_queue_count;
    queue_info.pQueuePriorities = priorities;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features supported13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    supported12.pNext = &supported13;
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &supported12;
    vkfn::GetPhysicalDeviceFeatures2(runtime.physical, &supported);
    if (!supported13.shaderIntegerDotProduct || !supported12.shaderFloat16 ||
        !supported13.subgroupSizeControl) {
        throw std::runtime_error("AMD GPU lacks required Vulkan packed arithmetic support");
    }
    VkPhysicalDeviceVulkan12Features enabled12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features enabled13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures enabled_features{};
    enabled12.pNext = &enabled13;
    enabled12.shaderFloat16 = VK_TRUE;
    enabled12.timelineSemaphore = VK_TRUE;
    enabled12.bufferDeviceAddress = VK_TRUE;
    enabled13.shaderIntegerDotProduct = VK_TRUE;
    enabled13.subgroupSizeControl = VK_TRUE;
    enabled13.computeFullSubgroups = VK_TRUE;
    if (!supported.features.shaderStorageBufferArrayDynamicIndexing)
        throw std::runtime_error("AMD GPU lacks dynamic expert-buffer indexing");
    enabled_features.shaderStorageBufferArrayDynamicIndexing = VK_TRUE;
    device_info.pEnabledFeatures = &enabled_features;
    device_info.pNext = &enabled12;
    uint32_t extension_count = 0;
    VK_CHECK(vkfn::EnumerateDeviceExtensionProperties(runtime.physical, nullptr,
                                                       &extension_count, nullptr));
    std::vector<VkExtensionProperties> extensions(extension_count);
    VK_CHECK(vkfn::EnumerateDeviceExtensionProperties(runtime.physical, nullptr,
                                                       &extension_count, extensions.data()));
    for (const VkExtensionProperties& extension : extensions) {
        if (std::strcmp(extension.extensionName,
                        VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0) {
            external_host_memory_enabled = true;
            break;
        }
    }
    const char* external_host_extension = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
    if (external_host_memory_enabled) {
        device_info.enabledExtensionCount = 1;
        device_info.ppEnabledExtensionNames = &external_host_extension;
    }
    VK_CHECK(vkfn::CreateDevice(runtime.physical, &device_info, nullptr, &runtime.device));
    load_device_functions(runtime.device);
    if (external_host_memory_enabled) {
        vkfn::GetMemoryHostPointerPropertiesEXT =
            load_device<PFN_vkGetMemoryHostPointerPropertiesEXT>(
                runtime.device, "vkGetMemoryHostPointerPropertiesEXT");
    }
    vkfn::GetDeviceQueue(runtime.device, runtime.queue_family, 0, &runtime.queue);
    vkfn::GetDeviceQueue(runtime.device, runtime.queue_family,
        requested_queue_count > 1 ? 1u : 0u, &runtime.secondary_queue);
    vkfn::GetPhysicalDeviceMemoryProperties(runtime.physical, &runtime.memory_properties);
    return runtime;
}

static uint32_t choose_memory_type(const Runtime& runtime, uint32_t allowed, bool& coherent) {
    for (uint32_t pass = 0; pass < 4; ++pass) {
        for (uint32_t type = 0; type < runtime.memory_properties.memoryTypeCount; ++type) {
            if (!(allowed & (1u << type))) continue;
            const VkMemoryPropertyFlags flags = runtime.memory_properties.memoryTypes[type].propertyFlags;
            if (!(flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) continue;
            const bool is_coherent = (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
            const bool is_cached = (flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
            if ((pass == 0 && is_coherent && is_cached) ||
                (pass == 1 && is_cached) || (pass == 2 && is_coherent) || pass == 3) {
                coherent = is_coherent;
                return type;
            }
        }
    }
    throw std::runtime_error("No host-visible memory type is available");
}

static Buffer create_buffer(const Runtime& runtime, VkDeviceSize size) {
    Buffer buffer;
    buffer.size = size;
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkfn::CreateBuffer(runtime.device, &buffer_info, nullptr, &buffer.handle));
    VkMemoryRequirements requirements{};
    vkfn::GetBufferMemoryRequirements(runtime.device, buffer.handle, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = choose_memory_type(runtime, requirements.memoryTypeBits, buffer.coherent);
    VK_CHECK(vkfn::AllocateMemory(runtime.device, &allocation, nullptr, &buffer.memory));
    buffer.allocation_size = allocation.allocationSize;
    active_vulkan_buffer_bytes += buffer.allocation_size;
    peak_vulkan_buffer_bytes = std::max(peak_vulkan_buffer_bytes, active_vulkan_buffer_bytes);
    VK_CHECK(vkfn::BindBufferMemory(runtime.device, buffer.handle, buffer.memory, 0));
    VK_CHECK(vkfn::MapMemory(runtime.device, buffer.memory, 0, VK_WHOLE_SIZE, 0, &buffer.mapped));
    std::memset(buffer.mapped, 0, static_cast<size_t>(size));
    return buffer;
}

static void flush_buffer(const Runtime& runtime, const Buffer& buffer) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkfn::FlushMappedMemoryRanges(runtime.device, 1, &range));
}

static void invalidate_buffer(const Runtime& runtime, const Buffer& buffer) {
    if (buffer.coherent) return;
    VkMappedMemoryRange range{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
    range.memory = buffer.memory;
    range.size = VK_WHOLE_SIZE;
    VK_CHECK(vkfn::InvalidateMappedMemoryRanges(runtime.device, 1, &range));
}

template <typename T>
static Buffer upload_vector(const Runtime& runtime, const std::vector<T>& values) {
    Buffer buffer = create_buffer(runtime, values.size() * sizeof(T));
    std::memcpy(buffer.mapped, values.data(), static_cast<size_t>(buffer.size));
    flush_buffer(runtime, buffer);
    return buffer;
}

static void destroy_buffer(const Runtime& runtime, Buffer& buffer) {
    if (buffer.mapped) vkfn::UnmapMemory(runtime.device, buffer.memory);
    if (buffer.handle) vkfn::DestroyBuffer(runtime.device, buffer.handle, nullptr);
    if (buffer.memory) vkfn::FreeMemory(runtime.device, buffer.memory, nullptr);
    active_vulkan_buffer_bytes -= buffer.allocation_size;
    buffer = {};
}

static std::vector<uint32_t> read_spirv(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Could not open shader: " + path);
    const std::streamsize length = stream.tellg();
    if (length <= 0 || length % 4 != 0) throw std::runtime_error("Invalid SPIR-V file: " + path);
    std::vector<uint32_t> code(static_cast<size_t>(length) / sizeof(uint32_t));
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(code.data()), length);
    if (!stream) throw std::runtime_error("Could not read shader: " + path);
    return code;
}

struct QuantizedMatrix {
    uint32_t rows = 0;
    uint32_t columns = 0;
    uint32_t packed_stride = 0;
    std::vector<int8_t> values;
    std::vector<uint32_t> packed;
    std::vector<float> scales;
};

static QuantizedMatrix make_quantized_matrix(uint32_t rows, uint32_t columns, uint32_t seed,
                                              float amplitude) {
    QuantizedMatrix matrix;
    matrix.rows = rows;
    matrix.columns = columns;
    matrix.packed_stride = (columns + 3) / 4;
    matrix.values.resize(static_cast<size_t>(rows) * columns);
    matrix.packed.assign(static_cast<size_t>(rows) * matrix.packed_stride, 0);
    matrix.scales.resize(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        std::vector<float> source(columns);
        float maximum = 0.0f;
        for (uint32_t column = 0; column < columns; ++column) {
            const float phase0 = static_cast<float>((row + 1) * (column + seed + 3)) * 0.019f;
            const float phase1 = static_cast<float>(row * 11 + column * 5 + seed * 7) * 0.013f;
            source[column] = amplitude * (0.78f * std::sin(phase0) + 0.22f * std::cos(phase1));
            maximum = std::max(maximum, std::abs(source[column]));
        }
        matrix.scales[row] = maximum > 0.0f ? maximum / 127.0f : 1.0f;
        for (uint32_t column = 0; column < columns; ++column) {
            int quantized = static_cast<int>(std::lround(source[column] / matrix.scales[row]));
            quantized = std::max(-127, std::min(127, quantized));
            matrix.values[static_cast<size_t>(row) * columns + column] = static_cast<int8_t>(quantized);
            const uint32_t byte_value = static_cast<uint8_t>(quantized);
            matrix.packed[static_cast<size_t>(row) * matrix.packed_stride + column / 4] |=
                byte_value << ((column % 4) * 8);
        }
    }
    return matrix;
}

static std::vector<float> cpu_rmsnorm(const std::vector<float>& input, uint32_t rows,
                                      uint32_t columns, const std::vector<float>& gamma,
                                      float epsilon) {
    std::vector<float> output(input.size());
    for (uint32_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (uint32_t column = 0; column < columns; ++column) {
            const float value = input[static_cast<size_t>(row) * columns + column];
            square_sum += value * value;
        }
        const float inverse_rms = 1.0f / std::sqrt(square_sum / static_cast<float>(columns) + epsilon);
        for (uint32_t column = 0; column < columns; ++column) {
            output[static_cast<size_t>(row) * columns + column] =
                input[static_cast<size_t>(row) * columns + column] * inverse_rms * gamma[column];
        }
    }
    return output;
}

static std::vector<float> cpu_linear(const std::vector<float>& input, uint32_t input_rows,
                                     const QuantizedMatrix& weights) {
    std::vector<float> output(static_cast<size_t>(input_rows) * weights.rows, 0.0f);
    for (uint32_t input_row = 0; input_row < input_rows; ++input_row) {
        for (uint32_t output_column = 0; output_column < weights.rows; ++output_column) {
            float sum = 0.0f;
            for (uint32_t inner = 0; inner < weights.columns; ++inner) {
                sum += input[static_cast<size_t>(input_row) * weights.columns + inner] *
                       (static_cast<float>(weights.values[static_cast<size_t>(output_column) *
                                                          weights.columns + inner]) *
                        weights.scales[output_column]);
            }
            output[static_cast<size_t>(input_row) * weights.rows + output_column] = sum;
        }
    }
    return output;
}

static std::vector<float> cpu_rope(const std::vector<float>& input, uint32_t tokens,
                                   uint32_t model_dimension, uint32_t head_dimension,
                                   const std::vector<float>& rope_cos,
                                   const std::vector<float>& rope_sin) {
    std::vector<float> output = input;
    const uint32_t qkv_width = 3 * model_dimension;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t projection = 0; projection < 2; ++projection) {
            for (uint32_t vector_dimension = 0; vector_dimension < model_dimension;
                 vector_dimension += 2) {
                const uint32_t pair = (vector_dimension % head_dimension) / 2;
                const float cosine = rope_cos[static_cast<size_t>(token) * (head_dimension / 2) + pair];
                const float sine = rope_sin[static_cast<size_t>(token) * (head_dimension / 2) + pair];
                const size_t base = static_cast<size_t>(token) * qkv_width +
                                    projection * model_dimension + vector_dimension;
                const float even = input[base];
                const float odd = input[base + 1];
                output[base] = even * cosine - odd * sine;
                output[base + 1] = even * sine + odd * cosine;
            }
        }
    }
    return output;
}

static std::vector<float> cpu_attention(const std::vector<float>& qkv, uint32_t tokens,
                                        uint32_t model_dimension, uint32_t heads,
                                        uint32_t head_dimension) {
    std::vector<float> output(static_cast<size_t>(tokens) * model_dimension, 0.0f);
    const uint32_t qkv_width = 3 * model_dimension;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dimension));
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < heads; ++head) {
            std::vector<float> scores(token + 1);
            float maximum = -std::numeric_limits<float>::infinity();
            const size_t query_base = static_cast<size_t>(token) * qkv_width + head * head_dimension;
            for (uint32_t source = 0; source <= token; ++source) {
                const size_t key_base = static_cast<size_t>(source) * qkv_width + model_dimension +
                                        head * head_dimension;
                float score = 0.0f;
                for (uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
                    score += qkv[query_base + dimension] * qkv[key_base + dimension];
                }
                scores[source] = score * scale;
                maximum = std::max(maximum, scores[source]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (uint32_t dimension = 0; dimension < head_dimension; ++dimension) {
                float value = 0.0f;
                for (uint32_t source = 0; source <= token; ++source) {
                    const size_t value_index = static_cast<size_t>(source) * qkv_width +
                                               2 * model_dimension + head * head_dimension + dimension;
                    value += (scores[source] / denominator) * qkv[value_index];
                }
                output[static_cast<size_t>(token) * model_dimension +
                       head * head_dimension + dimension] = value;
            }
        }
    }
    return output;
}

static std::vector<float> cpu_add(const std::vector<float>& left,
                                  const std::vector<float>& right) {
    std::vector<float> output(left.size());
    for (size_t index = 0; index < output.size(); ++index) output[index] = left[index] + right[index];
    return output;
}

static std::vector<float> cpu_swiglu(const std::vector<float>& gate_up, uint32_t rows,
                                     uint32_t feed_forward_dimension) {
    std::vector<float> output(static_cast<size_t>(rows) * feed_forward_dimension);
    for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t column = 0; column < feed_forward_dimension; ++column) {
            const size_t base = static_cast<size_t>(row) * 2 * feed_forward_dimension;
            const float gate = gate_up[base + column];
            const float up = gate_up[base + feed_forward_dimension + column];
            output[static_cast<size_t>(row) * feed_forward_dimension + column] =
                (gate / (1.0f + std::exp(-gate))) * up;
        }
    }
    return output;
}

struct ComputeResources {
    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    std::vector<VkShaderModule> shader_modules;
    std::vector<VkPipeline> pipelines;
};

static ComputeResources create_compute_resources(const Runtime& runtime, uint32_t set_count) {
    ComputeResources resources;
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t binding = 0; binding < 6; ++binding) {
        bindings[binding].binding = binding;
        bindings[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[binding].descriptorCount = 1;
        bindings[binding].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    descriptor_info.bindingCount = 6;
    descriptor_info.pBindings = bindings;
    VK_CHECK(vkfn::CreateDescriptorSetLayout(runtime.device, &descriptor_info, nullptr,
                                              &resources.descriptor_layout));
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.size = 16;
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &resources.descriptor_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CHECK(vkfn::CreatePipelineLayout(runtime.device, &layout_info, nullptr,
                                        &resources.pipeline_layout));
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 6 * set_count;
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = set_count;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    VK_CHECK(vkfn::CreateDescriptorPool(runtime.device, &pool_info, nullptr,
                                        &resources.descriptor_pool));
    return resources;
}

static VkPipeline create_pipeline(const Runtime& runtime, ComputeResources& resources,
                                  const std::string& shader_path) {
    const std::vector<uint32_t> code = read_spirv(shader_path);
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = code.size() * sizeof(uint32_t);
    module_info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateShaderModule(runtime.device, &module_info, nullptr, &module));
    VkPipelineShaderStageCreateInfo stage_info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = module;
    stage_info.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (shader_path.find("wave32") != std::string::npos) {
        subgroup.requiredSubgroupSize = 32;
        stage_info.pNext = &subgroup;
        stage_info.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = stage_info;
    pipeline_info.layout = resources.pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkfn::CreateComputePipelines(runtime.device, VK_NULL_HANDLE, 1, &pipeline_info,
                                          nullptr, &pipeline));
    resources.shader_modules.push_back(module);
    resources.pipelines.push_back(pipeline);
    return pipeline;
}

static VkDescriptorSet create_descriptor_set(const Runtime& runtime,
                                              const ComputeResources& resources,
                                              const std::array<Buffer*, 4>& buffers) {
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = resources.descriptor_pool;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &resources.descriptor_layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkfn::AllocateDescriptorSets(runtime.device, &allocation, &set));
    VkDescriptorBufferInfo infos[4]{};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t binding = 0; binding < 4; ++binding) {
        infos[binding].buffer = buffers[binding]->handle;
        infos[binding].range = buffers[binding]->size;
        writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[binding].dstSet = set;
        writes[binding].dstBinding = binding;
        writes[binding].descriptorCount = 1;
        writes[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[binding].pBufferInfo = &infos[binding];
    }
    vkfn::UpdateDescriptorSets(runtime.device, 4, writes, 0, nullptr);
    return set;
}

static void dispatch(VkCommandBuffer command_buffer, const ComputeResources& resources,
                     VkPipeline pipeline, VkDescriptorSet descriptor_set, const void* push_data,
                     uint32_t groups_x, uint32_t groups_y) {
    vkfn::CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkfn::CmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                resources.pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    vkfn::CmdPushConstants(command_buffer, resources.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, 16, push_data);
    if (dispatch_profile_enabled) {
        vkfn::CmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                dispatch_profile_pool, dispatch_profile_next++);
        dispatch_profile_pipelines.push_back(pipeline);
    }
    vkfn::CmdDispatch(command_buffer, groups_x, groups_y, 1);
    if (dispatch_profile_enabled) {
        vkfn::CmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                dispatch_profile_pool, dispatch_profile_next++);
    }
}

static void compute_barrier(VkCommandBuffer command_buffer) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &barrier, 0, nullptr, 0, nullptr);
}

static void compute_buffer_barrier(VkCommandBuffer command_buffer, const Buffer& buffer) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer.handle;
    barrier.offset = 0;
    barrier.size = buffer.size;
    vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, 1, &barrier, 0, nullptr);
}

static void compute_two_buffer_barrier(VkCommandBuffer command_buffer,
                                       const Buffer& first, const Buffer& second) {
    std::array<VkBufferMemoryBarrier, 2> barriers{};
    const std::array<const Buffer*, 2> buffers{&first, &second};
    for (size_t index = 0; index < barriers.size(); ++index) {
        barriers[index].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barriers[index].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barriers[index].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barriers[index].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barriers[index].buffer = buffers[index]->handle;
        barriers[index].offset = 0;
        barriers[index].size = buffers[index]->size;
    }
    vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             0, nullptr, static_cast<uint32_t>(barriers.size()),
                             barriers.data(), 0, nullptr);
}

struct LinearPush { uint32_t rows, columns, inner, packed_stride; };
struct RmsPush { uint32_t rows, columns; float epsilon; uint32_t unused; };
struct RopePush { uint32_t tokens, model_dimension, head_dimension, qkv_dimension; };
struct AttentionPush { uint32_t tokens, model_dimension, heads, head_dimension; };
struct AddPush { uint32_t count, unused0, unused1, unused2; };
struct SwiGluPush { uint32_t rows, feed_forward_dimension, unused0, unused1; };

#ifndef OVLLM_RUNTIME_ONLY
int main(int argc, char** argv) {
    try {
        const std::string shader_directory = argc > 1 ? argv[1] : ".";
        constexpr uint32_t TOKENS = 8;
        constexpr uint32_t MODEL = 64;
        constexpr uint32_t HEADS = 4;
        constexpr uint32_t HEAD_DIMENSION = MODEL / HEADS;
        constexpr uint32_t FEED_FORWARD = 128;
        constexpr float RMS_EPSILON = 1.0e-5f;

        std::vector<float> input(static_cast<size_t>(TOKENS) * MODEL);
        for (size_t index = 0; index < input.size(); ++index) {
            input[index] = 0.48f * std::sin(static_cast<float>(index + 3) * 0.037f) +
                           0.17f * std::cos(static_cast<float>(index + 11) * 0.021f);
        }
        std::vector<float> attention_gamma(MODEL);
        std::vector<float> feed_forward_gamma(MODEL);
        for (uint32_t column = 0; column < MODEL; ++column) {
            attention_gamma[column] = 1.0f + 0.08f * std::sin(static_cast<float>(column) * 0.13f);
            feed_forward_gamma[column] = 1.0f + 0.06f * std::cos(static_cast<float>(column) * 0.11f);
        }
        std::vector<float> rope_cos(static_cast<size_t>(TOKENS) * HEAD_DIMENSION / 2);
        std::vector<float> rope_sin(rope_cos.size());
        for (uint32_t token = 0; token < TOKENS; ++token) {
            for (uint32_t pair = 0; pair < HEAD_DIMENSION / 2; ++pair) {
                const float frequency = std::pow(10000.0f,
                    -2.0f * static_cast<float>(pair) / static_cast<float>(HEAD_DIMENSION));
                const float angle = static_cast<float>(token) * frequency;
                rope_cos[static_cast<size_t>(token) * (HEAD_DIMENSION / 2) + pair] = std::cos(angle);
                rope_sin[static_cast<size_t>(token) * (HEAD_DIMENSION / 2) + pair] = std::sin(angle);
            }
        }

        const QuantizedMatrix qkv_weights = make_quantized_matrix(3 * MODEL, MODEL, 3, 0.075f);
        const QuantizedMatrix output_weights = make_quantized_matrix(MODEL, MODEL, 17, 0.065f);
        const QuantizedMatrix gate_up_weights = make_quantized_matrix(2 * FEED_FORWARD, MODEL, 29, 0.060f);
        const QuantizedMatrix down_weights = make_quantized_matrix(MODEL, FEED_FORWARD, 43, 0.052f);

        const std::vector<float> cpu_norm1 = cpu_rmsnorm(input, TOKENS, MODEL, attention_gamma, RMS_EPSILON);
        const std::vector<float> cpu_qkv = cpu_linear(cpu_norm1, TOKENS, qkv_weights);
        const std::vector<float> cpu_rotated_qkv = cpu_rope(cpu_qkv, TOKENS, MODEL, HEAD_DIMENSION,
                                                            rope_cos, rope_sin);
        const std::vector<float> cpu_context = cpu_attention(cpu_rotated_qkv, TOKENS, MODEL, HEADS,
                                                              HEAD_DIMENSION);
        const std::vector<float> cpu_attention_projection = cpu_linear(cpu_context, TOKENS, output_weights);
        const std::vector<float> cpu_residual = cpu_add(input, cpu_attention_projection);
        const std::vector<float> cpu_norm2 = cpu_rmsnorm(cpu_residual, TOKENS, MODEL,
                                                         feed_forward_gamma, RMS_EPSILON);
        const std::vector<float> cpu_gate_up = cpu_linear(cpu_norm2, TOKENS, gate_up_weights);
        const std::vector<float> cpu_feed_forward = cpu_swiglu(cpu_gate_up, TOKENS, FEED_FORWARD);
        const std::vector<float> cpu_down = cpu_linear(cpu_feed_forward, TOKENS, down_weights);
        const std::vector<float> cpu_output = cpu_add(cpu_residual, cpu_down);

        Runtime runtime = create_runtime();
        std::cout << "Selected Vulkan device: " << runtime.properties.deviceName << "\n"
                  << "Vendor/device ID: 0x" << std::hex << runtime.properties.vendorID << ":0x"
                  << runtime.properties.deviceID << std::dec << "\n"
                  << "Compute queue family: " << runtime.queue_family << "\n";

        Buffer dummy = create_buffer(runtime, sizeof(float));
        Buffer input_buffer = upload_vector(runtime, input);
        Buffer attention_gamma_buffer = upload_vector(runtime, attention_gamma);
        Buffer feed_forward_gamma_buffer = upload_vector(runtime, feed_forward_gamma);
        Buffer rope_cos_buffer = upload_vector(runtime, rope_cos);
        Buffer rope_sin_buffer = upload_vector(runtime, rope_sin);
        Buffer qkv_weight_buffer = upload_vector(runtime, qkv_weights.packed);
        Buffer qkv_scale_buffer = upload_vector(runtime, qkv_weights.scales);
        Buffer output_weight_buffer = upload_vector(runtime, output_weights.packed);
        Buffer output_scale_buffer = upload_vector(runtime, output_weights.scales);
        Buffer gate_up_weight_buffer = upload_vector(runtime, gate_up_weights.packed);
        Buffer gate_up_scale_buffer = upload_vector(runtime, gate_up_weights.scales);
        Buffer down_weight_buffer = upload_vector(runtime, down_weights.packed);
        Buffer down_scale_buffer = upload_vector(runtime, down_weights.scales);
        Buffer norm1_buffer = create_buffer(runtime, cpu_norm1.size() * sizeof(float));
        Buffer qkv_buffer = create_buffer(runtime, cpu_qkv.size() * sizeof(float));
        Buffer rotated_qkv_buffer = create_buffer(runtime, cpu_rotated_qkv.size() * sizeof(float));
        Buffer context_buffer = create_buffer(runtime, cpu_context.size() * sizeof(float));
        Buffer attention_projection_buffer = create_buffer(runtime, cpu_attention_projection.size() * sizeof(float));
        Buffer residual_buffer = create_buffer(runtime, cpu_residual.size() * sizeof(float));
        Buffer norm2_buffer = create_buffer(runtime, cpu_norm2.size() * sizeof(float));
        Buffer gate_up_buffer = create_buffer(runtime, cpu_gate_up.size() * sizeof(float));
        Buffer feed_forward_buffer = create_buffer(runtime, cpu_feed_forward.size() * sizeof(float));
        Buffer down_buffer = create_buffer(runtime, cpu_down.size() * sizeof(float));
        Buffer output_buffer = create_buffer(runtime, cpu_output.size() * sizeof(float));

        constexpr uint32_t DISPATCH_COUNT = 11;
        ComputeResources resources = create_compute_resources(runtime, DISPATCH_COUNT);
        const auto shader = [&](const char* name) {
            return shader_directory + "\\" + name + ".comp.spv";
        };
        const VkPipeline qgemm_pipeline = create_pipeline(runtime, resources, shader("qgemm"));
        const VkPipeline rmsnorm_pipeline = create_pipeline(runtime, resources, shader("rmsnorm"));
        const VkPipeline rope_pipeline = create_pipeline(runtime, resources, shader("rope"));
        const VkPipeline attention_pipeline = create_pipeline(runtime, resources, shader("attention"));
        const VkPipeline add_pipeline = create_pipeline(runtime, resources, shader("add"));
        const VkPipeline swiglu_pipeline = create_pipeline(runtime, resources, shader("swiglu"));

        const VkDescriptorSet norm1_set = create_descriptor_set(runtime, resources,
            {&input_buffer, &attention_gamma_buffer, &norm1_buffer, &dummy});
        const VkDescriptorSet qkv_set = create_descriptor_set(runtime, resources,
            {&norm1_buffer, &qkv_weight_buffer, &qkv_scale_buffer, &qkv_buffer});
        const VkDescriptorSet rope_set = create_descriptor_set(runtime, resources,
            {&qkv_buffer, &rope_cos_buffer, &rope_sin_buffer, &rotated_qkv_buffer});
        const VkDescriptorSet attention_set = create_descriptor_set(runtime, resources,
            {&rotated_qkv_buffer, &context_buffer, &dummy, &dummy});
        const VkDescriptorSet output_projection_set = create_descriptor_set(runtime, resources,
            {&context_buffer, &output_weight_buffer, &output_scale_buffer, &attention_projection_buffer});
        const VkDescriptorSet attention_residual_set = create_descriptor_set(runtime, resources,
            {&input_buffer, &attention_projection_buffer, &residual_buffer, &dummy});
        const VkDescriptorSet norm2_set = create_descriptor_set(runtime, resources,
            {&residual_buffer, &feed_forward_gamma_buffer, &norm2_buffer, &dummy});
        const VkDescriptorSet gate_up_set = create_descriptor_set(runtime, resources,
            {&norm2_buffer, &gate_up_weight_buffer, &gate_up_scale_buffer, &gate_up_buffer});
        const VkDescriptorSet swiglu_set = create_descriptor_set(runtime, resources,
            {&gate_up_buffer, &feed_forward_buffer, &dummy, &dummy});
        const VkDescriptorSet down_set = create_descriptor_set(runtime, resources,
            {&feed_forward_buffer, &down_weight_buffer, &down_scale_buffer, &down_buffer});
        const VkDescriptorSet output_set = create_descriptor_set(runtime, resources,
            {&residual_buffer, &down_buffer, &output_buffer, &dummy});

        VkCommandPoolCreateInfo command_pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        command_pool_info.queueFamilyIndex = runtime.queue_family;
        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkfn::CreateCommandPool(runtime.device, &command_pool_info, nullptr, &command_pool));
        VkCommandBufferAllocateInfo command_allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_allocation.commandPool = command_pool;
        command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocation.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VK_CHECK(vkfn::AllocateCommandBuffers(runtime.device, &command_allocation, &command_buffer));
        VkCommandBufferBeginInfo begin_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkfn::BeginCommandBuffer(command_buffer, &begin_info));
        VkMemoryBarrier upload_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        upload_barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        upload_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkfn::CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &upload_barrier, 0, nullptr, 0, nullptr);

        const RmsPush rms1_push{TOKENS, MODEL, RMS_EPSILON, 0};
        dispatch(command_buffer, resources, rmsnorm_pipeline, norm1_set, &rms1_push, TOKENS, 1);
        compute_barrier(command_buffer);
        const LinearPush qkv_push{TOKENS, 3 * MODEL, MODEL, qkv_weights.packed_stride};
        dispatch(command_buffer, resources, qgemm_pipeline, qkv_set, &qkv_push,
                 (3 * MODEL + 15) / 16, (TOKENS + 15) / 16);
        compute_barrier(command_buffer);
        const RopePush rope_push{TOKENS, MODEL, HEAD_DIMENSION, 3 * MODEL};
        dispatch(command_buffer, resources, rope_pipeline, rope_set, &rope_push,
                 (3 * MODEL + 63) / 64, TOKENS);
        compute_barrier(command_buffer);
        const AttentionPush attention_push{TOKENS, MODEL, HEADS, HEAD_DIMENSION};
        dispatch(command_buffer, resources, attention_pipeline, attention_set, &attention_push,
                 TOKENS, HEADS);
        compute_barrier(command_buffer);
        const LinearPush output_projection_push{TOKENS, MODEL, MODEL, output_weights.packed_stride};
        dispatch(command_buffer, resources, qgemm_pipeline, output_projection_set,
                 &output_projection_push, (MODEL + 15) / 16, (TOKENS + 15) / 16);
        compute_barrier(command_buffer);
        const AddPush residual_push{TOKENS * MODEL, 0, 0, 0};
        dispatch(command_buffer, resources, add_pipeline, attention_residual_set, &residual_push,
                 (TOKENS * MODEL + 63) / 64, 1);
        compute_barrier(command_buffer);
        const RmsPush rms2_push{TOKENS, MODEL, RMS_EPSILON, 0};
        dispatch(command_buffer, resources, rmsnorm_pipeline, norm2_set, &rms2_push, TOKENS, 1);
        compute_barrier(command_buffer);
        const LinearPush gate_up_push{TOKENS, 2 * FEED_FORWARD, MODEL,
                                      gate_up_weights.packed_stride};
        dispatch(command_buffer, resources, qgemm_pipeline, gate_up_set, &gate_up_push,
                 (2 * FEED_FORWARD + 15) / 16, (TOKENS + 15) / 16);
        compute_barrier(command_buffer);
        const SwiGluPush swiglu_push{TOKENS, FEED_FORWARD, 0, 0};
        dispatch(command_buffer, resources, swiglu_pipeline, swiglu_set, &swiglu_push,
                 (TOKENS * FEED_FORWARD + 63) / 64, 1);
        compute_barrier(command_buffer);
        const LinearPush down_push{TOKENS, MODEL, FEED_FORWARD, down_weights.packed_stride};
        dispatch(command_buffer, resources, qgemm_pipeline, down_set, &down_push,
                 (MODEL + 15) / 16, (TOKENS + 15) / 16);
        compute_barrier(command_buffer);
        dispatch(command_buffer, resources, add_pipeline, output_set, &residual_push,
                 (TOKENS * MODEL + 63) / 64, 1);

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
        VK_CHECK(vkfn::QueueSubmit(runtime.queue, 1, &submit, VK_NULL_HANDLE));
        VK_CHECK(vkfn::QueueWaitIdle(runtime.queue));

        invalidate_buffer(runtime, output_buffer);
        std::vector<float> gpu_output(cpu_output.size());
        std::memcpy(gpu_output.data(), output_buffer.mapped, static_cast<size_t>(output_buffer.size));
        float maximum_absolute_error = 0.0f;
        float maximum_relative_error = 0.0f;
        double square_error_sum = 0.0;
        size_t failures = 0;
        for (size_t index = 0; index < gpu_output.size(); ++index) {
            const float absolute_error = std::abs(gpu_output[index] - cpu_output[index]);
            const float relative_error = absolute_error / std::max(1.0e-6f, std::abs(cpu_output[index]));
            maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
            maximum_relative_error = std::max(maximum_relative_error, relative_error);
            square_error_sum += static_cast<double>(absolute_error) * absolute_error;
            const float tolerance = 2.0e-3f + 2.0e-4f * std::abs(cpu_output[index]);
            if (!std::isfinite(gpu_output[index]) || absolute_error > tolerance) ++failures;
        }
        const double root_mean_square_error = std::sqrt(square_error_sum / gpu_output.size());
        std::cout << std::fixed << std::setprecision(8)
                  << "Block: tokens=" << TOKENS << ", model=" << MODEL << ", heads=" << HEADS
                  << ", head_dim=" << HEAD_DIMENSION << ", ffn=" << FEED_FORWARD << "\n"
                  << "GPU compute dispatches: " << DISPATCH_COUNT << "\n"
                  << "Linear weights: symmetric INT8 per-output-channel, packed 4/uint32\n"
                  << "GPU vs CPU max abs error: " << maximum_absolute_error << "\n"
                  << "GPU vs CPU max rel error: " << maximum_relative_error << "\n"
                  << "GPU vs CPU RMSE: " << root_mean_square_error << "\n"
                  << "Tolerance failures: " << failures << " / " << gpu_output.size() << "\n"
                  << (failures == 0 ? "RESULT: PASS - complete Vulkan transformer block is correct\n"
                                    : "RESULT: FAIL - transformer output exceeded tolerance\n");

        vkfn::DestroyCommandPool(runtime.device, command_pool, nullptr);
        for (VkPipeline pipeline : resources.pipelines) vkfn::DestroyPipeline(runtime.device, pipeline, nullptr);
        for (VkShaderModule module : resources.shader_modules) vkfn::DestroyShaderModule(runtime.device, module, nullptr);
        vkfn::DestroyDescriptorPool(runtime.device, resources.descriptor_pool, nullptr);
        vkfn::DestroyPipelineLayout(runtime.device, resources.pipeline_layout, nullptr);
        vkfn::DestroyDescriptorSetLayout(runtime.device, resources.descriptor_layout, nullptr);
        std::array<Buffer*, 25> all_buffers = {
            &output_buffer, &down_buffer, &feed_forward_buffer, &gate_up_buffer, &norm2_buffer,
            &residual_buffer, &attention_projection_buffer, &context_buffer, &rotated_qkv_buffer,
            &qkv_buffer, &norm1_buffer, &down_scale_buffer, &down_weight_buffer,
            &gate_up_scale_buffer, &gate_up_weight_buffer, &output_scale_buffer,
            &output_weight_buffer, &qkv_scale_buffer, &qkv_weight_buffer, &rope_sin_buffer,
            &rope_cos_buffer, &feed_forward_gamma_buffer, &attention_gamma_buffer,
            &input_buffer, &dummy};
        for (Buffer* buffer : all_buffers) destroy_buffer(runtime, *buffer);
        vkfn::DestroyDevice(runtime.device, nullptr);
        vkfn::DestroyInstance(runtime.instance, nullptr);
        FreeLibrary(runtime.loader);
        return failures == 0 ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
#endif
