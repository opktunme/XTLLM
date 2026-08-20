#define OVLLM_QWEN35_RUNTIME_ONLY 1
#define OVLLM_QWEN36_RUNTIME_ONLY 1
#define OVLLM_NEMOTRON3_RUNTIME_ONLY 1
#define OVLLM_LONG_CONTEXT_FORK 1
#include "m25_qwen35_longctx.cpp"
#include "m17_qwen36.cpp"
#include "m18_nemotron3.cpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <string_view>

namespace ovllm {

enum class Model { Qwen35, DeepSeekV4, Qwen36, Nemotron3 };

struct Options {
    std::filesystem::path executable;
    std::filesystem::path runtime_directory;
    std::string prompt;
    uint32_t tokens = 24;
    std::optional<double> ram_gib;
    std::optional<double> context_gib;
    std::optional<uint32_t> context_tokens;
    std::optional<uint32_t> device_slots;
    bool prewarm = true;
    bool qwen_thinking = false;
    bool plan_only = false;
    bool context_stress = false;
    std::string long_mode = "exact";
    std::optional<double> simulated_available_gib;
};

struct HostMemory {
    double installed_gib = 0.0;
    double available_gib = 0.0;
};

struct DeviceMemory {
    std::string name;
    double heap_gib = 0.0;
    double budget_gib = 0.0;
    double usage_gib = 0.0;
};

static constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

static void set_environment(const char* name, const std::string& value) {
    if (_putenv_s(name, value.c_str()) != 0)
        throw std::runtime_error(std::string("Could not set environment: ") + name);
}

static void clear_environment(const char* name) {
    if (_putenv_s(name, "") != 0)
        throw std::runtime_error(std::string("Could not clear environment: ") + name);
}

static HostMemory host_memory() {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
        throw std::runtime_error("GlobalMemoryStatusEx failed");
    return {double(status.ullTotalPhys) / kGiB,
            double(status.ullAvailPhys) / kGiB};
}

static bool supports_device_extension(const Runtime& runtime,
                                      const char* requested) {
    uint32_t count = 0;
    if (vkfn::EnumerateDeviceExtensionProperties(
            runtime.physical, nullptr, &count, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkfn::EnumerateDeviceExtensionProperties(
            runtime.physical, nullptr, &count, extensions.data()) != VK_SUCCESS)
        return false;
    for (const auto& extension : extensions)
        if (std::strcmp(extension.extensionName, requested) == 0) return true;
    return false;
}

static DeviceMemory device_memory() {
    Runtime runtime = create_runtime();
    DeviceMemory result{};
    try {
        result.name = runtime.properties.deviceName;
        uint32_t device_heap = UINT32_MAX;
        for (uint32_t i = 0; i < runtime.memory_properties.memoryHeapCount; ++i) {
            const auto& heap = runtime.memory_properties.memoryHeaps[i];
            if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                (device_heap == UINT32_MAX ||
                 heap.size > runtime.memory_properties.memoryHeaps[device_heap].size))
                device_heap = i;
        }
        if (device_heap == UINT32_MAX)
            throw std::runtime_error("AMD Vulkan device has no device-local heap");
        result.heap_gib = double(
            runtime.memory_properties.memoryHeaps[device_heap].size) / kGiB;

        if (supports_device_extension(runtime, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
            const auto get_memory_properties2 =
                reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties2>(
                    vkfn::GetInstanceProcAddr(
                        runtime.instance, "vkGetPhysicalDeviceMemoryProperties2"));
            if (get_memory_properties2) {
                VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
                VkPhysicalDeviceMemoryProperties2 properties{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
                properties.pNext = &budget;
                get_memory_properties2(runtime.physical, &properties);
                result.budget_gib = double(budget.heapBudget[device_heap]) / kGiB;
                result.usage_gib = double(budget.heapUsage[device_heap]) / kGiB;
            }
        }
    } catch (...) {
        if (runtime.device) vkfn::DestroyDevice(runtime.device, nullptr);
        if (runtime.instance) vkfn::DestroyInstance(runtime.instance, nullptr);
        if (runtime.loader) FreeLibrary(runtime.loader);
        throw;
    }
    vkfn::DestroyDevice(runtime.device, nullptr);
    vkfn::DestroyInstance(runtime.instance, nullptr);
    FreeLibrary(runtime.loader);
    return result;
}

static std::array<char, 8> read_magic(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Missing model container: " + path.string());
    std::array<char, 8> magic{};
    stream.read(magic.data(), magic.size());
    if (stream.gcount() != static_cast<std::streamsize>(magic.size()))
        throw std::runtime_error("Truncated model container: " + path.string());
    return magic;
}

static Model detect_model(const std::filesystem::path& directory) {
    const std::filesystem::path q4 = directory / "model-q4g64.ovs";
    if (std::filesystem::exists(q4)) {
        const auto magic = read_magic(q4);
        if (std::memcmp(magic.data(), "OQ35SHR\0", 8) == 0) return Model::Qwen35;
        if (std::memcmp(magic.data(), "OVD4SHR\0", 8) == 0) return Model::DeepSeekV4;
        if (std::memcmp(magic.data(), "OQ36SHR\0", 8) == 0) return Model::Qwen36;
    }
    const std::filesystem::path nvfp4 = directory / "model-nvfp4.ovs";
    if (std::filesystem::exists(nvfp4)) {
        const auto magic = read_magic(nvfp4);
        if (std::memcmp(magic.data(), "ON3NSHR\0", 8) == 0) return Model::Nemotron3;
    }
    throw std::runtime_error(
        "Unsupported runtime container (expected Qwen3.5, Qwen3.6, "
        "DeepSeek-V4, or Nemotron-3-Nano)");
}

static const char* model_name(Model model) {
    switch (model) {
        case Model::Qwen35: return "Qwen3.5-122B-A10B";
        case Model::DeepSeekV4: return "DeepSeek-V4-Flash-0731";
        case Model::Qwen36: return "Qwen3.6-35B-A3B";
        case Model::Nemotron3: return "Nemotron-3-Nano-30B-A3B";
    }
    return "unknown";
}

static double automatic_ram_budget(Model model, const HostMemory& memory,
                                   std::optional<double> simulated_available) {
    const double available = simulated_available.value_or(memory.available_gib);
    const double reserve = std::max(4.0, memory.installed_gib * 0.0625);
    double usable = std::floor(std::max(2.0, available - reserve) * 4.0) / 4.0;
    // On a 64-GiB machine, 49.25 GiB is the measured Qwen sweet spot: it
    // retains 85% of all routed experts while leaving roughly 5 GiB of real
    // headroom.  Larger hosts may use the backend's 56-GiB validated ceiling.
    const double qwen_cap = memory.installed_gib < 80.0 ? 49.25 : 56.0;
    double model_cap = 24.0;
    if (model == Model::Qwen35) model_cap = qwen_cap;
    else if (model == Model::Qwen36 || model == Model::Nemotron3)
        model_cap = 20.0;
    usable = std::min(usable, model_cap);
    return std::max(2.0, usable);
}

struct VramSizing {
    uint64_t live_bytes = 0;
    uint64_t fixed_bytes = 0;
    uint64_t per_slot_bytes = 0;
    uint64_t headroom_bytes = 0;
    uint32_t slots = 0;
};

static uint64_t live_device_bytes(const DeviceMemory& memory) {
    const double budget = memory.budget_gib > 0.0 ?
        std::min(memory.heap_gib, memory.budget_gib) : memory.heap_gib;
    const double remaining = std::max(0.0, budget - memory.usage_gib);
    return static_cast<uint64_t>(remaining * kGiB);
}

static uint64_t display_driver_headroom(uint64_t live_bytes) {
    // Scale headroom with the currently usable heap, while avoiding both an
    // unsafe tiny reserve and excessive waste on larger accelerator cards.
    return std::clamp<uint64_t>(live_bytes * 3u / 40u,
                                512ull << 20, 1536ull << 20);
}

static uint32_t slots_that_fit(uint64_t live, uint64_t fixed,
                               uint64_t per_slot, uint64_t headroom,
                               uint32_t minimum, uint32_t maximum) {
    if (live <= fixed + headroom)
        return minimum;
    const uint64_t count = (live - fixed - headroom) / per_slot;
    return static_cast<uint32_t>(std::clamp<uint64_t>(count, minimum, maximum));
}

static VramSizing automatic_qwen122_sizing(const DeviceMemory& memory) {
    VramSizing sizing;
    sizing.live_bytes = live_device_bytes(memory);
    sizing.fixed_bytes = 4268670688ull; // measured non-expert Vulkan allocations
    sizing.per_slot_bytes = qwen35::kExpertRecordBytes * qwen35::kLayers;
    sizing.headroom_bytes = display_driver_headroom(sizing.live_bytes);
    sizing.slots = slots_that_fit(sizing.live_bytes, sizing.fixed_bytes,
        sizing.per_slot_bytes, sizing.headroom_bytes, qwen35::kTopK, 32u);
    if (memory.heap_gib <= 12.5 && sizing.slots >= 28u) sizing.slots = 28u;
    return sizing;
}

static VramSizing automatic_qwen36_sizing(const DeviceMemory& memory) {
    VramSizing sizing;
    sizing.live_bytes = live_device_bytes(memory);
    sizing.fixed_bytes = 1945939623ull; // measured non-expert Vulkan allocations
    sizing.per_slot_bytes = qwen36::kExpertRecordBytes * qwen36::kLayers;
    sizing.headroom_bytes = display_driver_headroom(sizing.live_bytes);
    sizing.slots = slots_that_fit(sizing.live_bytes, sizing.fixed_bytes,
        sizing.per_slot_bytes, sizing.headroom_bytes, qwen36::kTopK, 128u);
    // Preserve the proven RX 6700 XT cache, but do not cap larger heaps at 96.
    if (memory.heap_gib <= 12.5 && sizing.slots >= 96u) sizing.slots = 96u;
    return sizing;
}

static VramSizing automatic_nemotron_sizing(const DeviceMemory& memory) {
    VramSizing sizing;
    sizing.live_bytes = live_device_bytes(memory);
    sizing.fixed_bytes = 1690121352ull; // measured non-expert Vulkan allocations
    sizing.per_slot_bytes = nemotron3::kExpertRecordBytes * nemotron3::kMoeLayers;
    sizing.headroom_bytes = display_driver_headroom(sizing.live_bytes);
    sizing.slots = slots_that_fit(sizing.live_bytes, sizing.fixed_bytes,
        sizing.per_slot_bytes, sizing.headroom_bytes, nemotron3::kTopK, 128u);
    // Preserve the proven RX 6700 XT cache, but do not cap larger heaps at 60.
    if (memory.heap_gib <= 12.5 && sizing.slots >= 60u) sizing.slots = 60u;
    return sizing;
}

static VramSizing automatic_deepseek_sizing(const DeviceMemory& memory) {
    VramSizing sizing;
    sizing.live_bytes = live_device_bytes(memory);
    sizing.fixed_bytes = 4455888867ull; // measured non-expert Vulkan allocations
    sizing.per_slot_bytes = dsv4::kExpertRecordBytes;
    sizing.headroom_bytes = display_driver_headroom(sizing.live_bytes);
    const uint64_t total = sizing.live_bytes > sizing.fixed_bytes +
            sizing.headroom_bytes ?
        (sizing.live_bytes - sizing.fixed_bytes - sizing.headroom_bytes) /
            sizing.per_slot_bytes : 0;
    sizing.slots = static_cast<uint32_t>(std::min<uint64_t>(total, 559u));
    return sizing;
}

static uint32_t reduced_retry_slots(Model model, uint32_t slots) {
    if (model == Model::DeepSeekV4 && slots > 13u) {
        const uint32_t reduced_total = slots - std::max(1u, slots / 8u);
        return std::clamp(reduced_total / dsv4::kLayers, 6u, 13u);
    }
    const uint32_t minimum = model == Model::Nemotron3 ?
        nemotron3::kTopK : (model == Model::DeepSeekV4 ?
        dsv4::kTopK : 8u);
    const uint32_t reduction = std::max<uint32_t>(1u, slots / 8u);
    return std::max(minimum, slots > reduction ? slots - reduction : minimum);
}

static double parse_double(const char* text, const char* option) {
    try { return std::stod(text); }
    catch (...) { throw std::runtime_error(std::string("Invalid ") + option); }
}

static uint32_t parse_u32(const char* text, const char* option) {
    try {
        const unsigned long value = std::stoul(text);
        if (value > UINT32_MAX) throw std::out_of_range("u32");
        return static_cast<uint32_t>(value);
    } catch (...) { throw std::runtime_error(std::string("Invalid ") + option); }
}

static Options parse_options(int argc, char** argv) {
    Options options;
    options.executable = std::filesystem::absolute(argv[0]);
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--ram-gib") {
            if (++i >= argc) throw std::runtime_error("--ram-gib needs a value");
            options.ram_gib = parse_double(argv[i], "--ram-gib");
        } else if (argument == "--context-gib") {
            if (++i >= argc) throw std::runtime_error("--context-gib needs a value");
            options.context_gib = parse_double(argv[i], "--context-gib");
        } else if (argument == "--context-tokens") {
            if (++i >= argc) throw std::runtime_error("--context-tokens needs a value");
            options.context_tokens = parse_u32(argv[i], "--context-tokens");
        } else if (argument == "--long-mode") {
            if (++i >= argc) throw std::runtime_error("--long-mode needs a value");
            options.long_mode = argv[i];
            if (options.long_mode != "exact" && options.long_mode != "fast" &&
                options.long_mode != "auto")
                throw std::runtime_error("--long-mode must be exact, fast, or auto");
        } else if (argument == "--device-slots") {
            if (++i >= argc) throw std::runtime_error("--device-slots needs a value");
            options.device_slots = parse_u32(argv[i], "--device-slots");
        } else if (argument == "--tokens") {
            if (++i >= argc) throw std::runtime_error("--tokens needs a value");
            options.tokens = parse_u32(argv[i], "--tokens");
        } else if (argument == "--think") {
            options.qwen_thinking = true;
        } else if (argument == "--no-prewarm") {
            options.prewarm = false;
        } else if (argument == "--plan") {
            options.plan_only = true;
        } else if (argument == "--context-stress") {
            options.context_stress = true;
        } else if (argument == "--simulate-available-ram-gib") {
            if (++i >= argc)
                throw std::runtime_error(
                    "--simulate-available-ram-gib needs a value");
            options.simulated_available_gib =
                parse_double(argv[i], "--simulate-available-ram-gib");
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("Unknown option: " + std::string(argument));
        } else {
            positional.emplace_back(argument);
        }
    }
    if (positional.empty())
        throw std::runtime_error("A runtime directory is required");
    options.runtime_directory = positional[0];
    if (positional.size() >= 2) options.prompt = positional[1];
    if (positional.size() >= 3)
        options.tokens = parse_u32(positional[2].c_str(), "new-token count");
    if (positional.size() > 3)
        throw std::runtime_error("Too many positional arguments");
    if (!options.plan_only && options.prompt.empty())
        throw std::runtime_error("A prompt is required (or use --plan)");
    return options;
}

static void configure_qwen(const Options& options, double ram_gib,
                           uint32_t slots) {
    set_environment("QWEN_RAM_GIB", std::to_string(ram_gib));
    set_environment("QWEN_DEVICE_SLOTS_PER_LAYER", std::to_string(slots));
    if (options.context_gib)
        set_environment("QWEN_CONTEXT_GIB", std::to_string(*options.context_gib));
    else
        clear_environment("QWEN_CONTEXT_GIB");
    if (options.context_tokens)
        set_environment("QWEN_CONTEXT_TOKENS", std::to_string(*options.context_tokens));
    else
        clear_environment("QWEN_CONTEXT_TOKENS");
    set_environment("QWEN_LONG_MODE", options.long_mode);
    if (options.context_stress) set_environment("QWEN_LONG_CONTEXT_STRESS", "1");
    else clear_environment("QWEN_LONG_CONTEXT_STRESS");
    set_environment("QWEN_PROGRESSIVE_EXPERTS", "1");
    // Qwen's record layout places gate/up before down.  The retained split
    // schedule starts exact gate/up work after the first aligned region is
    // ready while the down suffix transfers; the fixed mixed-prompt suite
    // improved 4.123 -> 4.220 tok/s with byte-identical output and traffic.
    set_environment("QWEN_TENSOR_SPLIT_EXPERIMENT", "1");
    clear_environment("QWEN_VERIFY2_EXPERIMENT");
    if (options.prewarm) set_environment("QWEN_FILL_RAM_CACHE", "1");
    else clear_environment("QWEN_FILL_RAM_CACHE");
    if (options.qwen_thinking) clear_environment("QWEN_NO_THINK");
    else set_environment("QWEN_NO_THINK", "1");
    // The measured Q4-BDA batch and TinyLFU experiments lost to the retained
    // progressive path.  Do not let stale research-shell variables reactivate them.
    clear_environment("QWEN_EXPERT_BATCH_BDA");
    clear_environment("QWEN_HOST_TINYLFU");
    // The acquisition lab's mixed-prompt and held-out validation both reduced
    // cold readiness with this exact-budget profile on the proven 28-slot
    // RX 6700 XT layout. Larger/smaller VRAM layouts keep their dynamically
    // sized uniform policy until independently profiled.
    if (slots == 28u) {
        set_environment("QWEN_DEVICE_SLOT_PROFILE",
            "13,26,32,32,31,25,32,22,31,27,30,27,29,18,29,31,21,21,26,28,"
            "27,29,25,31,32,31,31,32,31,32,25,30,31,32,25,25,23,32,30,31,"
            "26,31,31,26,22,31,29,32");
        set_environment("QWEN_HOST_LRU", "1");
    } else {
        clear_environment("QWEN_DEVICE_SLOT_PROFILE");
        clear_environment("QWEN_HOST_LRU");
    }
}

static void configure_qwen36(const Options& options, double ram_gib,
                             uint32_t slots) {
    set_environment("QWEN36_RAM_GIB", std::to_string(ram_gib));
    set_environment("QWEN36_DEVICE_SLOTS_PER_LAYER", std::to_string(slots));
    set_environment("QWEN36_PROGRESSIVE_EXPERTS", "1");
    if (options.context_gib)
        set_environment("QWEN36_CONTEXT_GIB", std::to_string(*options.context_gib));
    else clear_environment("QWEN36_CONTEXT_GIB");
    if (options.context_tokens)
        set_environment("QWEN36_CONTEXT_TOKENS", std::to_string(*options.context_tokens));
    else clear_environment("QWEN36_CONTEXT_TOKENS");
    if (options.context_stress) set_environment("QWEN36_LONG_CONTEXT_STRESS", "1");
    else clear_environment("QWEN36_LONG_CONTEXT_STRESS");
    if (options.prewarm) set_environment("QWEN36_FILL_RAM_CACHE", "1");
    else clear_environment("QWEN36_FILL_RAM_CACHE");
    if (options.qwen_thinking) clear_environment("QWEN36_NO_THINK");
    else set_environment("QWEN36_NO_THINK", "1");
}

static void configure_nemotron3(const Options& options, double ram_gib,
                                uint32_t slots) {
    set_environment("NEMOTRON3_RAM_GIB", std::to_string(ram_gib));
    set_environment("NEMOTRON3_DEVICE_SLOTS_PER_LAYER", std::to_string(slots));
    clear_environment("NEMOTRON3_NO_PROGRESSIVE");
    if (options.context_gib)
        set_environment("NEMOTRON3_CONTEXT_GIB", std::to_string(*options.context_gib));
    else clear_environment("NEMOTRON3_CONTEXT_GIB");
    if (options.context_tokens)
        set_environment("NEMOTRON3_CONTEXT_TOKENS", std::to_string(*options.context_tokens));
    else clear_environment("NEMOTRON3_CONTEXT_TOKENS");
    if (options.context_stress) set_environment("NEMOTRON3_LONG_CONTEXT_STRESS", "1");
    else clear_environment("NEMOTRON3_LONG_CONTEXT_STRESS");
    if (options.prewarm) set_environment("NEMOTRON3_FILL_RAM_CACHE", "1");
    else clear_environment("NEMOTRON3_FILL_RAM_CACHE");
    if (options.qwen_thinking) clear_environment("NEMOTRON3_NO_THINK");
    else set_environment("NEMOTRON3_NO_THINK", "1");
}

static void configure_deepseek(const Options& options, double ram_gib,
                               bool suite_profile, uint32_t uniform_slots) {
    set_environment("DSV4_RAM_GIB", std::to_string(ram_gib));
    set_environment("DSV4_Q4_EXPERTS", "1");
    set_environment("DSV4_Q4_SWAR", "1");
    set_environment("DSV4_SHARED_R1X4", "1");
    set_environment("DSV4_Q4_EXPERT_BATCH_BDA", "1");
    set_environment("DSV4_HC_FUSED", "1");
    set_environment("DSV4_RMS_ROPE_FUSED", "1");
    set_environment("DSV4_PROGRESSIVE_EXPERTS", "1");
    set_environment("DSV4_EXPERT_BLUEPRINT", "1");
    set_environment("DSV4_INDEX_INT8", "1");
    if (options.context_gib)
        set_environment("DSV4_CONTEXT_GIB", std::to_string(*options.context_gib));
    else
        clear_environment("DSV4_CONTEXT_GIB");
    if (options.context_tokens)
        set_environment("DSV4_CONTEXT_TOKENS", std::to_string(*options.context_tokens));
    else
        clear_environment("DSV4_CONTEXT_TOKENS");
    set_environment("DSV4_LONG_MODE", options.long_mode);
    if (options.context_stress) {
        set_environment("DSV4_LONG_CONTEXT_STRESS", "1");
        clear_environment("DSV4_LONG_CONTEXT_REAL");
    } else {
        clear_environment("DSV4_LONG_CONTEXT_STRESS");
        // A configured long-context allocation must also enable the real
        // learned-history execution path.  Previously only the synthetic
        // stress benchmark set a long-context execution flag, so ordinary
        // chat still tripped the retained 127-token short-path guard.
        if (options.context_gib || options.context_tokens)
            set_environment("DSV4_LONG_CONTEXT_REAL", "1");
        else
            clear_environment("DSV4_LONG_CONTEXT_REAL");
    }
    if (options.prewarm) set_environment("DSV4_FILL_RAM_CACHE", "1");
    else clear_environment("DSV4_FILL_RAM_CACHE");

    if (suite_profile) {
        set_environment("DSV4_SUITE_PROFILED_DEVICE_CACHE", "1");
        clear_environment("DSV4_CACHE_SLOTS");
    } else {
        clear_environment("DSV4_SUITE_PROFILED_DEVICE_CACHE");
        set_environment("DSV4_CACHE_SLOTS", std::to_string(uniform_slots));
    }
    // AMD's canonical Vulkan-host tier is the fast path up to 24 GiB.  Larger
    // manual budgets require plain RAM and its explicit bounded staging copy.
    if (ram_gib > 24.0 || options.context_gib)
        set_environment("DSV4_PLAIN_HOST_CACHE", "1");
    else clear_environment("DSV4_PLAIN_HOST_CACHE");

    const char* rejected[] = {
        "DSV4_EXTERNAL_HOST", "DSV4_LOCAL_HOST_CACHE",
        "DSV4_HYBRID_HOST_CACHE", "DSV4_IMPORT_PLAIN_L2",
        "DSV4_FOREIGN_L2_CHUNK", "DSV4_FOREIGN_L2_BLOCKS",
        "DSV4_HOST_TINYLFU", "DSV4_TOP6_SET_POLICY",
        "DSV4_ADJACENT_RANK1_HINT", "DSV4_SHARED_ACQUIRE_OVERLAP",
        "DSV4_EXPERT_BUNDLE_INDEX", "DSV4_EXPERT_BUNDLE_STORE",
        "DSV4_GLOBAL_DEVICE_CACHE", "DSV4_FORCE_CACHE_ADMISSION"
    };
    for (const char* name : rejected) clear_environment(name);
}

static int run_backend(Model model, const Options& options) {
    std::string executable = options.executable.string();
    std::string directory = std::filesystem::absolute(
        options.runtime_directory).string();
    std::string tokens = std::to_string(options.tokens);
    std::array<char*, 4> backend_argv{
        executable.data(), directory.data(),
        const_cast<char*>(options.prompt.c_str()), tokens.data()};
    switch (model) {
        case Model::Qwen35:
            return qwen35_cli_main(4, backend_argv.data());
        case Model::DeepSeekV4:
            return dsv4_cli_main(4, backend_argv.data());
        case Model::Qwen36:
            return qwen36_cli_main(4, backend_argv.data());
        case Model::Nemotron3:
            return nemotron3_cli_main(4, backend_argv.data());
    }
    return 1;
}

} // namespace ovllm

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr
                << "usage: xtllm.exe [options] <runtime-dir> <prompt> [new-tokens]\n"
                << "options: --ram-gib N --context-gib N --context-tokens N "
                   "--long-mode exact|fast|auto --device-slots N "
                   "--tokens N --think --context-stress "
                   "--no-prewarm --plan\n";
            return 2;
        }
        const ovllm::Options options = ovllm::parse_options(argc, argv);
        const ovllm::Model model = ovllm::detect_model(options.runtime_directory);
        const ovllm::HostMemory host = ovllm::host_memory();
        const ovllm::DeviceMemory device = ovllm::device_memory();
        const double available = options.simulated_available_gib.value_or(
            host.available_gib);
        const double ram_gib = options.ram_gib.value_or(
            ovllm::automatic_ram_budget(model, host,
                                         options.simulated_available_gib));
        if (ram_gib < 2.0 || ram_gib > 56.0)
            throw std::runtime_error("Selected RAM budget must be 2..56 GiB");
        const double context_gib = options.context_gib.value_or(0.0);
        if (context_gib < 0.0 || context_gib > 16.0)
            throw std::runtime_error("Selected context budget must be 0..16 GiB");
        if (!options.simulated_available_gib &&
            ram_gib + context_gib > host.available_gib - 2.0)
            throw std::runtime_error(
                "Selected RAM budget leaves less than 2 GiB currently available");

        uint32_t selected_slots = 0;
        bool suite_profile = false;
        ovllm::VramSizing vram_sizing{};
        if (model == ovllm::Model::Qwen35) {
            vram_sizing = ovllm::automatic_qwen122_sizing(device);
            selected_slots = options.device_slots.value_or(vram_sizing.slots);
            if (selected_slots < 8 || selected_slots > 32)
                throw std::runtime_error("Qwen device slots must be 8..32");
            ovllm::configure_qwen(options, ram_gib, selected_slots);
        } else if (model == ovllm::Model::Qwen36) {
            vram_sizing = ovllm::automatic_qwen36_sizing(device);
            selected_slots = options.device_slots.value_or(vram_sizing.slots);
            if (selected_slots < 8 || selected_slots > 128)
                throw std::runtime_error("Qwen3.6 device slots must be 8..128");
            ovllm::configure_qwen36(options, ram_gib, selected_slots);
        } else if (model == ovllm::Model::Nemotron3) {
            vram_sizing = ovllm::automatic_nemotron_sizing(device);
            selected_slots = options.device_slots.value_or(vram_sizing.slots);
            if (selected_slots < 6 || selected_slots > 128)
                throw std::runtime_error("Nemotron device slots must be 6..128");
            ovllm::configure_nemotron3(options, ram_gib, selected_slots);
        } else {
            vram_sizing = ovllm::automatic_deepseek_sizing(device);
            suite_profile = !options.device_slots && vram_sizing.slots >= 460u;
            const uint32_t uniform = static_cast<uint32_t>(std::clamp<uint64_t>(
                vram_sizing.slots / dsv4::kLayers, 6u, 13u));
            selected_slots = options.device_slots.value_or(
                suite_profile ? 460u : uniform);
            if (!suite_profile && (selected_slots < 6 || selected_slots > 13))
                throw std::runtime_error("DeepSeek uniform device slots must be 6..13");
            ovllm::configure_deepseek(options, ram_gib, suite_profile,
                                      selected_slots);
        }

        std::cout << std::fixed << std::setprecision(3)
                  << "XTLLM standalone engine\n"
                  << "model: " << ovllm::model_name(model) << " (auto-detected)\n"
                  << "system RAM installed/available: " << host.installed_gib
                  << " / " << available << " GiB\n"
                  << "selected inference RAM budget: " << ram_gib << " GiB"
                  << (options.prewarm ? " (prewarm enabled)\n" :
                                        " (prewarm disabled)\n")
                  << "Vulkan device: " << device.name << "\n"
                  << "device-local VRAM heap/budget/current usage: "
                  << device.heap_gib << " / " << device.budget_gib << " / "
                  << device.usage_gib << " GiB\n"
                  << "VRAM sizing fixed/per-slot/headroom: "
                  << double(vram_sizing.fixed_bytes) / ovllm::kGiB << " / "
                  << double(vram_sizing.per_slot_bytes) / ovllm::kGiB << " / "
                  << double(vram_sizing.headroom_bytes) / ovllm::kGiB
                  << " GiB\n";
        if (model != ovllm::Model::DeepSeekV4)
            std::cout << "selected expert VRAM cache: " << selected_slots
                      << " records per layer\n";
        else if (suite_profile)
            std::cout << "selected expert VRAM cache: 460 records, "
                         "profiled across 43 layers\n";
        else
            std::cout << "selected expert VRAM cache: " << selected_slots
                      << " records per layer\n";

        if (options.plan_only) return 0;
        last_device_allocation_oom = false;
        const int first = ovllm::run_backend(model, options);
        if (first == 0 || options.device_slots || !last_device_allocation_oom)
            return first;

        const uint32_t retry_slots = ovllm::reduced_retry_slots(
            model, selected_slots);
        if (retry_slots >= selected_slots) return first;
        std::cerr << "XTLLM: clean device-allocation OOM; retrying once with "
                  << retry_slots << " expert slots per layer\n";
        last_device_allocation_oom = false;
        if (model == ovllm::Model::Qwen35)
            ovllm::configure_qwen(options, ram_gib, retry_slots);
        else if (model == ovllm::Model::Qwen36)
            ovllm::configure_qwen36(options, ram_gib, retry_slots);
        else if (model == ovllm::Model::Nemotron3)
            ovllm::configure_nemotron3(options, ram_gib, retry_slots);
        else
            ovllm::configure_deepseek(options, ram_gib, false, retry_slots);
        return ovllm::run_backend(model, options);
    } catch (const std::exception& error) {
        std::cerr << "XTLLM error: " << error.what() << '\n';
        return 1;
    }
}
