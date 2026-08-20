#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace ovllm_trace {

#pragma pack(push, 1)
struct Header {
    char magic[8];
    uint32_t version;
    uint32_t model;
    uint32_t layers;
    uint32_t experts;
    uint32_t top_k;
    uint32_t record_bytes;
    uint32_t device_slots;
    uint32_t ram_slots;
    uint64_t reserved[3];
};

// Experts are uint8 because every supported routed model has exactly 256.
// Masks describe tier residency immediately after the authoritative router is
// visible and before this layer's acquisition changes placement.
struct Event {
    uint64_t trace_us;
    uint32_t position;
    uint16_t layer;
    uint8_t phase; // 0 prefill, 1 decode
    uint8_t top_k;
    uint8_t expert[8];
    uint8_t device_mask;
    uint8_t ram_mask;
    uint8_t disk_mask;
    uint8_t transit_mask;
    uint16_t device_slot[8];
    uint32_t acquire_us;
    uint32_t complete_us;
    float weight[8];
};
#pragma pack(pop)

static_assert(sizeof(Header) == 64);
static_assert(sizeof(Event) == 84);

class Writer {
public:
    Writer() = default;
    ~Writer() {
        if (!enabled()) return;
        std::ofstream stream(path_, std::ios::binary | std::ios::trunc);
        if (!stream) return;
        stream.write(reinterpret_cast<const char*>(&header_), sizeof(header_));
        if (!events_.empty())
            stream.write(reinterpret_cast<const char*>(events_.data()),
                         static_cast<std::streamsize>(events_.size() * sizeof(Event)));
    }

    void open(uint32_t model, uint32_t layers, uint32_t experts,
              uint32_t top_k, uint32_t record_bytes, uint32_t device_slots,
              uint32_t ram_slots) {
        const char* path = std::getenv("OVLLM_EXPERT_TRACE");
        if (!path || !*path) return;
        path_ = std::filesystem::path(path);
        std::memcpy(header_.magic, "OVEXTRC1", 8);
        header_.version = 1;
        header_.model = model;
        header_.layers = layers;
        header_.experts = experts;
        header_.top_k = top_k;
        header_.record_bytes = record_bytes;
        header_.device_slots = device_slots;
        header_.ram_slots = ram_slots;
        events_.reserve(32768);
        origin_ = Clock::now();
    }

    bool enabled() const { return !path_.empty(); }
    void set_decode(bool decode) { decode_ = decode; }

    template <size_t K>
    void event(uint32_t position, uint32_t layer,
               const std::array<uint32_t, K>& experts,
               const std::array<float, K>& weights,
               uint8_t device_mask, uint8_t ram_mask, uint8_t disk_mask,
               const std::array<uint32_t, K>& slots,
               std::chrono::steady_clock::time_point request_started,
               std::chrono::steady_clock::time_point completed) {
        static_assert(K <= 8);
        if (!enabled()) return;
        Event value{};
        value.trace_us = micros(request_started - origin_);
        value.position = position;
        value.layer = static_cast<uint16_t>(layer);
        value.phase = decode_ ? 1u : 0u;
        value.top_k = static_cast<uint8_t>(K);
        value.device_mask = device_mask;
        value.ram_mask = ram_mask;
        value.disk_mask = disk_mask;
        value.transit_mask = static_cast<uint8_t>(ram_mask | disk_mask);
        value.acquire_us = static_cast<uint32_t>(micros(completed - request_started));
        value.complete_us = static_cast<uint32_t>(micros(completed - origin_));
        for (size_t i = 0; i < 8; ++i) value.device_slot[i] = 0xffffu;
        for (size_t i = 0; i < K; ++i) {
            value.expert[i] = static_cast<uint8_t>(experts[i]);
            value.weight[i] = weights[i];
            value.device_slot[i] = slots[i] > 0xffffu
                ? 0xffffu : static_cast<uint16_t>(slots[i]);
        }
        events_.push_back(value);
    }

private:
    using Clock = std::chrono::steady_clock;
    template <typename Duration>
    static uint64_t micros(Duration duration) {
        const auto count = std::chrono::duration_cast<
            std::chrono::microseconds>(duration).count();
        return count > 0 ? static_cast<uint64_t>(count) : 0ull;
    }

    std::filesystem::path path_;
    Header header_{};
    std::vector<Event> events_;
    Clock::time_point origin_{};
    bool decode_ = false;
};

} // namespace ovllm_trace
