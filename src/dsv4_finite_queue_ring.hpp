#pragma once

// Experimental bounded command-buffer ring for DeepSeek expert pipelining.
// This file is deliberately not wired into m13.  It keeps every submission
// finite, waits before reusing a command buffer, and never polls from a shader.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>

namespace dsv4::experiment {

struct QueueRingApi {
    PFN_vkCreateCommandPool create_command_pool = nullptr;
    PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
    PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
    PFN_vkResetCommandBuffer reset_command_buffer = nullptr;
    PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
    PFN_vkEndCommandBuffer end_command_buffer = nullptr;
    PFN_vkCreateSemaphore create_semaphore = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;
    PFN_vkQueueSubmit queue_submit = nullptr;
    PFN_vkWaitSemaphores wait_semaphores = nullptr;
};

inline void queue_ring_check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed (VkResult " +
                                 std::to_string(static_cast<int>(result)) + ")");
}

template <size_t RingSize = 8>
class FiniteQueueRing {
    static_assert(RingSize >= 2, "A queue ring needs at least two command buffers");

public:
    static constexpr uint64_t kDefaultTimeoutNs = 10ull * 1000 * 1000 * 1000;

    FiniteQueueRing(VkDevice device, VkQueue queue, uint32_t queue_family,
                    QueueRingApi api, std::mutex* shared_queue_mutex = nullptr,
                    uint64_t timeout_ns = kDefaultTimeoutNs)
        : device_(device), queue_(queue), api_(api),
          queue_mutex_(shared_queue_mutex), timeout_ns_(timeout_ns) {
        validate_api();
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = queue_family;
        queue_ring_check(api_.create_command_pool(device_, &pool_info, nullptr, &pool_),
                         "vkCreateCommandPool");

        VkCommandBufferAllocateInfo allocate{
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool_;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = static_cast<uint32_t>(RingSize);
        queue_ring_check(api_.allocate_command_buffers(device_, &allocate,
                                                       commands_.data()),
                         "vkAllocateCommandBuffers");

        VkSemaphoreTypeCreateInfo timeline_type{
            VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timeline_type.initialValue = 0;
        VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        semaphore_info.pNext = &timeline_type;
        queue_ring_check(api_.create_semaphore(device_, &semaphore_info, nullptr,
                                               &timeline_),
                         "vkCreateSemaphore");
    }

    FiniteQueueRing(const FiniteQueueRing&) = delete;
    FiniteQueueRing& operator=(const FiniteQueueRing&) = delete;

    ~FiniteQueueRing() {
        // Destruction is also bounded.  A timeout intentionally does not turn
        // into an unbounded QueueWaitIdle; callers should drain() and surface
        // failures before normal destruction.
        if (timeline_ && submitted_ > completed_) {
            VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait_info.semaphoreCount = 1;
            wait_info.pSemaphores = &timeline_;
            wait_info.pValues = &submitted_;
            if (api_.wait_semaphores(device_, &wait_info, timeout_ns_) == VK_SUCCESS)
                completed_ = submitted_;
        }
        if (timeline_) api_.destroy_semaphore(device_, timeline_, nullptr);
        if (pool_) api_.destroy_command_pool(device_, pool_, nullptr);
    }

    template <class Record>
    uint64_t submit(Record&& record,
                    VkSemaphore wait_semaphore = VK_NULL_HANDLE,
                    uint64_t wait_value = 0,
                    VkPipelineStageFlags wait_stage =
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) {
        const size_t slot = next_slot_;
        if (reuse_after_[slot] > completed_) wait(reuse_after_[slot]);

        VkCommandBuffer command = commands_[slot];
        queue_ring_check(api_.reset_command_buffer(command, 0),
                         "vkResetCommandBuffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        queue_ring_check(api_.begin_command_buffer(command, &begin),
                         "vkBeginCommandBuffer");
        record(command);
        queue_ring_check(api_.end_command_buffer(command), "vkEndCommandBuffer");

        const uint64_t signal_value = ++submitted_;
        VkTimelineSemaphoreSubmitInfo timeline_info{
            VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        timeline_info.signalSemaphoreValueCount = 1;
        timeline_info.pSignalSemaphoreValues = &signal_value;
        if (wait_semaphore) {
            timeline_info.waitSemaphoreValueCount = 1;
            timeline_info.pWaitSemaphoreValues = &wait_value;
        }

        VkSubmitInfo submit_info{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit_info.pNext = &timeline_info;
        if (wait_semaphore) {
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = &wait_semaphore;
            submit_info.pWaitDstStageMask = &wait_stage;
        }
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command;
        submit_info.signalSemaphoreCount = 1;
        submit_info.pSignalSemaphores = &timeline_;

        if (queue_mutex_) {
            std::lock_guard<std::mutex> lock(*queue_mutex_);
            queue_ring_check(api_.queue_submit(queue_, 1, &submit_info, VK_NULL_HANDLE),
                             "vkQueueSubmit");
        } else {
            queue_ring_check(api_.queue_submit(queue_, 1, &submit_info, VK_NULL_HANDLE),
                             "vkQueueSubmit");
        }
        reuse_after_[slot] = signal_value;
        next_slot_ = (slot + 1) % RingSize;
        return signal_value;
    }

    void wait(uint64_t value) {
        if (value <= completed_) return;
        if (value > submitted_)
            throw std::runtime_error("FiniteQueueRing wait exceeds submitted value");
        VkSemaphoreWaitInfo wait_info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &timeline_;
        wait_info.pValues = &value;
        const VkResult result =
            api_.wait_semaphores(device_, &wait_info, timeout_ns_);
        if (result == VK_TIMEOUT)
            throw std::runtime_error("FiniteQueueRing submission exceeded bounded timeout");
        queue_ring_check(result, "vkWaitSemaphores");
        completed_ = std::max(completed_, value);
    }

    void drain() { wait(submitted_); }
    VkSemaphore timeline() const { return timeline_; }
    uint64_t submitted() const { return submitted_; }
    uint64_t completed() const { return completed_; }

private:
    void validate_api() const {
        if (!device_ || !queue_ || !api_.create_command_pool ||
            !api_.destroy_command_pool || !api_.allocate_command_buffers ||
            !api_.reset_command_buffer || !api_.begin_command_buffer ||
            !api_.end_command_buffer || !api_.create_semaphore ||
            !api_.destroy_semaphore || !api_.queue_submit ||
            !api_.wait_semaphores)
            throw std::runtime_error("FiniteQueueRing received an incomplete Vulkan API");
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    QueueRingApi api_{};
    std::mutex* queue_mutex_ = nullptr;
    uint64_t timeout_ns_ = kDefaultTimeoutNs;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, RingSize> commands_{};
    std::array<uint64_t, RingSize> reuse_after_{};
    VkSemaphore timeline_ = VK_NULL_HANDLE;
    size_t next_slot_ = 0;
    uint64_t submitted_ = 0;
    uint64_t completed_ = 0;
};

}  // namespace dsv4::experiment
