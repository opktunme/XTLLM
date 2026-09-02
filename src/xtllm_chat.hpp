#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace xtllm_chat {

enum class Role : uint8_t { user = 1, assistant = 2 };

struct Message {
    Role role{};
    std::string content;
};

inline constexpr std::string_view kReferencePrefix = "@XTLLM_CHAT_FILE:";
inline constexpr char kMagic[8] = {'X','T','C','H','A','T','1','\0'};

inline bool referenced_path(const std::string& value,
                            std::filesystem::path& path) {
    const char* enabled = std::getenv("XTLLM_STRUCTURED_CHAT");
    if (!enabled || std::string_view(enabled) != "1" ||
        value.compare(0, kReferencePrefix.size(), kReferencePrefix) != 0)
        return false;
    path = value.substr(kReferencePrefix.size());
    if (path.empty()) throw std::runtime_error("XTLLM chat transcript path is empty");
    return true;
}

template <typename T>
inline T read_scalar(std::ifstream& input, const char* label) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error(std::string("Truncated XTLLM chat ") + label);
    return value;
}

inline std::vector<Message> read(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Could not open XTLLM chat transcript");
    char magic[8]{};
    input.read(magic, sizeof(magic));
    if (!input || std::memcmp(magic, kMagic, sizeof(magic)) != 0)
        throw std::runtime_error("Invalid XTLLM chat transcript header");
    const uint32_t count = read_scalar<uint32_t>(input, "message count");
    if (!count || count > 64u)
        throw std::runtime_error("XTLLM chat transcript message count is invalid");
    std::vector<Message> result;
    result.reserve(count);
    uint64_t total = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const uint8_t role_value = read_scalar<uint8_t>(input, "role");
        const uint32_t bytes = read_scalar<uint32_t>(input, "content length");
        if (role_value < uint8_t(Role::user) ||
            role_value > uint8_t(Role::assistant) || bytes > 256u * 1024u ||
            total + bytes > 512u * 1024u)
            throw std::runtime_error("XTLLM chat transcript record is invalid");
        Message message{static_cast<Role>(role_value), std::string(bytes, '\0')};
        input.read(message.content.data(), static_cast<std::streamsize>(bytes));
        if (!input) throw std::runtime_error("Truncated XTLLM chat content");
        if (message.role != (index % 2u == 0u ? Role::user : Role::assistant))
            throw std::runtime_error("XTLLM chat roles must alternate from user");
        total += bytes;
        result.push_back(std::move(message));
    }
    if (result.back().role != Role::user)
        throw std::runtime_error("XTLLM chat transcript must end with a user turn");
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("XTLLM chat transcript has trailing data");
    return result;
}

enum class Suffix { qwen_think, nemotron_think, none };

template <typename Encoder>
inline std::vector<uint32_t> render_im(
    const std::vector<Message>& messages, Encoder encode, uint32_t im_start,
    uint32_t im_end, uint32_t think, uint32_t end_think, bool thinking,
    Suffix suffix, bool force_system = false,
    const std::string& system_text = {}) {
    std::vector<uint32_t> result;
    const auto text = [&](const std::string& value) {
        const std::vector<uint32_t> encoded = encode(value);
        result.insert(result.end(), encoded.begin(), encoded.end());
    };
    if (force_system || !system_text.empty()) {
        result.push_back(im_start);
        text("system\n" + system_text);
        result.push_back(im_end);
        text("\n");
    }
    for (const Message& message : messages) {
        result.push_back(im_start);
        text(std::string(message.role == Role::user ? "user\n" : "assistant\n") +
             message.content);
        result.push_back(im_end);
        text("\n");
    }
    result.push_back(im_start);
    text("assistant\n");
    if (suffix == Suffix::none) return result;
    result.push_back(think);
    if (thinking) {
        text("\n");
    } else if (suffix == Suffix::qwen_think) {
        text("\n\n");
        result.push_back(end_think);
        text("\n\n");
    } else {
        result.push_back(end_think);
    }
    return result;
}

template <typename Encoder>
inline std::vector<uint32_t> render_longcat(
    const std::vector<Message>& messages, Encoder encode, uint32_t user,
    uint32_t assistant, uint32_t end) {
    std::vector<uint32_t> result;
    for (const Message& message : messages) {
        result.push_back(message.role == Role::user ? user : assistant);
        const std::vector<uint32_t> encoded = encode(message.content);
        result.insert(result.end(), encoded.begin(), encoded.end());
        if (message.role == Role::assistant) result.push_back(end);
    }
    result.push_back(assistant);
    return result;
}

template <typename Encoder>
inline std::vector<uint32_t> render_deepseek(
    const std::vector<Message>& messages, Encoder encode, uint32_t bos,
    uint32_t eos, uint32_t user, uint32_t assistant, uint32_t think,
    uint32_t end_think, bool thinking, bool textual_roles,
    const std::string& system_text = {}) {
    std::vector<uint32_t> result{bos};
    const auto text = [&](const std::string& value) {
        const std::vector<uint32_t> encoded = encode(value);
        result.insert(result.end(), encoded.begin(), encoded.end());
    };
    if (textual_roles) {
        if (!system_text.empty()) {
            result.push_back(user);
            text("system\n" + system_text);
            result.push_back(assistant);
            text("\n");
        }
        for (const Message& message : messages) {
            result.push_back(user);
            text(std::string(message.role == Role::user ? "user\n" :
                             "assistant\n") + message.content);
            result.push_back(assistant);
            text("\n");
        }
        result.push_back(user);
        text("assistant\n");
        result.push_back(thinking ? think : end_think);
        text("\n");
        return result;
    }
    if (!system_text.empty()) text(system_text);
    for (const Message& message : messages) {
        if (message.role == Role::user) {
            result.push_back(user);
            text(message.content);
        } else {
            result.push_back(assistant);
            text(message.content);
            result.push_back(eos);
        }
    }
    result.push_back(assistant);
    result.push_back(thinking ? think : end_think);
    return result;
}

} // namespace xtllm_chat
