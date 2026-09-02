from __future__ import annotations

from pathlib import Path
import struct
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import ovllm_chat_server as chat  # noqa: E402


class ChatTranscriptTests(unittest.TestCase):
    def test_conversation_remains_structured(self):
        messages = [
            {"role": "user", "content": "Remember PURPLE."},
            {"role": "assistant", "content": "Understood."},
            {"role": "user", "content": "What word?"},
        ]
        self.assertEqual(chat.State.conversation_messages(messages), [
            ("user", "Remember PURPLE."),
            ("assistant", "Understood."),
            ("user", "What word?"),
        ])

    def test_binary_transcript_preserves_roles_and_utf8(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chat.bin"
            chat.write_chat_transcript(path, [
                ("user", "héllo"),
                ("assistant", "مرحبا"),
                ("user", "continue"),
            ])
            data = path.read_bytes()
        self.assertEqual(data[:8], chat.CHAT_MAGIC)
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 3)
        offset = 12
        decoded = []
        for _ in range(3):
            role, size = struct.unpack_from("<BI", data, offset)
            offset += 5
            decoded.append((role, data[offset:offset + size].decode("utf-8")))
            offset += size
        self.assertEqual(decoded, [
            (1, "héllo"), (2, "مرحبا"), (1, "continue")])
        self.assertEqual(offset, len(data))

    def test_invalid_role_order_is_rejected(self):
        with self.assertRaises(ValueError):
            chat.State.conversation_messages([
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ])

    def test_long_conversation_keeps_complete_role_pairs(self):
        messages = []
        for index in range(6):
            messages.extend([
                {"role": "user", "content": f"u{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ])
        messages.append({"role": "user", "content": "latest"})
        retained = chat.State.conversation_messages(messages)
        self.assertEqual(len(retained), 9)
        self.assertEqual(retained[0], ("user", "u2"))
        self.assertEqual(retained[-1], ("user", "latest"))


if __name__ == "__main__":
    unittest.main()
