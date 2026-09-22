from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ovllm  # noqa: E402


def runtime_args(**overrides):
    values = {
        "context_gib": None,
        "context_tokens": None,
        "long_mode": "auto",
        "ram_gib": None,
        "device_slots": None,
        "no_prewarm": False,
        "think": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class ProfileTests(unittest.TestCase):
    def test_full_is_default_for_integrated_models(self):
        for key in ("qwen38", "qwencoder", "longcat"):
            model = ovllm.find_model(key)
            name, profile = ovllm.select_profile(model, None)
            self.assertEqual(name, "full")
            self.assertIn("backend", profile)

    def test_exact_alias_selects_reference(self):
        model = ovllm.find_model("qwen38")
        name, _ = ovllm.select_profile(model, "exact")
        self.assertEqual(name, "reference")

    def test_qwen_full_clears_relaxed_fast_environment(self):
        model = ovllm.find_model("qwen38")
        _, full = ovllm.select_profile(model, "full")
        with patch.dict(os.environ, {
            "QWEN38_RELAXED_DRAFTS": "3",
            "QWEN38_VERIFY4_ACTIVE_TOPK": "7",
            "QWEN38_PARALLEL_HOST_COPY": "1",
            "QWEN38_VERIFY4_EXPERIMENT": "1",
        }, clear=False):
            environment, settings = ovllm.standalone_environment(
                model, full, runtime_args())
        self.assertNotIn("QWEN38_RELAXED_DRAFTS", environment)
        self.assertNotIn("QWEN38_VERIFY4_ACTIVE_TOPK", environment)
        self.assertNotIn("QWEN38_VERIFY4_EXPERIMENT", environment)
        self.assertNotIn("QWEN38_PARALLEL_HOST_COPY", environment)
        self.assertEqual(settings["QWEN38_DEVICE_SLOTS_PER_LAYER"], "62")
        self.assertEqual(settings["QWEN38_DWARFSTAR"], "1")
        self.assertEqual(settings["QWEN38_DWARF_PREFILL4"], "1")
        self.assertEqual(full["backend"], "xtllm-qwen38-flash-next.exe")

    def test_qwen_full_needs_no_q3_or_mtp_conversion(self):
        model = ovllm.find_model("qwen38")
        _, full = ovllm.select_profile(model, "full")
        requirements = ovllm.profile_requirements(model, full)
        self.assertFalse(any("q3" in item["path"] or "mtp" in item["path"]
                             for item in requirements))
        steps = [step["script"] for step in model["conversion"]
                 if "profiles" not in step or "full" in step["profiles"]]
        self.assertEqual(steps, ["convert_qwen38.py"])

    def test_qwen_reference_and_legacy_mtp_remain_isolated(self):
        model = ovllm.find_model("qwen38")
        for name in ("reference", "legacy-mtp", "fast"):
            _, profile = ovllm.select_profile(model, name)
            with patch.dict(os.environ, {"QWEN38_DWARFSTAR": "1",
                                         "QWEN38_DWARF_PREFILL4": "1"}):
                environment, _ = ovllm.standalone_environment(
                    model, profile, runtime_args())
            self.assertNotIn("QWEN38_DWARFSTAR", environment)
            self.assertNotIn("QWEN38_DWARF_PREFILL4", environment)
        legacy = model["profiles"]["legacy-mtp"]
        self.assertEqual(legacy["env"]["QWEN38_VERIFY4_ACTIVE_TOPK"], "10")
        self.assertEqual(legacy["default_slots"], 72)

    def test_qwen_full_respects_memory_overrides_and_small_cache(self):
        model = ovllm.find_model("qwen38")
        _, full = ovllm.select_profile(model, "full")
        for slots in (10, 39, 40, 62):
            settings = ovllm.standalone_settings(
                model, full, runtime_args(ram_gib=24, device_slots=slots))
            self.assertEqual(settings["QWEN38_RAM_GIB"], "24")
            self.assertEqual(settings["QWEN38_DEVICE_SLOTS_PER_LAYER"], str(slots))
            self.assertEqual(settings["QWEN38_DWARFSTAR"], "1")
            self.assertEqual("QWEN38_DWARF_PREFILL4" in settings, slots >= 40)

    def test_longcat_profiles_keep_distinct_shared_containers(self):
        model = ovllm.find_model("longcat")
        _, reference = ovllm.select_profile(model, "reference")
        _, full = ovllm.select_profile(model, "full")
        reference_paths = {
            item["path"] for item in ovllm.profile_requirements(model, reference)
        }
        full_paths = {
            item["path"] for item in ovllm.profile_requirements(model, full)
        }
        self.assertIn("model-q4g64.ovs", reference_paths)
        self.assertIn("model-hybrid-q4g64.ovs", full_paths)
        self.assertEqual(
            full["env"]["LONGCAT_SHARED_MODEL"], "model-hybrid-q4g64.ovs")

    def test_longcat_context_override_reaches_native_runtime(self):
        model = ovllm.find_model("longcat")
        _, full = ovllm.select_profile(model, "full")
        environment, settings = ovllm.standalone_environment(
            model, full, runtime_args(context_tokens=4096))
        self.assertEqual(settings["LONGCAT_CONTEXT_TOKENS"], "4096")
        self.assertEqual(environment["LONGCAT_CONTEXT_TOKENS"], "4096")

    def test_other_standalone_context_override_remains_rejected(self):
        model = ovllm.find_model("qwen38")
        _, full = ovllm.select_profile(model, "full")
        with self.assertRaises(ovllm.UserError):
            ovllm.standalone_settings(
                model, full, runtime_args(context_tokens=4096))

    def test_longcat_context_override_must_be_positive(self):
        model = ovllm.find_model("longcat")
        _, full = ovllm.select_profile(model, "full")
        with self.assertRaises(ovllm.UserError):
            ovllm.standalone_settings(
                model, full, runtime_args(context_tokens=0))

    def test_longcat_source_has_no_compiled_256_token_context_cap(self):
        source = (ROOT / "src" / "longcat_flash_lite.cpp").read_text(
            encoding="utf-8")
        shader = (ROOT / "shaders" / "longcat_mla_attention.comp").read_text(
            encoding="utf-8")
        self.assertNotIn("CTX=256", source)
        self.assertNotIn("scores[256]", shader)

    def test_fast_profiles_carry_quality_warnings(self):
        for key in ("qwen38", "qwencoder", "longcat"):
            model = ovllm.find_model(key)
            _, fast = ovllm.select_profile(model, "fast")
            self.assertTrue(fast.get("warning"))

    def test_profile_conversion_scripts_are_present(self):
        for model in ovllm.MODELS:
            for step in model["conversion"]:
                script = ROOT / "tools" / step["script"]
                self.assertTrue(script.is_file(), script)


if __name__ == "__main__":
    unittest.main()
