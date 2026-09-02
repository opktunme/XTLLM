#!/usr/bin/env python3
"""Friendly setup/run/chat front-end for the native XTLLM Vulkan engine.

This module keeps its historical filename as a compatibility import. New users
should invoke xtllm.py or xtllm.cmd.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "config" / "models.json"
GIB = 1024 ** 3


class UserError(RuntimeError):
    pass


def load_registry() -> list[dict[str, Any]]:
    try:
        value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise UserError(f"Could not read model registry: {REGISTRY_PATH}: {error}")
    if value.get("schema") != 1 or not isinstance(value.get("models"), list):
        raise UserError("Unsupported model registry schema")
    return value["models"]


MODELS = load_registry()


def find_model(value: str) -> dict[str, Any]:
    needle = value.casefold()
    for model in MODELS:
        choices = [model["key"], model["name"], model["repo"], *model["aliases"]]
        if needle in (choice.casefold() for choice in choices):
            return model
    available = ", ".join(model["key"] for model in MODELS)
    raise UserError(f"Unknown model '{value}'. Available aliases: {available}")


def select_profile(model: dict[str, Any], value: str | None) -> tuple[str | None, dict[str, Any]]:
    profiles = model.get("profiles")
    if not profiles:
        if value:
            raise UserError(f"{model['name']} does not define inference profiles")
        return None, {}
    aliases = {"exact": "reference", "exact_ref": "reference", "default": model["default_profile"]}
    requested = aliases.get((value or model["default_profile"]).casefold(),
                            (value or model["default_profile"]).casefold())
    if requested not in profiles:
        available = ", ".join(profiles)
        raise UserError(
            f"Unknown profile '{value}' for {model['name']}. Available: {available}"
        )
    return requested, profiles[requested]


def profile_requirements(model: dict[str, Any],
                         profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [*model["required"], *((profile or {}).get("required", []))]


def model_root(option: str | None) -> Path:
    configured = option or os.environ.get("XTLLM_MODELS") or os.environ.get("OVLLM_MODELS")
    return Path(configured).expanduser().resolve() if configured else ROOT / "models"


def model_paths(model: dict[str, Any], option: str | None) -> tuple[Path, Path, Path]:
    base = model_root(option) / model["slug"]
    return base, base / "checkpoint", base / "runtime"


def find_engine(model: dict[str, Any] | None = None,
                profile: dict[str, Any] | None = None) -> Path:
    executable_name = (profile or {}).get(
        "backend", model.get("backend", "xtllm.exe") if model else "xtllm.exe")
    configured = os.environ.get("XTLLM_ENGINE") or os.environ.get("OVLLM_ENGINE")
    configured_candidates: list[Path] = []
    if configured:
        override = Path(configured)
        configured_candidates.append(
            override if executable_name == "xtllm.exe" else override.parent / executable_name)
    candidates = configured_candidates + [
        ROOT / executable_name,
        ROOT / "build" / executable_name,
        ROOT / "dist" / "xtllm" / executable_name,
    ]
    if executable_name == "xtllm.exe":
        candidates += [
        ROOT / "ovllm-longctx.exe",  # legacy package compatibility
        ROOT / "build" / "ovllm-longctx.exe",
        ROOT / "build" / "ovllm_longctx.exe",
        ROOT / "dist" / "ovllm-longctx" / "ovllm-longctx.exe",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise UserError(
        f"XTLLM backend {executable_name} was not found. Use a GitHub release package or run "
        "scripts\\build-windows.ps1. XTLLM_ENGINE can override the path."
    )


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def complete_runtime(model: dict[str, Any], runtime: Path,
                     profile: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for requirement in profile_requirements(model, profile):
        candidate = runtime / requirement["path"]
        if not candidate.is_file() or candidate.stat().st_size < requirement["min_bytes"]:
            missing.append(requirement["path"])
    return not missing, missing


def format_command(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def run_checked(command: list[str], *, env: dict[str, str] | None = None,
                dry_run: bool = False) -> None:
    print(f"> {format_command(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def hf_command(*, allow_missing: bool = False) -> list[str]:
    local_name = "hf.exe" if os.name == "nt" else "hf"
    local = Path(sys.executable).parent / local_name
    executable = str(local) if local.is_file() else shutil.which("hf")
    if executable:
        return [executable]
    if allow_missing:
        return ["hf"]
    raise UserError(
        "Hugging Face CLI was not found. Run scripts\\install-windows.ps1, "
        "or: python -m pip install 'huggingface_hub[cli]'"
    )


def expand_args(values: list[str], checkpoint: Path, runtime: Path) -> list[str]:
    replacements = {
        "checkpoint": str(checkpoint),
        "runtime": str(runtime),
        "workers": str(max(1, min(8, (os.cpu_count() or 2) // 2))),
    }
    return [value.format(**replacements) for value in values]


def command_models(_: argparse.Namespace) -> int:
    print("Supported native backends:\n")
    for model in MODELS:
        print(f"  {model['key']:<10} {model['name']}")
        print(f"             {model['repo']} @ {model['revision'][:12]}")
        if model.get("profiles"):
            choices = ", ".join(
                f"{name}{' (default)' if name == model['default_profile'] else ''}"
                for name in model["profiles"])
            print(f"             profiles: {choices}")
    return 0


def command_doctor(args: argparse.Namespace) -> int:
    print(f"XTLLM root: {ROOT}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {sys.platform}")
    checks: list[tuple[str, bool, str]] = []
    try:
        engine = find_engine()
        shaders = list(engine.parent.glob("*.comp.spv"))
        checks.append(("native engine", True, str(engine)))
        checks.append(("SPIR-V shaders", len(shaders) >= 20,
                       f"{len(shaders)} beside executable"))
    except UserError as error:
        checks.append(("native engine", False, str(error)))
    profile_backends: dict[str, Path] = {}
    missing_backends: list[str] = []
    for model in MODELS:
        for profile in model.get("profiles", {}).values():
            name = profile.get("backend", model.get("backend", "xtllm.exe"))
            if name in profile_backends or name in missing_backends:
                continue
            try:
                profile_backends[name] = find_engine(model, profile)
            except UserError:
                missing_backends.append(name)
    if profile_backends or missing_backends:
        detail = (f"{len(profile_backends)} found" if not missing_backends else
                  f"missing: {', '.join(missing_backends)}")
        checks.append(("profile backends", not missing_backends, detail))
    try:
        hf = hf_command()
        checks.append(("Hugging Face CLI", True, hf[0]))
    except UserError:
        checks.append(("Hugging Face CLI", False,
                       "not found; run scripts\\install-windows.ps1"))
    try:
        import numpy  # type: ignore
        checks.append(("NumPy", True, numpy.__version__))
    except ImportError:
        checks.append(("NumPy", False, "not installed"))
    root = model_root(args.model_root)
    free = shutil.disk_usage(existing_parent(root)).free / GIB
    checks.append(("model storage", free >= 25.0, f"{free:.1f} GiB free at {root}"))
    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}")
    return 0 if all(passed for _, passed, _ in checks) else 1


def command_setup(args: argparse.Namespace) -> int:
    model = find_model(args.model)
    profile_name, profile = select_profile(model, args.profile)
    base, checkpoint, runtime = model_paths(model, args.model_root)
    complete, missing = complete_runtime(model, runtime, profile)
    if complete:
        suffix = f" ({profile_name} profile)" if profile_name else ""
        print(f"{model['name']}{suffix} is ready: {runtime}")
        return 0

    checkpoint_existing = tree_bytes(checkpoint) / GIB
    runtime_existing = tree_bytes(runtime) / GIB
    required = max(0.0, model["download_gib"] - checkpoint_existing)
    runtime_gib = profile.get("runtime_gib", model["runtime_gib"])
    required += max(0.0, runtime_gib - runtime_existing)
    required += model["scratch_gib"]
    free = shutil.disk_usage(existing_parent(base)).free / GIB
    print(f"Model: {model['name']}")
    if profile_name:
        print(f"Profile: {profile_name} - {profile['label']}")
    print(f"Official checkpoint: {model['repo']} @ {model['revision']}")
    print(f"Destination: {base}")
    print(f"Missing runtime assets: {', '.join(missing)}")
    print(f"Estimated additional disk required: {required:.1f} GiB; free: {free:.1f} GiB")
    if free < required:
        raise UserError(
            f"Insufficient disk space. Approximately {required:.1f} GiB is still required."
        )
    if not args.yes and not args.dry_run:
        if not sys.stdin.isatty():
            raise UserError("Use --yes for non-interactive setup")
        answer = input("Download and convert this checkpoint? [y/N] ").strip().casefold()
        if answer not in ("y", "yes"):
            print("Cancelled.")
            return 0

    if not args.dry_run:
        checkpoint.mkdir(parents=True, exist_ok=True)
        runtime.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        environment = os.environ.copy()
        if args.hf_token:
            environment["HF_TOKEN"] = args.hf_token
        command = [*hf_command(allow_missing=args.dry_run), "download", model["repo"], "--revision",
                   model["revision"], "--local-dir", str(checkpoint)]
        run_checked(command, env=environment, dry_run=args.dry_run)

    for step in model["conversion"]:
        if "profiles" in step and profile_name not in step["profiles"]:
            continue
        outputs = [runtime / item for item in step["when_missing"]]
        if all(item.is_file() for item in outputs):
            print(f"Skipping completed conversion step: {step['script']}")
            continue
        script = ROOT / "tools" / step["script"]
        if not script.is_file():
            raise UserError(f"Converter is missing: {script}")
        command = [sys.executable, str(script),
                   *expand_args(step["args"], checkpoint, runtime)]
        run_checked(command, dry_run=args.dry_run)

    if args.dry_run:
        print("Dry run complete; no download or conversion was performed.")
        return 0
    complete, missing = complete_runtime(model, runtime, profile)
    if not complete:
        raise UserError(f"Conversion finished without required assets: {', '.join(missing)}")
    manifest = {
        "schema": 1,
        "model": model["name"],
        "source": model["repo"],
        "revision": model["revision"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile_name,
        "files": {item["path"]: (runtime / item["path"]).stat().st_size
                  for item in profile_requirements(model, profile)},
    }
    temporary = runtime / "xtllm-model.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, runtime / "xtllm-model.json")
    print(f"Ready: {runtime}")
    profile_option = f" --profile {profile_name}" if profile_name else ""
    print(f"Run: xtllm run {model['key']} \"Explain why the sky is blue.\"{profile_option}")
    return 0


def runtime_for(model: dict[str, Any], option: str | None,
                profile: dict[str, Any] | None = None) -> Path:
    runtime = model_paths(model, option)[2]
    complete, missing = complete_runtime(model, runtime, profile)
    if not complete:
        raise UserError(
            f"{model['name']} is not set up ({', '.join(missing)} missing). "
            f"Run: xtllm setup {model['key']} --model-root \"{model_root(option)}\""
        )
    return runtime


def engine_options(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    for option, cli in (("ram_gib", "--ram-gib"), ("context_gib", "--context-gib"),
                        ("context_tokens", "--context-tokens"),
                        ("device_slots", "--device-slots")):
        value = getattr(args, option, None)
        if value is not None:
            result.extend((cli, str(value)))
    if getattr(args, "long_mode", None):
        result.extend(("--long-mode", args.long_mode))
    if getattr(args, "think", False):
        result.append("--think")
    if getattr(args, "no_prewarm", False):
        result.append("--no-prewarm")
    return result


def standalone_settings(model: dict[str, Any], profile: dict[str, Any],
                        args: argparse.Namespace) -> dict[str, str]:
    spec = model["standalone"]
    if getattr(args, "context_gib", None) is not None or \
            getattr(args, "long_mode", "auto") not in (None, "auto"):
        raise UserError(
            f"{model['name']} currently exposes its validated short-context path only; "
            "context-gib and long-mode overrides are not available."
        )
    context_tokens = getattr(args, "context_tokens", None)
    context_env = spec.get("context_tokens_env")
    if context_tokens is not None:
        if not context_env:
            raise UserError(
                f"{model['name']} does not expose a context-token override yet."
            )
        if context_tokens <= 0:
            raise UserError("--context-tokens must be positive")
    result = {
        spec["ram_env"]: str(getattr(args, "ram_gib", None) or
                             profile.get("default_ram_gib", spec["default_ram_gib"])),
        spec["slots_env"]: str(getattr(args, "device_slots", None) or
                               profile.get("default_slots", spec["default_slots"])),
    }
    if not getattr(args, "no_prewarm", False):
        result[spec["prewarm_env"]] = "1"
    no_think = spec.get("no_think_env")
    if no_think and not getattr(args, "think", False):
        result[no_think] = "1"
    result.update(profile.get("env", {}))
    if context_tokens is not None:
        result[context_env] = str(context_tokens)
    return result


def standalone_environment(model: dict[str, Any], profile: dict[str, Any],
                           args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    spec = model["standalone"]
    settings = standalone_settings(model, profile, args)
    environment = os.environ.copy()
    prefixes = tuple(spec.get("clear_env_prefixes", ()))
    if prefixes:
        for key in list(environment):
            if key.startswith(prefixes):
                environment.pop(key, None)
    else:
        for key in (spec["ram_env"], spec["slots_env"], spec["prewarm_env"],
                    spec.get("no_think_env")):
            if key:
                environment.pop(key, None)
    environment.update(settings)
    return environment, settings


def print_profile_warning(profile: dict[str, Any]) -> None:
    if profile.get("warning"):
        print(f"WARNING: {profile['warning']}", file=sys.stderr)


def command_plan(args: argparse.Namespace) -> int:
    model = find_model(args.model)
    profile_name, profile = select_profile(model, args.profile)
    runtime = runtime_for(model, args.model_root, profile)
    if "standalone" in model:
        _, settings = standalone_environment(model, profile, args)
        print(f"Model: {model['name']}")
        if profile_name:
            print(f"Profile: {profile_name} - {profile['label']}")
        print(f"Backend: {find_engine(model, profile)}")
        print(f"Runtime: {runtime}")
        for key, value in settings.items():
            print(f"{key}={value}")
        if profile.get("measured_tok_s"):
            print(f"Measured decode: {profile['measured_tok_s']:.2f} tok/s on the validation system")
        print_profile_warning(profile)
        context_env = model["standalone"].get("context_tokens_env")
        if context_env:
            requested = settings.get(context_env)
            detail = f"explicit {requested}-token capacity" if requested else \
                "runtime-sized from prompt plus generation"
            print(f"Context: {detail}; bounded by available Vulkan memory")
        else:
            print("Context: validated short-context backend")
        return 0
    command = [str(find_engine(model)), *engine_options(args), "--plan", str(runtime)]
    print(f"> {format_command(command)}", flush=True)
    return subprocess.run(command).returncode


def command_run(args: argparse.Namespace) -> int:
    model = find_model(args.model)
    _, profile = select_profile(model, args.profile)
    runtime = runtime_for(model, args.model_root, profile)
    environment = None
    if "standalone" in model:
        environment, _ = standalone_environment(model, profile, args)
        print_profile_warning(profile)
        command = [str(find_engine(model, profile)), str(runtime), args.prompt,
                   str(args.tokens)]
    else:
        command = [str(find_engine(model)), *engine_options(args), "--tokens", str(args.tokens),
                   str(runtime), args.prompt]
    print(f"> {format_command(command)}", flush=True)
    return subprocess.run(command, env=environment).returncode


def command_chat(args: argparse.Namespace) -> int:
    model = find_model(args.model)
    profile_name, profile = select_profile(model, args.profile)
    runtime = runtime_for(model, args.model_root, profile)
    server = ROOT / "tools" / "xtllm_chat_server.py"
    display_name = model["name"] + (f" · {profile_name}" if profile_name else "")
    command = [sys.executable, str(server), "--engine", str(find_engine(model, profile)),
               "--runtime", str(runtime), "--model-name", display_name,
               "--port", str(args.port), "--tokens", str(args.tokens)]
    if "standalone" in model:
        _, settings = standalone_environment(model, profile, args)
        print_profile_warning(profile)
        command.append("--standalone")
        for key, value in settings.items():
            command.append(f"--engine-env={key}={value}")
    else:
        for value in engine_options(args):
            command.append(f"--engine-option={value}")
    if args.no_browser:
        command.append("--no-browser")
    print(f"> {format_command(command)}", flush=True)
    return subprocess.run(command).returncode


def add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-root", help="model storage root (or XTLLM_MODELS)")
    parser.add_argument("--profile",
                        help="inference profile (reference, full, or fast where available)")
    parser.add_argument("--ram-gib", type=float)
    parser.add_argument("--context-gib", type=float)
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--long-mode", choices=("auto", "exact", "fast"), default="auto")
    parser.add_argument("--device-slots", type=int)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--no-prewarm", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xtllm",
        description="Download, convert, and run XTLLM's native Vulkan model backends")
    subparsers = parser.add_subparsers(dest="command", required=True)
    models = subparsers.add_parser("models", help="list supported checkpoints")
    models.set_defaults(function=command_models)
    doctor = subparsers.add_parser("doctor", help="check this installation")
    doctor.add_argument("--model-root")
    doctor.set_defaults(function=command_doctor)
    setup = subparsers.add_parser("setup", help="download and convert a checkpoint")
    setup.add_argument("model")
    setup.add_argument("--model-root")
    setup.add_argument("--profile",
                       help="convert the selected profile (defaults to the model's full profile)")
    setup.add_argument("--hf-token", help="Hugging Face token (prefer HF_TOKEN)")
    setup.add_argument("--yes", action="store_true")
    setup.add_argument("--dry-run", action="store_true")
    setup.add_argument("--skip-download", action="store_true",
                       help="convert an already complete local checkpoint")
    setup.set_defaults(function=command_setup)
    plan = subparsers.add_parser("plan", help="show automatic RAM/VRAM policy")
    plan.add_argument("model")
    add_runtime_options(plan)
    plan.set_defaults(function=command_plan)
    run = subparsers.add_parser("run", help="generate text")
    run.add_argument("model")
    run.add_argument("prompt")
    run.add_argument("--tokens", type=int, default=128)
    add_runtime_options(run)
    run.set_defaults(function=command_run)
    chat = subparsers.add_parser("chat", help="launch the localhost chat UI")
    chat.add_argument("model")
    chat.add_argument("--tokens", type=int, default=128)
    chat.add_argument("--port", type=int, default=7860)
    chat.add_argument("--no-browser", action="store_true")
    add_runtime_options(chat)
    chat.set_defaults(function=command_chat)
    return parser


def main() -> int:
    try:
        arguments = build_parser().parse_args()
        return int(arguments.function(arguments))
    except UserError as error:
        print(f"XTLLM: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"XTLLM: command failed with exit code {error.returncode}", file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
