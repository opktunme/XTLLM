#!/usr/bin/env python3
"""Preserve Nemotron NVFP4 E2M1 weights with BF16 K16 block scales.

This is a companion to convert_nemotron3.py.  It keeps every official E2M1
nibble unchanged, folds the official E4M3 block scale and FP32 global scale
into one BF16 scale per K16 block, and only uses Q4G64T for source tensors
which are not NVFP4 in the official checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import os
from pathlib import Path
from typing import BinaryIO, List, Sequence, Tuple

import numpy as np

import convert_nemotron3 as b


NVFP4_BF16 = 103
MAGIC_EXPERT = b"ON3NNV4\0"

UP_SCALE = 0
UP_WEIGHT = 623_616
DOWN_SCALE = 3_118_080
DOWN_WEIGHT = 3_741_696
EXPERT_PAYLOAD_BYTES = 6_236_160
EXPERT_RECORD_BYTES = 6_238_208


def e4m3_lut() -> np.ndarray:
    result = np.empty(256, np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent, mantissa = (code >> 3) & 15, code & 7
        if exponent == 0:
            value = math.ldexp(mantissa / 8.0, -6)
        elif exponent == 15 and mantissa == 7:
            value = math.nan
        else:
            value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        result[code] = sign * value
    return result


E4M3 = e4m3_lut()


def is_native(store: b.SafeStore, plan: b.Plan) -> bool:
    if plan.format != b.Q4G64T or len(plan.sources) != 1:
        return False
    name = plan.sources[0]
    return name.endswith(".weight") and name[:-7] + ".weight_scale" in store.infos


def write_native(output: BinaryIO, store: b.SafeStore, plan: b.Plan) -> b.Result:
    name = plan.sources[0]
    rows, columns = plan.shape
    if columns % 64 or columns % 16:
        raise ValueError(f"NVFP4 K dimension is not aligned: {plan.name}")
    packed = np.asarray(store.array(name), np.uint8)
    codes = np.asarray(store.array(name[:-7] + ".weight_scale"), np.uint8)
    global_scale = float(np.asarray(
        store.array(name[:-7] + ".weight_scale_2")).reshape(-1)[0])
    if packed.shape != (rows, columns // 2) or codes.shape != (rows, columns // 16):
        raise ValueError(f"NVFP4 shape mismatch: {plan.name}")

    scale_offset = output.tell()
    scales = E4M3[codes] * np.float32(global_scale)
    if not np.all(np.isfinite(scales)):
        raise ValueError(f"non-finite NVFP4 scale: {plan.name}")
    output.write(b.f32_bf16(scales).tobytes())
    scale_bytes = rows * (columns // 16) * 2
    b.align_file(output, 64)

    weight_offset = output.tell()
    groups = columns // 64
    # Preserve every nibble, changing only the byte-word order to the proven
    # [word-in-K64][K64-group] row layout used by the RDNA2 kernels.
    words = np.ascontiguousarray(packed).reshape(rows, groups, 8, 4)
    words = words.view("<u4").reshape(rows, groups, 8).transpose(0, 2, 1)
    output.write(np.ascontiguousarray(words).tobytes())
    weight_bytes = rows * columns // 2
    native_plan = replace(plan, format=NVFP4_BF16)
    return b.Result(native_plan, weight_offset, weight_bytes,
                    scale_offset, scale_bytes)


def write_plan(output: BinaryIO, store: b.SafeStore, plan: b.Plan) -> b.Result:
    b.align_file(output, 64)
    return write_native(output, store, plan) if is_native(store, plan) \
        else b.write_plan(output, store, plan)


def pack_tensor(result: b.Result) -> bytes:
    if result.plan.format != NVFP4_BF16:
        return b.pack_tensor(result)
    plan = result.plan
    name = plan.name.encode("utf-8")
    shape = list(plan.shape) + [0] * (8 - len(plan.shape))
    return b.TENSOR_ENTRY.pack(
        name, NVFP4_BF16, len(plan.shape), *shape,
        result.data_offset, result.data_bytes,
        result.auxiliary_offset, result.auxiliary_bytes,
        plan.shape[-1] // 2, plan.shape[-1] // 16 * 2, 16, 0,
    )


def write_shared(output_dir: Path, store: b.SafeStore,
                 plans: Sequence[b.Plan]) -> Path:
    final = output_dir / "model-nvfp4.ovs"
    partial = output_dir / "model-nvfp4.ovs.partial"
    if final.exists():
        print(f"shared: keeping existing {final}")
        return final
    if partial.exists():
        raise FileExistsError(f"preserving existing partial: {partial}")
    groups: List[Tuple[int, int, List[b.Plan]]] = [
        (0, -1, [p for p in plans if p.group < 0])]
    groups += [(1, layer, [p for p in plans if p.group == layer])
               for layer in range(b.LAYERS)]
    table_offset = b.HEADER_BYTES
    data_offset = b.align_value(table_offset + len(plans) * b.TENSOR_ENTRY.size, 4096)
    results, group_rows = [], []
    with partial.open("xb", buffering=16 * 1024 * 1024) as output:
        output.write(bytes(data_offset))
        first = 0
        for kind, index, group_plans in groups:
            b.align_file(output, 4096)
            begin = output.tell()
            for plan in group_plans:
                results.append(write_plan(output, store, plan))
            b.align_file(output, 4096)
            end = output.tell()
            group_rows.append((kind, index, first, len(group_plans), begin, end, 0, 0))
            first += len(group_plans)
            print(f"shared {'global' if index < 0 else f'layer {index + 1}/{b.LAYERS}'}",
                  flush=True)
        file_bytes = output.tell()
        output.seek(table_offset)
        for result in results:
            output.write(pack_tensor(result))
        header = b.SHARED_HEADER.pack(
            b.MAGIC_SHARED, b.VERSION, b.HEADER_BYTES, 1, b.TENSOR_ENTRY.size,
            b.DIM, b.MOE_DIM, b.LAYERS, b.Q_HEADS, b.KV_HEADS, b.HEAD_DIM,
            b.MAMBA_HEADS, b.MAMBA_GROUPS, b.MAMBA_HEAD_DIM, b.ROPE_DIM,
            b.VOCAB, b.EXPERTS, b.TOP_K, 1, b.LINEAR_CONV_KERNEL, 0, 0, 0,
            b.MAMBA_HEADS, b.MAMBA_HEAD_DIM, b.MAMBA_STATE, b.MAX_POSITION,
            0, 0, 1, 11, 2, 10, 11, 12, 13, b.UINT32_MAX,
            1e-5, 1e-5, 2.5, 0.0, 10_000.0, 0.0,
            0.0, 0.0, 0.0, float(b.MAX_POSITION), 0.0, 0.0,
            b.SHARED_HEADER.size, len(group_rows), table_offset, len(results),
            data_offset, file_bytes, EXPERT_RECORD_BYTES,
            len(b.MOE_LAYERS) * b.EXPERTS, b.MAIN_PARAMETERS, b.ACTIVE_PARAMETERS)
        output.seek(0)
        output.write(header)
        for row in group_rows:
            output.write(b.GROUP_ENTRY.pack(*row))
        output.flush(); os.fsync(output.fileno())
    partial.rename(final)
    print(f"shared: {final} ({file_bytes:,} bytes)")
    return final


def expert_header(file_bytes: int) -> bytes:
    records = len(b.MOE_LAYERS) * b.EXPERTS
    return b.EXPERT_HEADER.pack(
        MAGIC_EXPERT, b.VERSION, b.HEADER_BYTES, b.DIM, b.MOE_DIM,
        len(b.MOE_LAYERS), b.EXPERTS, 0, EXPERT_RECORD_BYTES,
        b.HEADER_BYTES, file_bytes, records, records, file_bytes,
        UP_WEIGHT, UP_SCALE, 0, 0, DOWN_WEIGHT, DOWN_SCALE, NVFP4_BF16)


def write_experts(output_dir: Path, store: b.SafeStore) -> Path:
    final = output_dir / "experts-nvfp4.ovx"
    partial = output_dir / "experts-nvfp4.ovx.partial"
    records = len(b.MOE_LAYERS) * b.EXPERTS
    expected = b.HEADER_BYTES + records * EXPERT_RECORD_BYTES
    if final.exists():
        print(f"experts: keeping existing {final}")
        return final
    if partial.exists():
        raise FileExistsError(f"preserving existing partial: {partial}")
    with partial.open("xb", buffering=16 * 1024 * 1024) as output:
        output.write(expert_header(expected))
        output.write(bytes(b.HEADER_BYTES - b.EXPERT_HEADER.size))
        for local, layer in enumerate(b.MOE_LAYERS):
            for expert in range(b.EXPERTS):
                start = output.tell()
                prefix = f"backbone.layers.{layer}.mixer.experts.{expert}"
                specs = [
                    ("up_proj", b.MOE_DIM, b.DIM, UP_SCALE, UP_WEIGHT),
                    ("down_proj", b.DIM, b.MOE_DIM, DOWN_SCALE, DOWN_WEIGHT)]
                for projection, rows, columns, wanted_scale, wanted_weight in specs:
                    plan = b.Plan("expert", (f"{prefix}.{projection}.weight",),
                                  b.Q4G64T, layer, (rows, columns))
                    result = write_native(output, store, plan)
                    if result.auxiliary_offset - start != wanted_scale or \
                            result.data_offset - start != wanted_weight:
                        raise AssertionError("native expert offset drift")
                if output.tell() - start != EXPERT_PAYLOAD_BYTES:
                    raise AssertionError("native expert payload drift")
                output.write(bytes(EXPERT_RECORD_BYTES - EXPERT_PAYLOAD_BYTES))
            output.flush(); os.fsync(output.fileno())
            print(f"experts layer {local + 1}/{len(b.MOE_LAYERS)}", flush=True)
        if output.tell() != expected:
            raise AssertionError("native expert file size drift")
    partial.rename(final)
    print(f"experts: {final} ({expected:,} bytes)")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument("--skip-experts", action="store_true")
    args = parser.parse_args()
    store = b.SafeStore(args.source)
    plans = b.build_plans()
    b.validate_checkpoint(store, plans)
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.skip_shared: write_shared(args.output, store, plans)
    if not args.skip_experts: write_experts(args.output, store)


if __name__ == "__main__":
    main()
