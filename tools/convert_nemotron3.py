#!/usr/bin/env python3
"""Convert NVIDIA Nemotron-3-Nano-30B-A3B-NVFP4 for our Vulkan runtime.

The text-only main model is converted into three independent artifacts:

* ``model-q4g64.ovs`` contains global and per-layer non-routed tensors.
  Embedding, LM head, and authoritative routers use row-Q8.  Matrix
  projections use signed Q4G64T (K64 groups with BF16 scales).  Norms,
  Gated-DeltaNet constants, and its short depthwise convolution use F32.
* ``experts-q4g64.ovx`` contains one fixed, 4-KiB-aligned Q4G64T record per
  (layer, expert).  It resumes at exact record boundaries and truncates only
  an incomplete trailing record created by an interrupted conversion.
* ``tokenizer.ovb`` contains the exact padded Qwen vocabulary, BPE merges,
  added tokens, tokenizer pipeline metadata, and authoritative ChatML Jinja.

Vision and MTP weights are deliberately not emitted.  The official
Transformers text-only class ignores both namespaces, so their omission does
not alter text-only main-model logits.  This converter performs no inference.

Expected checkpoint:
  Qwen/Qwen3.6-35B-A3B
  revision 995ad96eacd98c81ed38be0c5b274b04031597b0
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import mmap
import os
from pathlib import Path
import struct
from typing import BinaryIO, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np


MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"
MODEL_REVISION = "ce1b118ae66ec705d02c241525192832eb045fd3"

MAGIC_SHARED = b"ON3NSHR\0"
MAGIC_EXPERT = b"ON3NEXP\0"
MAGIC_TOKENIZER = b"OVBPE2\0\0"
VERSION = 1
HEADER_BYTES = 4096

# These structures intentionally retain the established runtime ABIs.
SHARED_HEADER = struct.Struct("<8s4I32I12f10Q40x")
GROUP_ENTRY = struct.Struct("<iiIIQQQQ16x")
TENSOR_ENTRY = struct.Struct("<96sII8Q8Q24x")
EXPERT_HEADER = struct.Struct("<8s8I12Q")
TOKENIZER_HEADER = struct.Struct("<8s20I8Q104x")
TOKEN_ENTRY = struct.Struct("<QII")
MERGE_ENTRY = struct.Struct("<4I")
assert SHARED_HEADER.size == 320
assert GROUP_ENTRY.size == 64
assert TENSOR_ENTRY.size == 256
assert EXPERT_HEADER.size == 136
assert TOKENIZER_HEADER.size == 256
assert TOKEN_ENTRY.size == 16
assert MERGE_ENTRY.size == 16

DIM = 2688
MOE_DIM = 1856
SHARED_DIM = 3712
LAYERS = 52
EXPERTS = 128
TOP_K = 6
VOCAB = 131072
BASE_VOCAB = 131072
MERGES = 269443
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
MAMBA_LAYERS = tuple(i for i, kind in enumerate(PATTERN) if kind == "M")
MOE_LAYERS = tuple(i for i, kind in enumerate(PATTERN) if kind == "E")
FULL_ATTENTION_LAYERS = tuple(i for i, kind in enumerate(PATTERN) if kind == "*")

Q_HEADS = 32
KV_HEADS = 2
HEAD_DIM = 128
ROPE_DIM = 0
MAMBA_HEADS = 64
MAMBA_HEAD_DIM = 64
MAMBA_GROUPS = 8
MAMBA_STATE = 128
MAMBA_INTERMEDIATE = 4096
MAMBA_CONV_CHANNELS = 6144
MAMBA_IN = 10304
LINEAR_CONV_KERNEL = 4
MAX_POSITION = 262144

MAIN_PARAMETERS = 0  # filled from the exact safetensors headers at validation
ACTIVE_PARAMETERS = 0
CHECKPOINT_BYTES = 19_339_781_632

F32 = 1
Q8_ROW = 100
Q4G64T = 102
Q4_GROUP = 64
UINT32_MAX = (1 << 32) - 1

UP_SCALE = 0
UP_WEIGHT = UP_SCALE + MOE_DIM * (DIM // Q4_GROUP) * 2
DOWN_SCALE = UP_WEIGHT + MOE_DIM * DIM // 2
DOWN_WEIGHT = DOWN_SCALE + DIM * (MOE_DIM // Q4_GROUP) * 2
EXPERT_PAYLOAD_BYTES = DOWN_WEIGHT + DIM * MOE_DIM // 2
EXPERT_RECORD_BYTES = (EXPERT_PAYLOAD_BYTES + 4095) // 4096 * 4096
GATE_SCALE = GATE_WEIGHT = 0
assert (UP_SCALE, UP_WEIGHT) == (0, 155_904)
assert (DOWN_SCALE, DOWN_WEIGHT) == (2_650_368, 2_806_272)
assert EXPERT_PAYLOAD_BYTES == 5_300_736
assert EXPERT_RECORD_BYTES == 5_304_320
assert EXPERT_RECORD_BYTES % 4096 == 0


def align_value(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def align_file(output: BinaryIO, multiple: int) -> None:
    padding = (-output.tell()) % multiple
    if padding:
        output.write(bytes(padding))


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


@dataclass(frozen=True)
class TensorInfo:
    shard: str
    dtype: str
    shape: Tuple[int, ...]
    data_start: int
    begin: int
    end: int


class SafeStore:
    """Read-only, lazily mapped view of a complete sharded safetensors model."""

    def __init__(self, directory: Path):
        self.directory = directory
        index_path = directory / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing official checkpoint index: {index_path}")
        index = json.loads(index_path.read_text("utf-8"))
        if int(index.get("metadata", {}).get("total_size", -1)) != CHECKPOINT_BYTES:
            raise ValueError("checkpoint index total_size does not match official Nemotron")
        self.weight_map: Dict[str, str] = index["weight_map"]
        if len(self.weight_map) != 24147:
            raise ValueError(f"unexpected checkpoint tensor count: {len(self.weight_map)}")

        self.infos: Dict[str, TensorInfo] = {}
        self.files: Dict[str, BinaryIO] = {}
        self.maps: Dict[str, mmap.mmap] = {}
        by_shard: Dict[str, List[str]] = {}
        for name, shard in self.weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        if len(by_shard) != 5:
            raise ValueError(f"unexpected official shard count: {len(by_shard)}")
        missing = [name for name in sorted(by_shard) if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"checkpoint incomplete: {len(missing)}/5 shards missing: {missing[:6]}"
            )

        dtype_bytes = {"F32": 4, "BF16": 2, "U8": 1, "F8_E4M3": 1}
        for shard, names in sorted(by_shard.items()):
            path = directory / shard
            with path.open("rb") as source:
                encoded_bytes = source.read(8)
                if len(encoded_bytes) != 8:
                    raise ValueError(f"truncated safetensors shard: {shard}")
                header_bytes = struct.unpack("<Q", encoded_bytes)[0]
                if header_bytes > 16 * 1024 * 1024:
                    raise ValueError(f"implausible safetensors header: {shard}")
                encoded = source.read(header_bytes)
                if len(encoded) != header_bytes:
                    raise ValueError(f"truncated safetensors header: {shard}")
                header = json.loads(encoded)
            data_start = 8 + header_bytes
            maximum = 0
            for name in names:
                if name not in header:
                    raise ValueError(f"index/header disagreement for {name}")
                entry = header[name]
                begin, end = map(int, entry["data_offsets"])
                shape = tuple(map(int, entry["shape"]))
                dtype = entry["dtype"]
                if dtype not in dtype_bytes:
                    raise ValueError(f"unexpected native dtype {dtype}: {name}")
                if end - begin != product(shape) * dtype_bytes[dtype]:
                    raise ValueError(f"bad tensor extent: {name}")
                self.infos[name] = TensorInfo(shard, dtype, shape, data_start, begin, end)
                maximum = max(maximum, end)
            if data_start + maximum != path.stat().st_size:
                raise ValueError(f"partial or unexpected safetensors shard: {shard}")

        if len(self.infos) != len(self.weight_map):
            raise AssertionError("not every indexed tensor was discovered")
        dtype_parameters: Dict[str, int] = {key: 0 for key in dtype_bytes}
        for info in self.infos.values():
            dtype_parameters[info.dtype] += product(info.shape)
        if not dtype_parameters["U8"] or not dtype_parameters["BF16"]:
            raise ValueError(f"checkpoint dtype mix is unexpected: {dtype_parameters}")

    def info(self, name: str) -> TensorInfo:
        try:
            return self.infos[name]
        except KeyError as error:
            raise KeyError(f"missing checkpoint tensor: {name}") from error

    def mapping(self, shard: str) -> mmap.mmap:
        if shard not in self.maps:
            source = (self.directory / shard).open("rb")
            self.files[shard] = source
            self.maps[shard] = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        return self.maps[shard]

    def array(self, name: str) -> np.ndarray:
        info = self.info(name)
        dtype = {"F32": "<f4", "BF16": "<u2", "U8": "u1", "F8_E4M3": "u1"}[info.dtype]
        return np.ndarray(
            info.shape,
            dtype=dtype,
            buffer=self.mapping(info.shard),
            offset=info.data_start + info.begin,
        )


def bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype="<u2").astype(np.uint32) << 16).view("<f4")


def f32_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype="<u2")


def decode_array(store: SafeStore, name: str) -> np.ndarray:
    info = store.info(name)
    values = store.array(name)
    if info.dtype == "BF16":
        return bf16_f32(values)
    if info.dtype == "F32":
        return np.asarray(values, dtype="<f4")
    raise AssertionError(info.dtype)


ChunkFactory = Callable[[], Iterator[Tuple[int, np.ndarray]]]


def pack_q4g64(
    output: BinaryIO,
    chunks: ChunkFactory,
    rows: int,
    columns: int,
    *,
    cache_decoded: bool = False,
) -> Tuple[int, int, int, int]:
    """Write one signed-Q4/K64 transposed matrix and its BF16 scales."""
    if rows <= 0 or columns <= 0 or columns % Q4_GROUP:
        raise ValueError(f"invalid Q4 matrix shape: {(rows, columns)}")
    groups = columns // Q4_GROUP
    cached = list(chunks()) if cache_decoded else None

    maxima = np.empty((rows, groups), dtype=np.float32)
    seen = 0
    for first, source_values in cached if cached is not None else chunks():
        values = np.asarray(source_values, dtype=np.float32).reshape(-1, groups, Q4_GROUP)
        if first != seen or first + len(values) > rows:
            raise ValueError("Q4 chunk sequence is not contiguous")
        maxima[first:first + len(values)] = np.max(np.abs(values), axis=2)
        seen += len(values)
    if seen != rows:
        raise ValueError(f"Q4 chunk rows incomplete: {seen}/{rows}")
    scales = maxima / np.float32(7.0)

    scale_offset = output.tell()
    output.write(f32_bf16(scales).tobytes())
    scale_bytes = rows * groups * 2
    align_file(output, 64)
    weight_offset = output.tell()

    seen = 0
    for first, source_values in cached if cached is not None else chunks():
        values = np.asarray(source_values, dtype=np.float32).reshape(-1, groups, Q4_GROUP)
        count = len(values)
        if first != seen:
            raise ValueError("Q4 second-pass chunk sequence changed")
        denominator = np.maximum(scales[first:first + count, :, None], np.float32(1e-30))
        quantized = np.rint(values / denominator).astype(np.int8)
        np.clip(quantized, -7, 7, out=quantized)
        nibbles = quantized.view(np.uint8).reshape(count, groups, 8, 8)
        packed = np.zeros((count, groups, 8), dtype="<u4")
        for nibble in range(8):
            packed |= (
                (nibbles[:, :, :, nibble] & 15).astype(np.uint32)
                << np.uint32(nibble * 4)
            )
        output.write(np.ascontiguousarray(packed.transpose(0, 2, 1)).tobytes())
        seen += count
    weight_bytes = rows * columns // 2
    if seen != rows or output.tell() - weight_offset != weight_bytes:
        raise AssertionError("Q4 packed size drift")
    return weight_offset, weight_bytes, scale_offset, scale_bytes


@dataclass(frozen=True)
class Plan:
    name: str
    sources: Tuple[str, ...]
    format: int
    group: int
    shape: Tuple[int, ...]
    transform: str = "direct"


@dataclass(frozen=True)
class Result:
    plan: Plan
    data_offset: int
    data_bytes: int
    auxiliary_offset: int = 0
    auxiliary_bytes: int = 0


def build_plans() -> List[Plan]:
    result = [
        Plan("embed", ("backbone.embeddings.weight",), Q8_ROW, -1, (VOCAB, DIM)),
        Plan("final_norm", ("backbone.norm_f.weight",), F32, -1, (DIM,)),
        Plan("lm_head", ("lm_head.weight",), Q8_ROW, -1, (VOCAB, DIM)),
    ]
    for layer in range(LAYERS):
        source = f"backbone.layers.{layer}"
        runtime = f"layers.{layer}"
        result.append(Plan(f"{runtime}.norm", (f"{source}.norm.weight",), F32,
                           layer, (DIM,)))
        mixer = f"{source}.mixer"
        if layer in MAMBA_LAYERS:
            result.extend([
                Plan(f"{runtime}.mamba_in", (f"{mixer}.in_proj.weight",), Q4G64T,
                     layer, (MAMBA_IN, DIM)),
                Plan(f"{runtime}.mamba_out", (f"{mixer}.out_proj.weight",), Q4G64T,
                     layer, (DIM, MAMBA_INTERMEDIATE)),
                Plan(f"{runtime}.mamba_conv_weight", (f"{mixer}.conv1d.weight",), F32,
                     layer, (MAMBA_CONV_CHANNELS, LINEAR_CONV_KERNEL), "conv"),
                Plan(f"{runtime}.mamba_conv_bias", (f"{mixer}.conv1d.bias",), F32,
                     layer, (MAMBA_CONV_CHANNELS,)),
                Plan(f"{runtime}.mamba_params",
                     (f"{mixer}.A_log", f"{mixer}.D", f"{mixer}.dt_bias",
                      f"{mixer}.norm.weight"), F32, layer,
                     (MAMBA_HEADS * 3 + MAMBA_INTERMEDIATE,), "concat_vector"),
            ])
        elif layer in FULL_ATTENTION_LAYERS:
            result.extend([
                Plan(f"{runtime}.q_proj", (f"{mixer}.q_proj.weight",), Q4G64T,
                     layer, (Q_HEADS * HEAD_DIM, DIM)),
                Plan(f"{runtime}.k_proj", (f"{mixer}.k_proj.weight",), Q4G64T,
                     layer, (KV_HEADS * HEAD_DIM, DIM)),
                Plan(f"{runtime}.v_proj", (f"{mixer}.v_proj.weight",), Q4G64T,
                     layer, (KV_HEADS * HEAD_DIM, DIM)),
                Plan(f"{runtime}.o_proj", (f"{mixer}.o_proj.weight",), Q4G64T,
                     layer, (DIM, Q_HEADS * HEAD_DIM)),
            ])
        else:
            result.extend([
                Plan(f"{runtime}.router", (f"{mixer}.gate.weight",), Q8_ROW,
                     layer, (EXPERTS, DIM)),
                Plan(f"{runtime}.correction", (f"{mixer}.gate.e_score_correction_bias",),
                     F32, layer, (EXPERTS,)),
                Plan(f"{runtime}.shared_up", (f"{mixer}.shared_experts.up_proj.weight",),
                     Q4G64T, layer, (SHARED_DIM, DIM)),
                Plan(f"{runtime}.shared_down", (f"{mixer}.shared_experts.down_proj.weight",),
                     Q4G64T, layer, (DIM, SHARED_DIM)),
            ])
    if len(result) != 286 or len({plan.name for plan in result}) != len(result):
        raise AssertionError("Nemotron shared plan count/name drift")
    return result


def validate_plan(store: SafeStore, plan: Plan) -> None:
    infos = [store.info(name) for name in plan.sources]
    if plan.transform == "direct":
        if len(infos) != 1 or (plan.format != Q4G64T and infos[0].shape != plan.shape):
            raise ValueError(f"shape mismatch for {plan.name}: {[x.shape for x in infos]}")
    elif plan.transform == "concat_rows":
        if len(plan.shape) != 2 or any(len(info.shape) != 2 for info in infos):
            raise ValueError(f"invalid row concatenation for {plan.name}")
        if any(info.shape[1] != plan.shape[1] for info in infos) or \
                sum(info.shape[0] for info in infos) != plan.shape[0]:
            raise ValueError(f"shape mismatch for {plan.name}: {[x.shape for x in infos]}")
    elif plan.transform == "concat_vector":
        if plan.shape != (sum(product(info.shape) for info in infos),):
            raise ValueError(f"vector concatenation mismatch for {plan.name}")
    elif plan.transform == "conv":
        if len(infos) != 1 or infos[0].shape != \
                (MAMBA_CONV_CHANNELS, 1, LINEAR_CONV_KERNEL) or \
                plan.shape != (MAMBA_CONV_CHANNELS, LINEAR_CONV_KERNEL):
            raise ValueError(f"convolution shape mismatch for {plan.name}")
    else:
        raise AssertionError(plan.transform)
    if plan.format == Q8_ROW and any(info.dtype not in ("BF16", "F32") for info in infos):
        raise ValueError(f"unsupported Q8 source for {plan.name}")
    if plan.format == Q4G64T:
        if len(plan.sources) != 1:
            raise ValueError(f"Nemotron Q4 source must be direct: {plan.name}")
        name = plan.sources[0]
        info = infos[0]
        rows, columns = plan.shape
        scale_name = name[:-7] + ".weight_scale" if name.endswith(".weight") else ""
        if scale_name in store.infos:
            scale = store.info(scale_name)
            scale2 = store.info(name[:-7] + ".weight_scale_2")
            if info.dtype != "U8" or info.shape != (rows, columns // 2) or \
                    scale.shape != (rows, columns // 16) or product(scale2.shape) != 1:
                raise ValueError(f"bad NVFP4 source for {plan.name}")
        elif info.dtype != "BF16" or info.shape != plan.shape:
            raise ValueError(f"bad dense source for {plan.name}: {info.dtype} {info.shape}")


def validate_checkpoint(store: SafeStore, plans: Sequence[Plan]) -> None:
    global MAIN_PARAMETERS, ACTIVE_PARAMETERS
    for plan in plans:
        validate_plan(store, plan)
    routed_parameters = 0
    for layer in MOE_LAYERS:
        for expert in range(EXPERTS):
            prefix = f"backbone.layers.{layer}.mixer.experts.{expert}"
            for projection, shape in (("up_proj", (MOE_DIM, DIM)),
                                      ("down_proj", (DIM, MOE_DIM))):
                name = f"{prefix}.{projection}.weight"
                info = store.info(name)
                scale = store.info(f"{prefix}.{projection}.weight_scale")
                if info.dtype != "U8" or info.shape != (shape[0], shape[1] // 2) or \
                        scale.shape != (shape[0], shape[1] // 16):
                    raise ValueError(f"unexpected routed expert tensor {name}")
                routed_parameters += shape[0] * shape[1]
    ignored = (".weight_scale", ".weight_scale_2", ".input_scale")
    main_parameters = 0
    for name, info in store.infos.items():
        if name.endswith(ignored):
            continue
        if not (name == "lm_head.weight" or name.startswith("backbone.")):
            continue
        if info.dtype == "U8" and name.endswith(".weight") and \
                name[:-7] + ".weight_scale" in store.infos:
            main_parameters += product(info.shape) * 2
        else:
            main_parameters += product(info.shape)
    MAIN_PARAMETERS = main_parameters
    ACTIVE_PARAMETERS = main_parameters - routed_parameters + \
        routed_parameters * TOP_K // EXPERTS
    print(
        f"validated {MODEL_ID}@{MODEL_REVISION}: "
        f"main={MAIN_PARAMETERS:,}, active={ACTIVE_PARAMETERS:,}, tensors={len(store.infos):,}",
        flush=True,
    )


def matrix_chunks(store: SafeStore, plan: Plan, chunk_rows: int = 128) -> ChunkFactory:
    if plan.transform not in ("direct", "concat_rows"):
        raise ValueError(f"cannot make Q4 chunks for {plan.name}")

    name = plan.sources[0]
    scale_name = name[:-7] + ".weight_scale" if name.endswith(".weight") else ""
    is_nvfp4 = scale_name in store.infos
    def iterate() -> Iterator[Tuple[int, np.ndarray]]:
        values = store.array(name)
        if not is_nvfp4:
            for first in range(0, len(values), chunk_rows):
                yield first, bf16_f32(values[first:first + chunk_rows])
            return
        scales = store.array(scale_name)
        scale2 = float(np.asarray(store.array(name[:-7] + ".weight_scale_2")).reshape(-1)[0])
        e2m1 = np.array([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6], np.float32)
        e4m3 = np.empty(256, np.float32)
        for code in range(256):
            sign = -1.0 if code & 0x80 else 1.0
            exponent, mantissa = (code >> 3) & 15, code & 7
            value = math.ldexp(mantissa / 8.0, -6) if exponent == 0 else \
                    math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
            e4m3[code] = sign * value
        for first in range(0, len(values), chunk_rows):
            packed = np.asarray(values[first:first + chunk_rows], np.uint8)
            decoded = np.empty((len(packed), packed.shape[1] * 2), np.float32)
            decoded[:,0::2] = e2m1[packed & 15]
            decoded[:,1::2] = e2m1[packed >> 4]
            decoded *= np.repeat(e4m3[np.asarray(scales[first:first+chunk_rows],np.uint8)] *
                                 np.float32(scale2), 16, axis=1)
            yield first, decoded

    return iterate


def expert_chunks(store: SafeStore, name: str, rows: int, columns: int) -> ChunkFactory:
    plan = Plan("expert", (name,), Q4G64T, 0, (rows, columns))
    return matrix_chunks(store, plan)


def write_q8(output: BinaryIO, store: SafeStore, plan: Plan) -> Result:
    if plan.transform != "direct" or len(plan.shape) != 2:
        raise ValueError(f"Q8 expects a direct matrix: {plan.name}")
    source_name = plan.sources[0]
    values = store.array(source_name)
    rows, columns = plan.shape
    scales = np.empty(rows, dtype="<f4")
    data_offset = output.tell()
    for first in range(0, rows, 32):
        decoded = (bf16_f32(values[first:first + 32])
                   if store.info(source_name).dtype == "BF16"
                   else np.asarray(values[first:first + 32], dtype="<f4"))
        maximum = np.max(np.abs(decoded), axis=1)
        scale = np.maximum(maximum / np.float32(127.0), np.float32(1e-30))
        scales[first:first + len(decoded)] = scale
        quantized = np.rint(decoded / scale[:, None]).astype(np.int16)
        output.write(np.clip(quantized, -127, 127).astype(np.int8).tobytes())
    data_bytes = rows * columns
    if output.tell() - data_offset != data_bytes:
        raise AssertionError("Q8 matrix size drift")
    align_file(output, 64)
    auxiliary_offset = output.tell()
    output.write(scales.tobytes())
    return Result(plan, data_offset, data_bytes, auxiliary_offset, scales.nbytes)


def write_f32(output: BinaryIO, store: SafeStore, plan: Plan) -> Result:
    data_offset = output.tell()
    if plan.transform == "direct":
        values = decode_array(store, plan.sources[0]).reshape(plan.shape)
        output.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
    elif plan.transform == "conv":
        values = decode_array(store, plan.sources[0]).reshape(plan.shape)
        output.write(np.ascontiguousarray(values, dtype="<f4").tobytes())
    elif plan.transform == "concat_vector":
        for name in plan.sources:
            output.write(np.ascontiguousarray(decode_array(store, name).reshape(-1), dtype="<f4").tobytes())
    else:
        raise ValueError(f"unsupported F32 transform: {plan.name} {plan.transform}")
    data_bytes = product(plan.shape) * 4
    if output.tell() - data_offset != data_bytes:
        raise AssertionError(f"F32 tensor size drift: {plan.name}")
    return Result(plan, data_offset, data_bytes)


def write_plan(output: BinaryIO, store: SafeStore, plan: Plan) -> Result:
    validate_plan(store, plan)
    align_file(output, 64)
    if plan.format == F32:
        return write_f32(output, store, plan)
    if plan.format == Q8_ROW:
        return write_q8(output, store, plan)
    if plan.format == Q4G64T:
        rows, columns = plan.shape
        weight, weight_bytes, scale, scale_bytes = pack_q4g64(
            output, matrix_chunks(store, plan), rows, columns
        )
        return Result(plan, weight, weight_bytes, scale, scale_bytes)
    raise AssertionError(plan.format)


def pack_tensor(result: Result) -> bytes:
    plan = result.plan
    name = plan.name.encode("utf-8")
    if len(name) >= 96:
        raise ValueError(f"runtime tensor name too long: {plan.name}")
    shape = list(plan.shape) + [0] * (8 - len(plan.shape))
    row_stride = 0
    auxiliary_stride = 0
    flags = 0
    if plan.format == Q8_ROW:
        row_stride = plan.shape[-1]
        auxiliary_stride = 4
    elif plan.format == Q4G64T:
        row_stride = plan.shape[-1] // 2
        auxiliary_stride = plan.shape[-1] // Q4_GROUP * 2
        flags = Q4_GROUP
    return TENSOR_ENTRY.pack(
        name,
        plan.format,
        len(plan.shape),
        *shape,
        result.data_offset,
        result.data_bytes,
        result.auxiliary_offset,
        result.auxiliary_bytes,
        row_stride,
        auxiliary_stride,
        flags,
        0,
    )


def validate_existing_shared(path: Path) -> None:
    if path.stat().st_size < HEADER_BYTES:
        raise ValueError(f"truncated existing Qwen shared file: {path}")
    with path.open("rb") as source:
        fields = SHARED_HEADER.unpack(source.read(SHARED_HEADER.size))
    if fields[0] != MAGIC_SHARED or fields[1] != VERSION or fields[2] != HEADER_BYTES:
        raise ValueError(f"incompatible existing Qwen shared file: {path}")
    # Ten QWORDs are the final ten unpacked values. file_bytes is QWORD 5.
    qwords = fields[-10:]
    if qwords[3] != 286 or qwords[5] != path.stat().st_size or \
            qwords[6] != EXPERT_RECORD_BYTES or qwords[7] != len(MOE_LAYERS) * EXPERTS:
        raise ValueError(f"invalid existing Qwen shared file: {path}")


def write_shared(output_dir: Path, store: SafeStore, plans: Sequence[Plan]) -> Path:
    final = output_dir / "model-q4g64.ovs"
    partial = output_dir / "model-q4g64.ovs.partial"
    if final.exists():
        validate_existing_shared(final)
        print(f"shared: keeping existing {final}")
        return final
    if partial.exists():
        attempt = 2
        while (output_dir / f"model-q4g64.ovs.partial{attempt}").exists():
            attempt += 1
        partial = output_dir / f"model-q4g64.ovs.partial{attempt}"
        print(f"shared: preserving earlier partial; retrying as {partial}", flush=True)

    groups: List[Tuple[int, int, List[Plan]]] = [
        (0, -1, [plan for plan in plans if plan.group < 0])
    ]
    groups.extend((1, layer, [plan for plan in plans if plan.group == layer]) for layer in range(LAYERS))
    if len(groups) != LAYERS + 1 or any(not group_plans for _, _, group_plans in groups):
        raise AssertionError("shared group construction drift")
    table_offset = HEADER_BYTES
    data_offset = align_value(table_offset + len(plans) * TENSOR_ENTRY.size, 4096)
    results: List[Result] = []
    group_rows = []
    with partial.open("xb", buffering=16 * 1024 * 1024) as output:
        output.write(bytes(data_offset))
        first_tensor = 0
        for kind, index, group_plans in groups:
            align_file(output, 4096)
            begin = output.tell()
            for plan in group_plans:
                results.append(write_plan(output, store, plan))
            align_file(output, 4096)
            end = output.tell()
            group_rows.append((kind, index, first_tensor, len(group_plans), begin, end, 0, 0))
            first_tensor += len(group_plans)
            label = "global" if index < 0 else f"layer {index + 1}/{LAYERS}"
            print(f"shared {label}: {end - begin:,} bytes", flush=True)
        if len(results) != len(plans):
            raise AssertionError("not every shared plan was written")
        file_bytes = output.tell()
        output.seek(table_offset)
        for result in results:
            output.write(pack_tensor(result))
        header = SHARED_HEADER.pack(
            MAGIC_SHARED, VERSION, HEADER_BYTES, 1, TENSOR_ENTRY.size,
            DIM, MOE_DIM, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM,
            MAMBA_HEADS, MAMBA_GROUPS, MAMBA_HEAD_DIM, ROPE_DIM,
            VOCAB, EXPERTS, TOP_K, 1,
            LINEAR_CONV_KERNEL, 0, 0, 0,
            MAMBA_HEADS, MAMBA_HEAD_DIM, MAMBA_STATE, MAX_POSITION,
            0, 0,
            1, 11, 2, 10, 11, 12, 13, UINT32_MAX,
            1e-5, 1e-5, 2.5, 0.0, 10_000.0, 0.0,
            0.0, 0.0, 0.0, float(MAX_POSITION), 0.0, 0.0,
            SHARED_HEADER.size, len(group_rows), table_offset, len(results), data_offset,
            file_bytes, EXPERT_RECORD_BYTES, len(MOE_LAYERS) * EXPERTS,
            MAIN_PARAMETERS, ACTIVE_PARAMETERS,
        )
        output.seek(0)
        output.write(header)
        for row in group_rows:
            output.write(GROUP_ENTRY.pack(*row))
        if output.tell() > table_offset:
            raise AssertionError("group table overlaps tensor table")
        output.flush()
        os.fsync(output.fileno())
    partial.rename(final)
    validate_existing_shared(final)
    print(f"shared: {final} ({file_bytes:,} bytes)")
    return final


def expert_header(file_bytes: int) -> bytes:
    records = len(MOE_LAYERS) * EXPERTS
    return EXPERT_HEADER.pack(
        MAGIC_EXPERT, VERSION, HEADER_BYTES, DIM, MOE_DIM, len(MOE_LAYERS), EXPERTS, 0,
        EXPERT_RECORD_BYTES,
        HEADER_BYTES, file_bytes, records, records, file_bytes,
        UP_WEIGHT, UP_SCALE, 0, 0, DOWN_WEIGHT, DOWN_SCALE, 0,
    )


def validate_expert_header(path: Path, expected_bytes: int) -> None:
    if path.stat().st_size < HEADER_BYTES:
        raise ValueError(f"truncated Qwen expert file: {path}")
    with path.open("rb") as source:
        fields = EXPERT_HEADER.unpack(source.read(EXPERT_HEADER.size))
    expected_prefix = (
        MAGIC_EXPERT, VERSION, HEADER_BYTES, DIM, MOE_DIM, len(MOE_LAYERS), EXPERTS, 0,
        EXPERT_RECORD_BYTES,
    )
    if fields[:9] != expected_prefix:
        raise ValueError(f"incompatible Qwen expert header: {path}")
    if fields[9:] != (
        HEADER_BYTES, expected_bytes, len(MOE_LAYERS) * EXPERTS, len(MOE_LAYERS) * EXPERTS,
        expected_bytes, UP_WEIGHT, UP_SCALE, 0, 0,
        DOWN_WEIGHT, DOWN_SCALE, 0,
    ):
        raise ValueError(f"invalid Qwen expert header fields: {path}")


def write_experts(
    output_dir: Path,
    store: SafeStore,
    flush_records: int = 32,
) -> Path:
    final = output_dir / "experts-q4g64.ovx"
    partial = output_dir / "experts-q4g64.ovx.partial"
    records = len(MOE_LAYERS) * EXPERTS
    expected = HEADER_BYTES + records * EXPERT_RECORD_BYTES
    if final.exists():
        validate_expert_header(final, expected)
        if final.stat().st_size != expected:
            raise ValueError(f"bad existing Qwen expert file size: {final}")
        print(f"experts: keeping existing {final}")
        return final

    completed = 0
    if partial.exists():
        validate_expert_header(partial, expected)
        if partial.stat().st_size > expected:
            raise ValueError(f"Qwen expert partial exceeds final size: {partial}")
        data_bytes = partial.stat().st_size - HEADER_BYTES
        completed, remainder = divmod(data_bytes, EXPERT_RECORD_BYTES)
        if remainder:
            boundary = HEADER_BYTES + completed * EXPERT_RECORD_BYTES
            with partial.open("r+b") as output:
                output.truncate(boundary)
                output.flush()
                os.fsync(output.fileno())
            print(
                f"experts: discarded interrupted trailing record ({remainder:,} bytes); "
                f"resuming at {completed}/{records}",
                flush=True,
            )
        else:
            print(f"experts: resuming at record {completed}/{records}", flush=True)

    mode = "r+b" if partial.exists() else "xb"
    with partial.open(mode, buffering=16 * 1024 * 1024) as output:
        if completed == 0:
            output.write(expert_header(expected))
            output.write(bytes(HEADER_BYTES - EXPERT_HEADER.size))
        output.seek(HEADER_BYTES + completed * EXPERT_RECORD_BYTES)
        for record in range(completed, records):
            layer_local, expert = divmod(record, EXPERTS)
            layer = MOE_LAYERS[layer_local]
            prefix = f"backbone.layers.{layer}.mixer.experts.{expert}"
            up = f"{prefix}.up_proj.weight"
            down = f"{prefix}.down_proj.weight"
            start = output.tell()
            specifications = [
                (up, MOE_DIM, DIM, UP_WEIGHT, UP_SCALE),
                (down, DIM, MOE_DIM, DOWN_WEIGHT, DOWN_SCALE),
            ]
            for name, rows, columns, wanted_weight, wanted_scale in specifications:
                weight, weight_bytes, scale, scale_bytes = pack_q4g64(
                    output,
                    expert_chunks(store, name, rows, columns),
                    rows,
                    columns,
                    cache_decoded=True,
                )
                if weight - start != wanted_weight or scale - start != wanted_scale:
                    raise AssertionError(f"expert record offset drift: layer={layer} expert={expert}")
                if weight_bytes != rows * columns // 2 or \
                        scale_bytes != rows * (columns // Q4_GROUP) * 2:
                    raise AssertionError("expert projection size drift")
            if output.tell() - start != EXPERT_PAYLOAD_BYTES:
                raise AssertionError("expert payload size drift")
            output.write(bytes(EXPERT_RECORD_BYTES - EXPERT_PAYLOAD_BYTES))
            done = record + 1
            if done % max(1, flush_records) == 0 or expert == EXPERTS - 1:
                output.flush()
                os.fsync(output.fileno())
            if expert == EXPERTS - 1:
                print(f"experts layer {layer_local + 1}/{len(MOE_LAYERS)}", flush=True)
        if output.tell() != expected:
            raise AssertionError("expert output size drift")
    partial.rename(final)
    validate_expert_header(final, expected)
    if final.stat().st_size != expected:
        raise AssertionError("final Qwen expert file size drift")
    print(f"experts: {final} ({expected:,} bytes)")
    return final


def byte_decoder() -> Dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    characters = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): value for value, character in zip(byte_values, characters)}


def tokenizer_partial_path(output_dir: Path) -> Path:
    candidate = output_dir / "tokenizer.ovb.partial"
    if not candidate.exists():
        return candidate
    attempt = 2
    while (output_dir / f"tokenizer.ovb.partial{attempt}").exists():
        attempt += 1
    return output_dir / f"tokenizer.ovb.partial{attempt}"


def validate_existing_tokenizer(path: Path) -> None:
    if path.stat().st_size < TOKENIZER_HEADER.size:
        raise ValueError(f"truncated Qwen tokenizer container: {path}")
    with path.open("rb") as source:
        fields = TOKENIZER_HEADER.unpack(source.read(TOKENIZER_HEADER.size))
    if fields[0] != MAGIC_TOKENIZER or fields[1] != 2 or \
            fields[2] != TOKENIZER_HEADER.size or fields[3] != VOCAB or \
            fields[4] != BASE_VOCAB or fields[5] != MERGES or fields[-2] != path.stat().st_size:
        raise ValueError(f"invalid existing Qwen tokenizer container: {path}")


def write_tokenizer(metadata_dir: Path, output_dir: Path) -> Path:
    final = output_dir / "tokenizer.ovb"
    if final.exists():
        validate_existing_tokenizer(final)
        print(f"tokenizer: keeping existing {final}")
        return final
    tokenizer_path = metadata_dir / "tokenizer.json"
    tokenizer_config_path = metadata_dir / "tokenizer_config.json"
    chat_template_path = metadata_dir / "chat_template.jinja"
    for path in (tokenizer_path, tokenizer_config_path, chat_template_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing official tokenizer metadata: {path}")
    tokenizer = json.loads(tokenizer_path.read_text("utf-8"))
    tokenizer_config = json.loads(tokenizer_config_path.read_text("utf-8"))
    chat_template = chat_template_path.read_text("utf-8")
    if tokenizer_config.get("chat_template") != chat_template:
        raise ValueError("chat_template.jinja differs from tokenizer_config.json")
    model = tokenizer["model"]
    vocabulary = model["vocab"]
    if model.get("type") != "BPE" or len(vocabulary) != BASE_VOCAB or \
            len(model.get("merges", [])) != MERGES:
        raise ValueError("unexpected official Qwen tokenizer model")

    pieces: List[bytes | None] = [None] * VOCAB
    flags = [0] * VOCAB
    decoder = byte_decoder()
    for piece, token_value in vocabulary.items():
        token = int(token_value)
        if token < 0 or token >= BASE_VOCAB:
            raise ValueError(f"base token id out of range: {token}")
        try:
            pieces[token] = bytes(decoder[character] for character in piece)
        except KeyError as error:
            raise ValueError(f"base BPE piece is not byte-decodable: token {token}") from error

    added_by_id: Dict[int, dict] = {}
    for entry in tokenizer.get("added_tokens", []):
        added_by_id[int(entry["id"])] = entry
    for encoded_id, entry in tokenizer_config.get("added_tokens_decoder", {}).items():
        token = int(encoded_id)
        prior = added_by_id.get(token)
        if prior is not None and prior.get("content") != entry.get("content"):
            raise ValueError(f"added-token metadata disagreement at id {token}")
        combined = dict(prior or {})
        combined.update(entry)
        combined["id"] = token
        added_by_id[token] = combined
    if len(added_by_id) != 1000 or min(added_by_id) != 0 or max(added_by_id) != 999:
        raise ValueError(f"unexpected Nemotron added-token range/count")
    if set(added_by_id) != set(range(1000)):
        raise ValueError("Nemotron added-token IDs are not contiguous")
    for token, entry in sorted(added_by_id.items()):
        pieces[token] = str(entry["content"]).encode("utf-8")
        flags[token] = 1 | (2 if entry.get("special") else 0)
    for token in range(VOCAB):
        if pieces[token] is None:
            pieces[token] = f"<|unused_{token}|>".encode("ascii")
            flags[token] |= 4

    merges = []
    for rank, encoded in enumerate(model["merges"]):
        left, right = encoded if isinstance(encoded, list) else encoded.split(" ", 1)
        combined = left + right
        if left not in vocabulary or right not in vocabulary or combined not in vocabulary:
            raise ValueError(f"invalid BPE merge at rank {rank}")
        merges.append((int(vocabulary[left]), int(vocabulary[right]), int(vocabulary[combined]), rank))

    metadata = json.dumps(
        {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "normalizer": tokenizer.get("normalizer"),
            "pre_tokenizer": tokenizer.get("pre_tokenizer"),
            "decoder": tokenizer.get("decoder"),
            "post_processor": tokenizer.get("post_processor"),
            "pretokenize_regex": tokenizer_config.get("pretokenize_regex"),
            "chat_template": chat_template,
            "actual_vocabulary": VOCAB,
            "padded_vocabulary": VOCAB,
            "eos_token_ids": [2, 11],
            "thinking_generation_suffix": "<|im_start|>assistant\n<think>\n",
            "nonthinking_generation_suffix": "<|im_start|>assistant\n<think></think>",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    token_table = TOKENIZER_HEADER.size
    merge_table = align_value(token_table + VOCAB * TOKEN_ENTRY.size, 64)
    pieces_offset = align_value(merge_table + len(merges) * MERGE_ENTRY.size, 64)
    pieces_bytes = sum(len(piece) for piece in pieces if piece is not None)
    metadata_offset = align_value(pieces_offset + pieces_bytes, 64)
    file_bytes = metadata_offset + len(metadata)
    header = TOKENIZER_HEADER.pack(
        MAGIC_TOKENIZER, 2, TOKENIZER_HEADER.size,
        VOCAB, BASE_VOCAB, len(merges),
        1, 11, 2, 0,
        10, 11, 12, 13, UINT32_MAX, UINT32_MAX, UINT32_MAX,
        TOKEN_ENTRY.size, MERGE_ENTRY.size, len(added_by_id), 0,
        token_table, merge_table, pieces_offset, pieces_bytes,
        metadata_offset, len(metadata), file_bytes, 0,
    )
    partial = tokenizer_partial_path(output_dir)
    if partial.name != "tokenizer.ovb.partial":
        print(f"tokenizer: preserving prior partial; writing {partial}", flush=True)
    cursor = pieces_offset
    with partial.open("xb", buffering=8 * 1024 * 1024) as output:
        output.write(header)
        for token, piece in enumerate(pieces):
            assert piece is not None
            output.write(TOKEN_ENTRY.pack(cursor, len(piece), flags[token]))
            cursor += len(piece)
        align_file(output, 64)
        for merge in merges:
            output.write(MERGE_ENTRY.pack(*merge))
        align_file(output, 64)
        for piece in pieces:
            assert piece is not None
            output.write(piece)
        align_file(output, 64)
        output.write(metadata)
        if output.tell() != file_bytes:
            raise AssertionError("tokenizer file size drift")
        output.flush()
        os.fsync(output.fileno())
    partial.rename(final)
    validate_existing_tokenizer(final)
    print(f"tokenizer: {final} ({file_bytes:,} bytes)")
    return final


def resolve_metadata(source: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if (source / "tokenizer.json").is_file() and (source / "config.json").is_file():
        return source
    sibling = source.parent / "official"
    if (sibling / "tokenizer.json").is_file() and (sibling / "config.json").is_file():
        return sibling
    raise FileNotFoundError("could not locate official config/tokenizer metadata; pass --metadata")


def validate_config(metadata_dir: Path) -> None:
    config = json.loads((metadata_dir / "config.json").read_text("utf-8"))
    expected = {
        "hidden_size": DIM,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "mamba_num_heads": MAMBA_HEADS,
        "mamba_head_dim": MAMBA_HEAD_DIM,
        "n_groups": MAMBA_GROUPS,
        "ssm_state_size": MAMBA_STATE,
        "conv_kernel": LINEAR_CONV_KERNEL,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "moe_intermediate_size": MOE_DIM,
        "moe_shared_expert_intermediate_size": SHARED_DIM,
        "vocab_size": VOCAB,
        "max_position_embeddings": MAX_POSITION,
        "norm_eps": 1e-5,
        "routed_scaling_factor": 2.5,
    }
    if config.get("model_type") != "nemotron_h":
        raise ValueError("metadata is not Nemotron-H")
    for key, wanted in expected.items():
        if config.get(key) != wanted:
            raise ValueError(f"unexpected config.{key}: {config.get(key)!r} != {wanted!r}")
    if config.get("hybrid_override_pattern") != PATTERN:
        raise ValueError("official Nemotron layer pattern changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory containing all 5 official safetensors shards")
    parser.add_argument("--metadata", type=Path,
                        help="official config/tokenizer directory (defaults to source or sibling official)")
    parser.add_argument("--output", type=Path, required=True,
                        help="new Nemotron runtime output directory")
    parser.add_argument("--tokenizer-only", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--inspect", action="store_true",
                        help="validate the complete checkpoint and conversion plan without writing weights")
    parser.add_argument("--flush-records", type=int, default=32,
                        help="fsync expert output every N completed records (default: 32)")
    arguments = parser.parse_args()
    if arguments.flush_records <= 0:
        parser.error("--flush-records must be positive")
    if arguments.tokenizer_only and arguments.skip_tokenizer:
        parser.error("--tokenizer-only and --skip-tokenizer are mutually exclusive")

    metadata_dir = resolve_metadata(arguments.source, arguments.metadata)
    validate_config(metadata_dir)
    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_tokenizer:
        write_tokenizer(metadata_dir, arguments.output)
    if arguments.tokenizer_only:
        return

    store = SafeStore(arguments.source)
    plans = build_plans()
    validate_checkpoint(store, plans)
    if arguments.inspect:
        return
    if not arguments.skip_shared:
        write_shared(arguments.output, store, plans)
    if not arguments.skip_experts:
        write_experts(arguments.output, store, arguments.flush_records)


if __name__ == "__main__":
    main()
