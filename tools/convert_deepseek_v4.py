#!/usr/bin/env python3
"""Convert DeepSeek-V4-Flash-0731 into the native Vulkan runtime layout.

The converter deliberately preserves routed experts byte-for-byte in the
checkpoint's packed E2M1/UE8M0 representation.  Everything else is grouped in
one self-indexed model.ovs file so a layer can be mapped/read independently.
The default shared representation is per-output-row symmetric Q8, which maps
directly to the RX 6700 XT's packed integer dot-product path.  Passing
--shared-format native-fp8 instead preserves checkpoint FP8 matrices and their
128x128 UE8M0 scale grids for reference work.

On-disk contracts (all integers little endian):

  model.ovs
    SharedHeader  <8s4I32I12f10Q40x>       320 bytes
    GroupEntry    <iiIIQQQQ16x>             64 bytes
    TensorEntry   <96sII8Q8Q24x>           256 bytes
    Header area is 4096 bytes.  Tensor table follows at 4096 and data groups
    are 4096-byte aligned.  The exact corresponding C++ structs live in
    src/m13_deepseek_v4.cpp.

  experts.ovx
    ExpertHeader  <8s8I12Q>                136 bytes, padded to 4096
    Each (layer, expert) record is exactly 13,369,344 bytes:
      w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale
    Weight bytes are raw torch float4_e2m1fn_x2 (K-even low nibble, K-odd high
    nibble); scale bytes are raw float8_e8m0fnu, one per 32 logical K values.

  tokenizer.ovb
    TokenizerHeader <8s20I8Q104x>          256 bytes
    TokenEntry      <QII>                   16 bytes
    MergeEntry      <4I>                    16 bytes
    Token pieces are raw decoded bytes and BPE merges are explicit
    (left-id, right-id, result-id, rank), so C++ never has to infer merge rank
    from vocabulary IDs.  Canonical pre-tokenizer JSON is retained at EOF.

The script preflights every referenced safetensors shard before creating any
output.  An interrupted experts.ovx.partial resumes only at a complete fixed
record boundary; no existing completed output is overwritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import mmap
import os
from pathlib import Path
import re
import struct
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SHARED_MAGIC = b"OVD4SHR\0"
EXPERT_MAGIC = b"OVD4EXP\0"
TOKENIZER_MAGIC = b"OVBPE2\0\0"
VERSION = 1
HEADER_BYTES = 4096

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

DIMENSION = 4096
MOE_DIMENSION = 2048
LAYERS = 43
EXPERTS = 256
TOP_K = 6
VOCABULARY = 129280
BASE_VOCABULARY = 128000

EXPERT_WEIGHT_BYTES = 4 * 1024 * 1024
EXPERT_SCALE_BYTES = 256 * 1024
EXPERT_RECORD_BYTES = 13_369_344
EXPERT_W1_WEIGHT = 0
EXPERT_W1_SCALE = EXPERT_W1_WEIGHT + EXPERT_WEIGHT_BYTES
EXPERT_W3_WEIGHT = EXPERT_W1_SCALE + EXPERT_SCALE_BYTES
EXPERT_W3_SCALE = EXPERT_W3_WEIGHT + EXPERT_WEIGHT_BYTES
EXPERT_W2_WEIGHT = EXPERT_W3_SCALE + EXPERT_SCALE_BYTES
EXPERT_W2_SCALE = EXPERT_W2_WEIGHT + EXPERT_WEIGHT_BYTES
assert EXPERT_W2_SCALE + EXPERT_SCALE_BYTES == EXPERT_RECORD_BYTES

# TensorFormat values are ABI with src/m13_deepseek_v4.cpp.
F32 = 1
BF16 = 2
F16 = 3
E4M3 = 4
E8M0 = 5
I8 = 6
U8 = 7
I64 = 8
U32 = 9
Q8_ROW = 100
FP8_BLOCK128 = 101

RAW_DTYPE_CODE = {
    "F32": F32,
    "BF16": BF16,
    "F16": F16,
    "F8_E4M3": E4M3,
    "F8_E8M0": E8M0,
    "I8": I8,
    "U8": U8,
    "I64": I64,
    "U32": U32,
}

NUMPY_DTYPE = {
    "F32": np.dtype("<f4"),
    "BF16": np.dtype("<u2"),
    "F16": np.dtype("<f2"),
    "F8_E4M3": np.dtype("u1"),
    "F8_E8M0": np.dtype("u1"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "I64": np.dtype("<i8"),
    "U32": np.dtype("<u4"),
}

EXPERT_RE = re.compile(
    r"^(?:layers\.\d+|mtp\.\d+)\.ffn\.experts\.\d+\.w[123]\.(?:weight|scale)$"
)
LAYER_RE = re.compile(r"^layers\.(\d+)\.")
MTP_RE = re.compile(r"^mtp\.(\d+)\.")


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
class SafeTensorInfo:
    name: str
    shard: str
    dtype: str
    shape: Tuple[int, ...]
    data_start: int
    begin: int
    end: int

    @property
    def bytes(self) -> int:
        return self.end - self.begin


class SafeTensorStore:
    """Lazy mmap access with an eager, no-output preflight of every shard."""

    def __init__(self, directory: Path):
        self.directory = directory
        index_path = directory / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as source:
            index = json.load(source)
        self.weight_map: Dict[str, str] = index["weight_map"]
        self.names = list(self.weight_map)
        self._infos: Dict[str, SafeTensorInfo] = {}
        self._files: Dict[str, BinaryIO] = {}
        self._maps: Dict[str, mmap.mmap] = {}
        self._preflight()

    def _preflight(self) -> None:
        by_shard: Dict[str, List[str]] = {}
        for name, shard in self.weight_map.items():
            by_shard.setdefault(shard, []).append(name)
        missing = [name for name in sorted(by_shard) if not (self.directory / name).is_file()]
        if missing:
            preview = ", ".join(missing[:8])
            if len(missing) > 8:
                preview += f", ... (+{len(missing) - 8})"
            raise FileNotFoundError(
                f"checkpoint incomplete: {len(missing)}/{len(by_shard)} shards missing: {preview}"
            )

        for shard, required in sorted(by_shard.items()):
            path = self.directory / shard
            file_bytes = path.stat().st_size
            with path.open("rb") as source:
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"{shard}: truncated safetensors prefix")
                header_bytes = struct.unpack("<Q", prefix)[0]
                if header_bytes == 0 or header_bytes > 512 * 1024 * 1024:
                    raise ValueError(f"{shard}: implausible header size {header_bytes}")
                encoded = source.read(header_bytes)
                if len(encoded) != header_bytes:
                    raise ValueError(f"{shard}: truncated safetensors header")
            try:
                header = json.loads(encoded)
            except Exception as error:
                raise ValueError(f"{shard}: invalid safetensors header: {error}") from error
            data_start = 8 + header_bytes
            maximum_end = 0
            for name in required:
                if name not in header:
                    raise ValueError(f"{shard}: index tensor missing from shard header: {name}")
                entry = header[name]
                dtype = entry["dtype"]
                if dtype not in NUMPY_DTYPE:
                    raise ValueError(f"{name}: unsupported safetensors dtype {dtype}")
                shape = tuple(int(value) for value in entry["shape"])
                begin, end = (int(value) for value in entry["data_offsets"])
                expected = product(shape) * NUMPY_DTYPE[dtype].itemsize
                if begin < 0 or end < begin or end - begin != expected:
                    raise ValueError(
                        f"{name}: bad data extent {begin}:{end} for {dtype} {shape}"
                    )
                if data_start + end > file_bytes:
                    raise ValueError(f"{shard}: partial data for {name}")
                maximum_end = max(maximum_end, end)
                self._infos[name] = SafeTensorInfo(
                    name, shard, dtype, shape, data_start, begin, end
                )
            # Safetensors has no trailing payload.  Equality catches a download
            # that happened to end between tensors rather than within one.
            if data_start + maximum_end != file_bytes:
                raise ValueError(
                    f"{shard}: file size {file_bytes} != indexed extent "
                    f"{data_start + maximum_end}; shard may be partial"
                )

    def info(self, name: str) -> SafeTensorInfo:
        try:
            return self._infos[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor not found: {name}") from error

    def _mapping(self, shard: str) -> mmap.mmap:
        if shard not in self._maps:
            source = (self.directory / shard).open("rb")
            self._files[shard] = source
            self._maps[shard] = mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
        return self._maps[shard]

    def array(self, name: str) -> np.ndarray:
        info = self.info(name)
        dtype = NUMPY_DTYPE[info.dtype]
        mapping = self._mapping(info.shard)
        values = np.frombuffer(
            mapping,
            dtype=dtype,
            count=product(info.shape),
            offset=info.data_start + info.begin,
        )
        return values.reshape(info.shape)

    def raw(self, name: str) -> memoryview:
        info = self.info(name)
        mapping = self._mapping(info.shard)
        begin = info.data_start + info.begin
        return memoryview(mapping)[begin : begin + info.bytes]


def make_e4m3_lut() -> np.ndarray:
    values = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exponent = (code >> 3) & 0x0F
        mantissa = code & 0x07
        if exponent == 0:
            value = math.ldexp(mantissa / 8.0, -6)
        elif exponent == 15 and mantissa == 7:
            value = math.nan
        else:
            value = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
        values[code] = sign * value
    return values


E4M3_LUT = make_e4m3_lut()
E8M0_LUT = np.empty(256, dtype=np.float32)
E8M0_LUT[:255] = np.ldexp(
    np.ones(255, dtype=np.float64), np.arange(255, dtype=np.int32) - 127
).astype(np.float32)
E8M0_LUT[255] = np.nan


def bf16_to_f32(values: np.ndarray) -> np.ndarray:
    words = np.asarray(values, dtype=np.uint16).astype(np.uint32)
    return (words << np.uint32(16)).view(np.float32)


def matrix_rows_f32(
    store: SafeTensorStore,
    name: str,
    first: int,
    last: int,
    scale_name: Optional[str] = None,
) -> np.ndarray:
    info = store.info(name)
    source = store.array(name)[first:last]
    if info.dtype == "BF16":
        return bf16_to_f32(source)
    if info.dtype == "F16" or info.dtype == "F32":
        return np.asarray(source, dtype=np.float32)
    if info.dtype == "F8_E4M3":
        if scale_name is None:
            raise ValueError(f"{name}: FP8 matrix has no scale tensor")
        decoded = E4M3_LUT[np.asarray(source, dtype=np.uint8)]
        scale_info = store.info(scale_name)
        scales = store.array(scale_name)
        rows, columns = info.shape
        expected = ((rows + 127) // 128, (columns + 127) // 128)
        if scale_info.dtype != "F8_E8M0" or scale_info.shape != expected:
            raise ValueError(
                f"{scale_name}: expected E8M0 {expected}, got "
                f"{scale_info.dtype} {scale_info.shape}"
            )
        row_blocks = np.arange(first, last, dtype=np.int64) // 128
        block_scales = E8M0_LUT[np.asarray(scales[row_blocks], dtype=np.uint8)]
        expanded = np.repeat(block_scales, 128, axis=1)[:, :columns]
        result = decoded * expanded
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name}: non-finite FP8 dequantization")
        return result
    raise ValueError(f"{name}: cannot convert {info.dtype} matrix to Q8")


@dataclass
class TensorPlan:
    name: str
    source: str
    mode: str
    dtype: int
    shape: Tuple[int, ...]
    auxiliary: Optional[str] = None


@dataclass
class TensorResult:
    plan: TensorPlan
    data_offset: int
    data_bytes: int
    auxiliary_offset: int
    auxiliary_bytes: int
    row_stride: int
    auxiliary_stride: int
    flags: int

    def pack(self) -> bytes:
        encoded = self.plan.name.encode("utf-8")
        if len(encoded) >= 96:
            raise ValueError(f"tensor name too long for model.ovs: {self.plan.name}")
        dimensions = list(self.plan.shape) + [0] * (8 - len(self.plan.shape))
        return TENSOR_ENTRY.pack(
            encoded,
            self.plan.dtype,
            len(self.plan.shape),
            *dimensions,
            self.data_offset,
            self.data_bytes,
            self.auxiliary_offset,
            self.auxiliary_bytes,
            self.row_stride,
            self.auxiliary_stride,
            self.flags,
            0,
        )


@dataclass
class GroupPlan:
    kind: int
    index: int
    tensors: List[TensorPlan]


def fp8_scale_name(weight_name: str) -> str:
    if not weight_name.endswith("weight"):
        raise ValueError(f"FP8 tensor is not a named weight: {weight_name}")
    return weight_name[: -len("weight")] + "scale"


def build_groups(
    store: SafeTensorStore, shared_format: str, include_indexer: bool
) -> List[GroupPlan]:
    included: List[str] = []
    for name in store.names:
        if EXPERT_RE.match(name) or MTP_RE.match(name):
            continue
        # For short-context bring-up CSA has <= index_topk compressed slots, so
        # the learned indexer cannot change the selected set.  It can be kept
        # for long-context work with --include-indexer.
        if not include_indexer and ".attn.indexer." in name:
            continue
        included.append(name)

    included_set = set(included)
    consumed_scales = set()
    for name in included:
        info = store.info(name)
        if info.dtype == "F8_E4M3" and len(info.shape) == 2:
            scale = fp8_scale_name(name)
            if scale not in included_set:
                raise ValueError(f"{name}: missing included scale tensor {scale}")
            consumed_scales.add(scale)

    groups: Dict[Tuple[int, int], List[TensorPlan]] = {(0, -1): []}
    for layer in range(LAYERS):
        groups[(1, layer)] = []

    for name in included:
        if name in consumed_scales:
            continue
        match = LAYER_RE.match(name)
        key = (1, int(match.group(1))) if match else (0, -1)
        if key not in groups:
            raise ValueError(f"unexpected tensor group for {name}")
        info = store.info(name)
        auxiliary: Optional[str] = None

        if name.endswith(".tid2eid"):
            if info.dtype != "I64" or len(info.shape) != 2:
                raise ValueError(f"{name}: expected I64 hash-route table")
            mode, dtype = "u32", U32
        elif len(info.shape) == 2 and info.dtype in ("BF16", "F16", "F8_E4M3"):
            if info.dtype == "F8_E4M3":
                auxiliary = fp8_scale_name(name)
            if shared_format == "q8":
                mode, dtype = "q8", Q8_ROW
            elif info.dtype == "F8_E4M3":
                mode, dtype = "native_fp8", FP8_BLOCK128
            else:
                mode, dtype = "raw", RAW_DTYPE_CODE[info.dtype]
        elif shared_format == "q8" and info.dtype in ("BF16", "F16"):
            # Norms, sinks, and other low-rank/non-matrix values are consumed as
            # F32 by the Vulkan/CPU scalar paths.
            mode, dtype = "f32", F32
        else:
            if info.dtype not in RAW_DTYPE_CODE:
                raise ValueError(f"{name}: unsupported raw dtype {info.dtype}")
            mode, dtype = "raw", RAW_DTYPE_CODE[info.dtype]
        groups[key].append(TensorPlan(name, name, mode, dtype, info.shape, auxiliary))

    result = [GroupPlan(0, -1, groups[(0, -1)])]
    result.extend(GroupPlan(1, layer, groups[(1, layer)]) for layer in range(LAYERS))
    for group in result:
        if not group.tensors:
            raise ValueError(f"empty output tensor group kind={group.kind} index={group.index}")
    return result


def write_raw(output: BinaryIO, store: SafeTensorStore, plan: TensorPlan) -> TensorResult:
    align_file(output, 64)
    offset = output.tell()
    raw = store.raw(plan.source)
    output.write(raw)
    return TensorResult(plan, offset, len(raw), 0, 0, 0, 0, 0)


def write_f32(output: BinaryIO, store: SafeTensorStore, plan: TensorPlan) -> TensorResult:
    align_file(output, 64)
    offset = output.tell()
    info = store.info(plan.source)
    values = store.array(plan.source)
    if info.dtype == "BF16":
        converted = bf16_to_f32(values)
    else:
        converted = np.asarray(values, dtype=np.float32)
    encoded = np.ascontiguousarray(converted, dtype="<f4").tobytes()
    output.write(encoded)
    return TensorResult(plan, offset, len(encoded), 0, 0, 0, 0, 0)


def write_u32(output: BinaryIO, store: SafeTensorStore, plan: TensorPlan) -> TensorResult:
    values = store.array(plan.source)
    if np.any(values < 0) or np.any(values >= EXPERTS):
        raise ValueError(f"{plan.source}: route table contains invalid expert ID")
    converted = np.ascontiguousarray(values, dtype="<u4")
    align_file(output, 64)
    offset = output.tell()
    output.write(converted.tobytes())
    return TensorResult(
        plan,
        offset,
        converted.nbytes,
        0,
        0,
        int(converted.shape[1]) * 4,
        0,
        0,
    )


def write_native_fp8(
    output: BinaryIO, store: SafeTensorStore, plan: TensorPlan
) -> TensorResult:
    if plan.auxiliary is None:
        raise ValueError(f"{plan.source}: native FP8 entry has no scale")
    align_file(output, 64)
    data_offset = output.tell()
    weights = store.raw(plan.source)
    output.write(weights)
    align_file(output, 64)
    auxiliary_offset = output.tell()
    scales = store.raw(plan.auxiliary)
    output.write(scales)
    rows, columns = plan.shape
    return TensorResult(
        plan,
        data_offset,
        len(weights),
        auxiliary_offset,
        len(scales),
        columns,
        (columns + 127) // 128,
        128 | (128 << 16),
    )


def write_q8(
    output: BinaryIO,
    store: SafeTensorStore,
    plan: TensorPlan,
    chunk_rows: int = 128,
) -> TensorResult:
    rows, columns = plan.shape
    scales = np.empty(rows, dtype="<f4")
    for first in range(0, rows, chunk_rows):
        last = min(rows, first + chunk_rows)
        values = matrix_rows_f32(store, plan.source, first, last, plan.auxiliary)
        maximum = np.max(np.abs(values), axis=1)
        scales[first:last] = np.where(
            maximum > 0.0, maximum / np.float32(127.0), np.float32(1.0)
        )
    align_file(output, 64)
    auxiliary_offset = output.tell()
    output.write(scales.tobytes())
    align_file(output, 64)
    data_offset = output.tell()
    for first in range(0, rows, chunk_rows):
        last = min(rows, first + chunk_rows)
        values = matrix_rows_f32(store, plan.source, first, last, plan.auxiliary)
        quantized = np.rint(values / scales[first:last, None]).clip(-127, 127)
        output.write(np.ascontiguousarray(quantized, dtype=np.int8).tobytes())
    return TensorResult(
        plan,
        data_offset,
        rows * columns,
        auxiliary_offset,
        rows * 4,
        columns,
        4,
        1,  # symmetric signed Q8, one F32 scale per output row
    )


def write_tensor(output: BinaryIO, store: SafeTensorStore, plan: TensorPlan) -> TensorResult:
    if plan.mode == "raw":
        return write_raw(output, store, plan)
    if plan.mode == "f32":
        return write_f32(output, store, plan)
    if plan.mode == "u32":
        return write_u32(output, store, plan)
    if plan.mode == "native_fp8":
        return write_native_fp8(output, store, plan)
    if plan.mode == "q8":
        return write_q8(output, store, plan)
    raise AssertionError(f"unknown tensor plan mode {plan.mode}")


def special_token_ids(source_dir: Path) -> Dict[str, int]:
    with (source_dir / "tokenizer.json").open("r", encoding="utf-8") as source:
        tokenizer = json.load(source)
    lookup = dict(tokenizer["model"]["vocab"])
    for entry in tokenizer.get("added_tokens", []):
        lookup[entry["content"]] = int(entry["id"])
    required = {
        "bos": "<｜begin▁of▁sentence｜>",
        "eos": "<｜end▁of▁sentence｜>",
        "pad_piece": "<｜▁pad▁｜>",
        "user": "<｜User｜>",
        "assistant": "<｜Assistant｜>",
        "think": "<think>",
        "end_think": "</think>",
        "dsml": "｜DSML｜",
        "latest_reminder": "<｜latest_reminder｜>",
    }
    result = {}
    for key, piece in required.items():
        if piece not in lookup:
            raise ValueError(f"tokenizer missing official piece {piece!r}")
        result[key] = int(lookup[piece])
    if result["bos"] != 0 or result["eos"] != 1 or result["pad_piece"] != 2:
        raise ValueError(f"unexpected DeepSeek special-token IDs: {result}")
    # tokenizer_config.json intentionally aliases configured padding to EOS.
    result["pad"] = result["eos"]
    return result


def validate_config(config: dict) -> None:
    expected = {
        "hidden_size": DIMENSION,
        "moe_intermediate_size": MOE_DIMENSION,
        "num_hidden_layers": LAYERS,
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOP_K,
        "vocab_size": VOCABULARY,
        "expert_dtype": "fp4",
        "num_nextn_predict_layers": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"config {key}={config.get(key)!r}; expected {value!r}")
    quant = config.get("quantization_config", {})
    if quant.get("fmt") != "e4m3" or quant.get("scale_fmt") != "ue8m0" or tuple(
        quant.get("weight_block_size", [])
    ) != (128, 128):
        raise ValueError("checkpoint does not use expected E4M3/UE8M0 128x128 shared format")


def completed_shared(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < HEADER_BYTES:
        return False
    with path.open("rb") as source:
        encoded = source.read(SHARED_HEADER.size)
    values = SHARED_HEADER.unpack(encoded)
    return (
        values[0] == SHARED_MAGIC
        and values[1] == VERSION
        and values[2] == HEADER_BYTES
        and values[-5] == path.stat().st_size  # file_bytes is Q field 6 of 10
    )


def write_shared(
    source_dir: Path,
    output_dir: Path,
    store: SafeTensorStore,
    config: dict,
    shared_format: str,
    include_indexer: bool,
) -> Path:
    final = output_dir / "model.ovs"
    partial = output_dir / "model.ovs.partial"
    if final.exists():
        if not completed_shared(final):
            raise ValueError(f"existing {final} is not a complete DeepSeek container")
        print(f"shared: keeping existing {final}")
        return final
    if partial.exists():
        raise FileExistsError(
            f"incomplete {partial} already exists; preserve it and choose a new --output directory"
        )

    groups = build_groups(store, shared_format, include_indexer)
    flat_plans = [plan for group in groups for plan in group.tensors]
    tensor_table_offset = HEADER_BYTES
    data_offset = align_value(
        tensor_table_offset + len(flat_plans) * TENSOR_ENTRY.size, HEADER_BYTES
    )
    token_ids = special_token_ids(source_dir)
    results: List[TensorResult] = []
    group_records = []

    with partial.open("xb", buffering=16 * 1024 * 1024) as output:
        output.write(bytes(data_offset))
        first_tensor = 0
        for ordinal, group in enumerate(groups):
            align_file(output, HEADER_BYTES)
            begin = output.tell()
            for plan in group.tensors:
                results.append(write_tensor(output, store, plan))
            align_file(output, HEADER_BYTES)
            end = output.tell()
            group_records.append(
                (group.kind, group.index, first_tensor, len(group.tensors), begin, end, begin, end - begin)
            )
            first_tensor += len(group.tensors)
            label = "global" if group.kind == 0 else f"layer {group.index + 1}/{LAYERS}"
            print(f"shared {label}: {end - begin:,} bytes", flush=True)
        file_bytes = output.tell()

        output.seek(tensor_table_offset)
        for result in results:
            output.write(result.pack())

        output.seek(0)
        rope = config["rope_scaling"]
        integer_fields = (
            config["hidden_size"],
            config["moe_intermediate_size"],
            config["num_hidden_layers"],
            config["num_attention_heads"],
            config["num_key_value_heads"],
            config["head_dim"],
            config["q_lora_rank"],
            config["o_lora_rank"],
            config["o_groups"],
            config["qk_rope_head_dim"],
            config["vocab_size"],
            config["n_routed_experts"],
            config["num_experts_per_tok"],
            config["n_shared_experts"],
            config["hc_mult"],
            config["hc_sinkhorn_iters"],
            config["num_hash_layers"],
            config["sliding_window"],
            config["index_n_heads"],
            config["index_head_dim"],
            config["index_topk"],
            config["max_position_embeddings"],
            0,  # MTP deliberately omitted from baseline files.
            LAYERS,
            token_ids["bos"],
            token_ids["eos"],
            token_ids["pad"],
            token_ids["user"],
            token_ids["assistant"],
            token_ids["think"],
            token_ids["end_think"],
            token_ids["dsml"],
        )
        float_fields = (
            float(config["rms_norm_eps"]),
            float(config["hc_eps"]),
            float(config["routed_scaling_factor"]),
            float(config["swiglu_limit"]),
            float(config["rope_theta"]),
            float(config["compress_rope_theta"]),
            float(rope["factor"]),
            float(rope["beta_fast"]),
            float(rope["beta_slow"]),
            float(rope["original_max_position_embeddings"]),
            0.0,
            0.0,
        )
        q_fields = (
            SHARED_HEADER.size,
            len(groups),
            tensor_table_offset,
            len(results),
            data_offset,
            file_bytes,
            EXPERT_RECORD_BYTES,
            LAYERS * EXPERTS,
            0,
            0,
        )
        output.write(
            SHARED_HEADER.pack(
                SHARED_MAGIC,
                VERSION,
                HEADER_BYTES,
                2 if shared_format == "q8" else 1,
                TENSOR_ENTRY.size,
                *integer_fields,
                *float_fields,
                *q_fields,
            )
        )
        output.seek(SHARED_HEADER.size)
        for record in group_records:
            output.write(GROUP_ENTRY.pack(*record))
        ratios = [int(value) for value in config["compress_ratios"][:LAYERS]]
        output.write(struct.pack(f"<{LAYERS}I", *ratios))
        output.flush()
        os.fsync(output.fileno())

    partial.rename(final)
    print(f"shared: {final} ({file_bytes:,} bytes)")
    return final


def expert_piece_names(layer: int, expert: int) -> Tuple[str, ...]:
    prefix = f"layers.{layer}.ffn.experts.{expert}"
    return tuple(
        f"{prefix}.{suffix}"
        for suffix in (
            "w1.weight",
            "w1.scale",
            "w3.weight",
            "w3.scale",
            "w2.weight",
            "w2.scale",
        )
    )


def validate_expert_piece(store: SafeTensorStore, name: str, slot: int) -> None:
    info = store.info(name)
    expected = (
        ("I8", (MOE_DIMENSION, DIMENSION // 2), EXPERT_WEIGHT_BYTES),
        ("F8_E8M0", (MOE_DIMENSION, DIMENSION // 32), EXPERT_SCALE_BYTES),
        ("I8", (MOE_DIMENSION, DIMENSION // 2), EXPERT_WEIGHT_BYTES),
        ("F8_E8M0", (MOE_DIMENSION, DIMENSION // 32), EXPERT_SCALE_BYTES),
        ("I8", (DIMENSION, MOE_DIMENSION // 2), EXPERT_WEIGHT_BYTES),
        ("F8_E8M0", (DIMENSION, MOE_DIMENSION // 32), EXPERT_SCALE_BYTES),
    )[slot]
    if info.dtype != expected[0] or info.shape != expected[1] or info.bytes != expected[2]:
        raise ValueError(
            f"{name}: expected {expected}, got {(info.dtype, info.shape, info.bytes)}"
        )


def expert_header_bytes(file_bytes: int) -> bytes:
    core_records = LAYERS * EXPERTS
    encoded = EXPERT_HEADER.pack(
        EXPERT_MAGIC,
        VERSION,
        HEADER_BYTES,
        DIMENSION,
        MOE_DIMENSION,
        LAYERS,
        EXPERTS,
        0,  # MTP intentionally omitted.
        EXPERT_RECORD_BYTES,
        HEADER_BYTES,
        HEADER_BYTES + core_records * EXPERT_RECORD_BYTES,
        core_records,
        core_records,
        file_bytes,
        EXPERT_W1_WEIGHT,
        EXPERT_W1_SCALE,
        EXPERT_W3_WEIGHT,
        EXPERT_W3_SCALE,
        EXPERT_W2_WEIGHT,
        EXPERT_W2_SCALE,
        0,
    )
    return encoded + bytes(HEADER_BYTES - len(encoded))


def completed_experts(path: Path, expected_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected_bytes:
        return False
    with path.open("rb") as source:
        encoded = source.read(EXPERT_HEADER.size)
    values = EXPERT_HEADER.unpack(encoded)
    return values[0] == EXPERT_MAGIC and values[1] == VERSION and values[13] == expected_bytes


def write_experts(output_dir: Path, store: SafeTensorStore) -> Path:
    final = output_dir / "experts.ovx"
    partial = output_dir / "experts.ovx.partial"
    core_records = LAYERS * EXPERTS
    expected_bytes = HEADER_BYTES + core_records * EXPERT_RECORD_BYTES
    if final.exists():
        if not completed_experts(final, expected_bytes):
            raise ValueError(f"existing {final} is not a complete DeepSeek expert container")
        print(f"experts: keeping existing {final}")
        return final

    if partial.exists():
        size = partial.stat().st_size
        if size < HEADER_BYTES or (size - HEADER_BYTES) % EXPERT_RECORD_BYTES:
            raise ValueError(f"{partial}: not at a resumable expert-record boundary")
        with partial.open("rb") as source:
            if source.read(8) != EXPERT_MAGIC:
                raise ValueError(f"{partial}: wrong partial-file magic")
        completed = (size - HEADER_BYTES) // EXPERT_RECORD_BYTES
        if completed > core_records:
            raise ValueError(f"{partial}: contains too many expert records")
        print(f"experts: resuming at record {completed:,}/{core_records:,}")
        mode = "ab"
    else:
        completed = 0
        with partial.open("xb") as output:
            output.write(expert_header_bytes(expected_bytes))
            output.flush()
            os.fsync(output.fileno())
        mode = "ab"

    with partial.open(mode, buffering=16 * 1024 * 1024) as output:
        for record in range(completed, core_records):
            layer, expert = divmod(record, EXPERTS)
            names = expert_piece_names(layer, expert)
            for slot, name in enumerate(names):
                validate_expert_piece(store, name, slot)
                output.write(store.raw(name))
            if expert == EXPERTS - 1:
                output.flush()
                os.fsync(output.fileno())
                print(f"experts layer {layer + 1}/{LAYERS}", flush=True)

    if partial.stat().st_size != expected_bytes:
        raise ValueError(
            f"expert output size {partial.stat().st_size} != expected {expected_bytes}"
        )
    partial.rename(final)
    print(f"experts: {final} ({expected_bytes:,} bytes)")
    return final


def gpt2_byte_decoder() -> Dict[str, int]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(0xA1, 0xAC + 1))
    byte_values += list(range(0xAE, 0xFF + 1))
    codepoints = list(byte_values)
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            codepoints.append(256 + extra)
            extra += 1
    return {chr(codepoint): byte for byte, codepoint in zip(byte_values, codepoints)}


def completed_tokenizer(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < TOKENIZER_HEADER.size:
        return False
    with path.open("rb") as source:
        encoded = source.read(TOKENIZER_HEADER.size)
    values = TOKENIZER_HEADER.unpack(encoded)
    return values[0] == TOKENIZER_MAGIC and values[1] == 2 and values[-2] == path.stat().st_size


def write_tokenizer(source_dir: Path, output_dir: Path) -> Path:
    final = output_dir / "tokenizer.ovb"
    partial = output_dir / "tokenizer.ovb.partial"
    if final.exists():
        if not completed_tokenizer(final):
            raise ValueError(f"existing {final} is not a complete DeepSeek tokenizer")
        print(f"tokenizer: keeping existing {final}")
        return final
    if partial.exists():
        raise FileExistsError(
            f"incomplete {partial} already exists; preserve it and choose a new --output directory"
        )

    with (source_dir / "tokenizer.json").open("r", encoding="utf-8") as source:
        tokenizer = json.load(source)
    model = tokenizer["model"]
    vocabulary = model["vocab"]
    if len(vocabulary) != BASE_VOCABULARY:
        raise ValueError(f"base tokenizer vocabulary is {len(vocabulary)}, expected {BASE_VOCABULARY}")
    added = tokenizer.get("added_tokens", [])
    added_ids = {int(entry["id"]) for entry in added}
    by_id: List[Optional[bytes]] = [None] * VOCABULARY
    flags = [0] * VOCABULARY
    decoder = gpt2_byte_decoder()
    for piece, token in vocabulary.items():
        token = int(token)
        try:
            by_id[token] = bytes(decoder[character] for character in piece)
        except KeyError as error:
            # IDs 0..2 are present in both the base table and added-token
            # overlay; their full-width Unicode spelling is intentionally not
            # a ByteLevel symbol sequence and is replaced just below.
            if token in added_ids:
                continue
            raise ValueError(f"base BPE piece has non-byte-level character: {piece!r}") from error
    for entry in added:
        token = int(entry["id"])
        if token >= VOCABULARY:
            raise ValueError(f"added token ID {token} exceeds model vocabulary")
        by_id[token] = entry["content"].encode("utf-8")
        flags[token] |= 1
        if entry.get("special", False):
            flags[token] |= 2
    for token, piece in enumerate(by_id):
        if piece is None:
            by_id[token] = f"<|unused_{token}|>".encode("ascii")
            flags[token] |= 4

    merges = []
    for rank, encoded in enumerate(model["merges"]):
        if isinstance(encoded, list):
            left, right = encoded
        else:
            left, right = encoded.split(" ", 1)
        merged = left + right
        try:
            merges.append((int(vocabulary[left]), int(vocabulary[right]), int(vocabulary[merged]), rank))
        except KeyError as error:
            raise ValueError(f"merge rank {rank} references unknown piece: {encoded!r}") from error

    ids = special_token_ids(source_dir)
    token_table_offset = TOKENIZER_HEADER.size
    merge_table_offset = align_value(token_table_offset + VOCABULARY * TOKEN_ENTRY.size, 64)
    pieces_offset = align_value(merge_table_offset + len(merges) * MERGE_ENTRY.size, 64)
    pieces_bytes = sum(len(piece) for piece in by_id if piece is not None)
    metadata_offset = align_value(pieces_offset + pieces_bytes, 64)
    metadata = json.dumps(
        {
            "normalizer": tokenizer.get("normalizer"),
            "pre_tokenizer": tokenizer.get("pre_tokenizer"),
            "post_processor": tokenizer.get("post_processor"),
            "decoder": tokenizer.get("decoder"),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    file_bytes = metadata_offset + len(metadata)
    header = TOKENIZER_HEADER.pack(
        TOKENIZER_MAGIC,
        2,
        TOKENIZER_HEADER.size,
        VOCABULARY,
        BASE_VOCABULARY,
        len(merges),
        ids["bos"],
        ids["eos"],
        ids["pad"],
        ids["pad_piece"],
        ids["user"],
        ids["assistant"],
        ids["think"],
        ids["end_think"],
        ids["dsml"],
        ids["latest_reminder"],
        0xFFFFFFFF,
        TOKEN_ENTRY.size,
        MERGE_ENTRY.size,
        len(added),
        0,
        token_table_offset,
        merge_table_offset,
        pieces_offset,
        pieces_bytes,
        metadata_offset,
        len(metadata),
        file_bytes,
        0,
    )

    piece_cursor = pieces_offset
    with partial.open("xb", buffering=8 * 1024 * 1024) as output:
        output.write(header)
        for token in range(VOCABULARY):
            piece = by_id[token]
            assert piece is not None
            output.write(TOKEN_ENTRY.pack(piece_cursor, len(piece), flags[token]))
            piece_cursor += len(piece)
        align_file(output, 64)
        if output.tell() != merge_table_offset:
            raise AssertionError("tokenizer merge table offset drift")
        for merge in merges:
            output.write(MERGE_ENTRY.pack(*merge))
        align_file(output, 64)
        if output.tell() != pieces_offset:
            raise AssertionError("tokenizer piece offset drift")
        for piece in by_id:
            assert piece is not None
            output.write(piece)
        align_file(output, 64)
        if output.tell() != metadata_offset:
            raise AssertionError("tokenizer metadata offset drift")
        output.write(metadata)
        if output.tell() != file_bytes:
            raise AssertionError("tokenizer file size drift")
        output.flush()
        os.fsync(output.fileno())
    partial.rename(final)
    print(f"tokenizer: {final} ({file_bytes:,} bytes, {len(merges):,} merges)")
    return final


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek-V4-Flash-0731 for the native Vulkan runtime"
    )
    parser.add_argument("--source", required=True, type=Path, help="HF checkpoint directory")
    parser.add_argument("--output", required=True, type=Path, help="new runtime model directory")
    parser.add_argument(
        "--shared-format",
        choices=("q8", "native-fp8"),
        default="q8",
        help="shared/dense storage; q8 is the RDNA2 execution format (default)",
    )
    parser.add_argument(
        "--include-indexer",
        action="store_true",
        help="include long-context CSA indexer weights (omitted for short-context bring-up)",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--skip-shared", action="store_true")
    parser.add_argument("--skip-experts", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    source_dir = arguments.source.resolve()
    output_dir = arguments.output.resolve()
    with (source_dir / "config.json").open("r", encoding="utf-8") as source:
        config = json.load(source)
    validate_config(config)
    needs_weights = arguments.validate_only or not (
        arguments.skip_shared and arguments.skip_experts
    )
    store: Optional[SafeTensorStore] = None
    if needs_weights:
        # This is intentionally before mkdir/output opening.  A partial
        # checkpoint can only produce a diagnostic, never a partially
        # converted weight container.
        store = SafeTensorStore(source_dir)
        print(
            f"checkpoint preflight: PASS ({len(set(store.weight_map.values()))} shards, "
            f"{len(store.weight_map):,} tensors)"
        )
        if arguments.validate_only:
            return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_tokenizer:
        write_tokenizer(source_dir, output_dir)
    if not arguments.skip_shared:
        assert store is not None
        write_shared(
            source_dir,
            output_dir,
            store,
            config,
            arguments.shared_format,
            arguments.include_indexer,
        )
    if not arguments.skip_experts:
        assert store is not None
        write_experts(output_dir, store)


if __name__ == "__main__":
    main()
