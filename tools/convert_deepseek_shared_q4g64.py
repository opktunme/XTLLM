#!/usr/bin/env python3
"""Convert DeepSeek model.ovs shared-layer matrices from row-Q8 to Q4G64T.

This is a deliberately narrow companion to convert_deepseek_v4.py.  It reads
the already validated/self-indexed row-Q8 model.ovs and rebuilds the same
container with selected layer matrices in a GPU-native grouped-Q4 layout.
Global tensors (including embedding/head) and routers remain Q8 by default.

Q4G64T tensor ABI (TensorFormat 102):
  data: int4 two's-complement weights, two consecutive K values per byte
        (K-even low nibble, K-odd high nibble), range [-7, 7].  Each uint packs
        eight K-consecutive values. Within each row uints are transposed as
        [word_in_K64=0..7][K64_group], so adjacent lanes read adjacent groups.
  auxiliary: BF16 scale for each consecutive K64 group, row major
  value(row, k) = sign_extend_nibble(data[row, k]) * auxiliary[row, k//64]
  row_stride = columns / 2 bytes
  auxiliary_stride = (columns / 64) * 2 bytes
  flags = 64 (the K group size)

The output is written as OUTPUT.partial and atomically renamed only after it
has been flushed.  Existing outputs/partials are never overwritten or deleted.
"""

from __future__ import annotations

import argparse
import mmap
import os
from pathlib import Path
import struct
from typing import BinaryIO, Iterable, List, Sequence, Set, Tuple

import numpy as np


SHARED_MAGIC = b"OVD4SHR\0"
VERSION = 1
HEADER_BYTES = 4096
SHARED_FORMAT_Q8 = 2
SHARED_FORMAT_MIXED_Q4G64 = 3
Q8_ROW = 100
Q4G64T = 102
Q4_GROUP = 64

SHARED_HEADER = struct.Struct("<8s4I32I12f10Q40x")
GROUP_ENTRY = struct.Struct("<iiIIQQQQ16x")
TENSOR_ENTRY = struct.Struct("<96sII8Q8Q24x")

SHARED_FORMAT_OFFSET = 16
Q_FIELDS_OFFSET = 200
FILE_BYTES_Q_INDEX = 5


def align_value(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def align_file(output: BinaryIO, alignment: int) -> None:
    padding = (-output.tell()) % alignment
    if padding:
        output.write(bytes(padding))


def bounded_name(raw: bytes) -> str:
    end = raw.find(b"\0")
    if end < 0:
        raise ValueError("unterminated tensor name")
    return raw[:end].decode("utf-8")


def parse_layers(text: str, count: int) -> Set[int]:
    if text.strip().lower() == "all":
        return set(range(count))
    result: Set[int] = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first, last = int(first_text), int(last_text)
            if last < first:
                raise ValueError(f"invalid layer range {item!r}")
            result.update(range(first, last + 1))
        else:
            result.add(int(item))
    if not result or min(result) < 0 or max(result) >= count:
        raise ValueError(f"layers must be in [0, {count - 1}]")
    return result


def copy_range(source: mmap.mmap, output: BinaryIO, offset: int, count: int,
               chunk_bytes: int = 16 * 1024 * 1024) -> None:
    while count:
        take = min(count, chunk_bytes)
        output.write(source[offset:offset + take])
        offset += take
        count -= take


def f32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Round finite IEEE-F32 values to BF16, returning little-endian uint16."""
    bits = np.ascontiguousarray(values, dtype="<f4").view("<u4")
    rounded = bits + (np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1)))
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype="<u2")


def convert_q8_matrix(
    source: mmap.mmap,
    output: BinaryIO,
    data_offset: int,
    auxiliary_offset: int,
    rows: int,
    columns: int,
    chunk_rows: int,
) -> Tuple[int, int, int, int]:
    """Write Q4G64T auxiliary then data and return their offsets/byte counts."""
    if columns % Q4_GROUP or columns % 2:
        raise ValueError(f"Q4G64 requires K divisible by {Q4_GROUP}, got {columns}")
    groups = columns // Q4_GROUP
    q8 = np.ndarray((rows, columns), dtype=np.int8, buffer=source, offset=data_offset)
    row_scales = np.ndarray((rows,), dtype="<f4", buffer=source, offset=auxiliary_offset)

    # Maxima are small and avoid retaining/reconstructing a full FP32 matrix.
    maxima = np.empty((rows, groups), dtype=np.uint8)
    q4_scales = np.empty((rows, groups), dtype="<u2")
    for first in range(0, rows, chunk_rows):
        last = min(rows, first + chunk_rows)
        values = q8[first:last].reshape(last - first, groups, Q4_GROUP)
        maximum = np.max(np.abs(values.astype(np.int16)), axis=2).astype(np.uint8)
        maxima[first:last] = maximum
        scale = (row_scales[first:last, None].astype(np.float32) *
                 maximum.astype(np.float32) / np.float32(7.0))
        q4_scales[first:last] = f32_to_bf16(scale)

    align_file(output, 64)
    new_auxiliary_offset = output.tell()
    output.write(q4_scales.tobytes(order="C"))
    new_auxiliary_bytes = q4_scales.nbytes

    align_file(output, 64)
    new_data_offset = output.tell()
    for first in range(0, rows, chunk_rows):
        last = min(rows, first + chunk_rows)
        values = q8[first:last].astype(np.int16).reshape(last - first, groups, Q4_GROUP)
        maximum = maxima[first:last].astype(np.int16)[:, :, None]
        quantized = np.zeros_like(values, dtype=np.int8)
        np.rint(
            values.astype(np.float32) * np.float32(7.0) /
            np.maximum(maximum, np.int16(1)).astype(np.float32),
            out=quantized,
            casting="unsafe",
        )
        np.clip(quantized, -7, 7, out=quantized)
        # One uint contains eight consecutive K values.  Transpose the eight
        # uint positions within every K64 group so a wave reading one group per
        # lane gets coalesced addresses: [row][word_in_group][group].
        nibbles = quantized.reshape(last - first, groups, 8, 8).view(np.uint8)
        packed = np.zeros((last - first, groups, 8), dtype="<u4")
        for nibble in range(8):
            packed |= ((nibbles[:, :, :, nibble] & np.uint8(0x0F)).astype(np.uint32)
                       << np.uint32(nibble * 4))
        transposed = np.ascontiguousarray(packed.transpose(0, 2, 1), dtype="<u4")
        output.write(transposed.tobytes())
    new_data_bytes = rows * columns // 2
    if output.tell() - new_data_offset != new_data_bytes:
        raise AssertionError("Q4G64 packed-size mismatch")
    return new_data_offset, new_data_bytes, new_auxiliary_offset, new_auxiliary_bytes


def patch_tensor(
    entry: Sequence[object], data_offset: int, data_bytes: int,
    auxiliary_offset: int, auxiliary_bytes: int, dtype: int | None = None,
    row_stride: int | None = None, auxiliary_stride: int | None = None,
    flags: int | None = None,
) -> bytes:
    values = list(entry)
    if dtype is not None:
        values[1] = dtype
    values[11:15] = [data_offset, data_bytes, auxiliary_offset, auxiliary_bytes]
    if row_stride is not None:
        values[15] = row_stride
    if auxiliary_stride is not None:
        values[16] = auxiliary_stride
    if flags is not None:
        values[17] = flags
    return TENSOR_ENTRY.pack(*values)


def should_convert(name: str, dtype: int, rank: int, layer: int,
                   selected_layers: Set[int], include_router: bool,
                   include_global: bool, include_embedding: bool,
                   include_head: bool) -> bool:
    if layer == -1:
        selected = include_global or (include_embedding and name == "embed.weight") or \
            (include_head and name == "head.weight")
        return selected and name in {"embed.weight", "head.weight"} and \
            dtype == Q8_ROW and rank == 2
    if layer not in selected_layers or dtype != Q8_ROW or rank != 2:
        return False
    if not include_router and name.endswith(".ffn.gate.weight"):
        return False
    return True


def convert(arguments: argparse.Namespace) -> None:
    source_path = arguments.source.resolve()
    output_path = arguments.output.resolve()
    partial_path = Path(str(output_path) + ".partial")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing {output_path}")
    if partial_path.exists():
        raise FileExistsError(f"preserving existing incomplete {partial_path}")
    if source_path == output_path or source_path == partial_path:
        raise ValueError("source and output paths must differ")

    with source_path.open("rb") as source_file:
        source = mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            if len(source) < HEADER_BYTES:
                raise ValueError("truncated DeepSeek shared container")
            header = SHARED_HEADER.unpack_from(source)
            if (header[0] != SHARED_MAGIC or header[1] != VERSION or
                    header[2] != HEADER_BYTES or header[3] != SHARED_FORMAT_Q8 or
                    header[4] != TENSOR_ENTRY.size):
                raise ValueError("source must be a complete row-Q8 DeepSeek model.ovs")
            layers = int(header[7])
            selected_layers = parse_layers(arguments.layers, layers)
            q_fields = struct.unpack_from("<10Q", source, Q_FIELDS_OFFSET)
            (group_table_offset, group_count, tensor_table_offset, tensor_count,
             data_offset, source_file_bytes, *_) = q_fields
            if source_file_bytes != len(source):
                raise ValueError("source header file_bytes mismatch")
            if tensor_table_offset + tensor_count * TENSOR_ENTRY.size > data_offset:
                raise ValueError("source tensor table overlaps data")
            groups = [GROUP_ENTRY.unpack_from(source, group_table_offset + i * GROUP_ENTRY.size)
                      for i in range(group_count)]
            entries = [TENSOR_ENTRY.unpack_from(source, tensor_table_offset + i * TENSOR_ENTRY.size)
                       for i in range(tensor_count)]

            tensor_layer = [-2] * tensor_count
            for group in groups:
                kind, layer, first, count = group[:4]
                if first + count > tensor_count:
                    raise ValueError("invalid source tensor group")
                for index in range(first, first + count):
                    tensor_layer[index] = layer if kind == 1 else -1
            if any(layer == -2 for layer in tensor_layer):
                raise ValueError("ungrouped source tensor")

            selected: Set[int] = set()
            source_bytes = 0
            output_payload_bytes = 0
            for index, entry in enumerate(entries):
                name = bounded_name(entry[0])
                dtype, rank = int(entry[1]), int(entry[2])
                if should_convert(name, dtype, rank, tensor_layer[index], selected_layers,
                                  arguments.include_router, arguments.include_global,
                                  arguments.include_embedding, arguments.include_head):
                    rows, columns = int(entry[3]), int(entry[4])
                    if columns % Q4_GROUP:
                        raise ValueError(f"{name}: K={columns} is not Q4G64 compatible")
                    if (entry[12] != rows * columns or entry[14] != rows * 4 or
                            entry[15] != columns or entry[16] != 4):
                        raise ValueError(f"{name}: unexpected row-Q8 ABI")
                    selected.add(index)
                    source_bytes += int(entry[12] + entry[14])
                    output_payload_bytes += rows * columns // 2 + rows * (columns // 64) * 2

            if not selected:
                raise ValueError("no layer Q8 matrices selected")
            estimated = len(source) - source_bytes + output_payload_bytes
            print(f"selected matrices: {len(selected)}")
            print(f"selected source Q8: {source_bytes:,} bytes")
            print(f"selected Q4G64 payload: {output_payload_bytes:,} bytes")
            print(f"estimated output (before changed alignment): {estimated:,} bytes")
            if arguments.dry_run:
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)
            patched_entries: List[bytes] = [b""] * tensor_count
            patched_groups: List[bytes] = []
            with partial_path.open("xb", buffering=16 * 1024 * 1024) as output:
                output.write(bytes(data_offset))
                for group_index, group in enumerate(groups):
                    kind, layer, first, count = group[:4]
                    align_file(output, HEADER_BYTES)
                    group_begin = output.tell()
                    for tensor_index in range(first, first + count):
                        entry = entries[tensor_index]
                        name = bounded_name(entry[0])
                        rank = int(entry[2])
                        rows = int(entry[3]) if rank else 0
                        columns = int(entry[4]) if rank >= 2 else 0
                        old_data_offset, old_data_bytes = int(entry[11]), int(entry[12])
                        old_aux_offset, old_aux_bytes = int(entry[13]), int(entry[14])
                        if tensor_index in selected:
                            (new_data_offset, new_data_bytes, new_aux_offset,
                             new_aux_bytes) = convert_q8_matrix(
                                source, output, old_data_offset, old_aux_offset,
                                rows, columns, arguments.chunk_rows)
                            patched_entries[tensor_index] = patch_tensor(
                                entry, new_data_offset, new_data_bytes,
                                new_aux_offset, new_aux_bytes, dtype=Q4G64T,
                                row_stride=columns // 2,
                                auxiliary_stride=(columns // Q4_GROUP) * 2,
                                flags=Q4_GROUP)
                        else:
                            align_file(output, 64)
                            new_data_offset = output.tell()
                            copy_range(source, output, old_data_offset, old_data_bytes)
                            new_aux_offset = 0
                            if old_aux_bytes:
                                align_file(output, 64)
                                new_aux_offset = output.tell()
                                copy_range(source, output, old_aux_offset, old_aux_bytes)
                            patched_entries[tensor_index] = patch_tensor(
                                entry, new_data_offset, old_data_bytes,
                                new_aux_offset, old_aux_bytes)
                    align_file(output, HEADER_BYTES)
                    group_end = output.tell()
                    patched_groups.append(GROUP_ENTRY.pack(
                        kind, layer, first, count, group_begin, group_end,
                        group_begin, group_end - group_begin))
                    label = "global" if kind == 0 else f"layer {layer + 1}/{layers}"
                    print(f"{label}: {group_end - group_begin:,} bytes", flush=True)

                final_bytes = output.tell()
                patched_header = bytearray(source[:HEADER_BYTES])
                struct.pack_into("<I", patched_header, SHARED_FORMAT_OFFSET,
                                 SHARED_FORMAT_MIXED_Q4G64)
                struct.pack_into("<Q", patched_header,
                                 Q_FIELDS_OFFSET + FILE_BYTES_Q_INDEX * 8, final_bytes)
                for index, encoded in enumerate(patched_groups):
                    begin = group_table_offset + index * GROUP_ENTRY.size
                    patched_header[begin:begin + GROUP_ENTRY.size] = encoded
                output.seek(0)
                output.write(patched_header)
                output.seek(tensor_table_offset)
                for encoded in patched_entries:
                    if len(encoded) != TENSOR_ENTRY.size:
                        raise AssertionError("missing patched tensor metadata")
                    output.write(encoded)
                output.flush()
                os.fsync(output.fileno())

            partial_path.rename(output_path)
            print(f"wrote {output_path}: {final_bytes:,} bytes")
        finally:
            source.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="row-Q8 model.ovs")
    parser.add_argument("output", type=Path, help="new mixed model-q4g64.ovs")
    parser.add_argument("--layers", default="all",
                        help="zero-based layers: all, 0, 0-7, or 0,2,4-8")
    parser.add_argument("--include-router", action="store_true",
                        help="also quantize ffn.gate.weight (default keeps router Q8)")
    parser.add_argument("--include-global", action="store_true",
                        help="also quantize embed.weight and head.weight")
    parser.add_argument("--include-embedding", action="store_true",
                        help="also quantize embed.weight only")
    parser.add_argument("--include-head", action="store_true",
                        help="also quantize head.weight only")
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and report sizes without writing output")
    arguments = parser.parse_args()
    if arguments.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    convert(arguments)


if __name__ == "__main__":
    main()
