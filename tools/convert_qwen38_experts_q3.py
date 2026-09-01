#!/usr/bin/env python3
"""Repack the existing Qwen3.8 Q4 expert stream as compact signed Q3/K64.

This is an additive conversion: the Q4 source is read-only, an interrupted
partial is retained, and no existing artifact is deleted or truncated.
"""

from __future__ import annotations

import argparse
import mmap
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_qwen38 as q4


Q4_RECORD_BYTES = 2_613_248
Q3_RECORD_BYTES = 1_998_848
Q3_RECORD_WORDS = Q3_RECORD_BYTES // 4
Q3_GATE_SCALE = 0
Q3_GATE_WEIGHT = 51_200
Q3_UP_SCALE = 665_600
Q3_UP_WEIGHT = 716_800
Q3_DOWN_SCALE = 1_331_200
Q3_DOWN_WEIGHT = 1_382_400
Q3_DATA_BYTES = 1_996_800
RECORDS = q4.LAYERS * q4.EXPERTS


def q3_header(file_bytes: int) -> bytes:
    return q4.EXPERT_HEADER.pack(
        q4.MAGIC_EXPERT, q4.VERSION, q4.HEADER_BYTES, q4.DIM, q4.MOE_DIM,
        q4.LAYERS, q4.EXPERTS, 0, Q3_RECORD_BYTES,
        q4.HEADER_BYTES, file_bytes, RECORDS, RECORDS, file_bytes,
        Q3_GATE_WEIGHT, Q3_GATE_SCALE, Q3_UP_WEIGHT, Q3_UP_SCALE,
        Q3_DOWN_WEIGHT, Q3_DOWN_SCALE, 0,
    )


def validate_q3_header(path: Path, expected: int) -> None:
    with path.open("rb") as source:
        fields = q4.EXPERT_HEADER.unpack(source.read(q4.EXPERT_HEADER.size))
    wanted = q4.EXPERT_HEADER.unpack(q3_header(expected))
    if fields != wanted:
        raise ValueError(f"incompatible Q3 partial/header: {path}")


def repack_matrix(
    source: mmap.mmap,
    record_base: int,
    scale_offset: int,
    weight_offset: int,
    rows: int,
    groups: int,
) -> Tuple[bytes, bytes]:
    scale_u16 = np.ndarray(
        (rows, groups), dtype="<u2", buffer=source,
        offset=record_base + scale_offset,
    )
    q3_scales = q4.f32_bf16(
        q4.bf16_f32(scale_u16) * np.float32(7.0 / 3.0)
    ).tobytes()

    packed_q4 = np.ndarray(
        (rows, 8, groups), dtype="<u4", buffer=source,
        offset=record_base + weight_offset,
    )
    chunks = np.zeros(packed_q4.shape, dtype=np.uint32)
    for nibble in range(8):
        code4 = ((packed_q4 >> np.uint32(4 * nibble)) & np.uint32(15)).astype(np.int16)
        signed4 = np.where(code4 >= 8, code4 - 16, code4)
        signed3 = np.rint(signed4.astype(np.float32) * np.float32(3.0 / 7.0)).astype(np.int16)
        code3 = (signed3 & 7).astype(np.uint32)
        chunks |= code3 << np.uint32(3 * nibble)

    packed_q3 = np.empty((rows, 6, groups), dtype="<u4")
    c0, c1, c2, c3, c4, c5, c6, c7 = (
        chunks[:, index, :] for index in range(8)
    )
    packed_q3[:, 0, :] = c0 | (c1 << np.uint32(24))
    packed_q3[:, 1, :] = (c1 >> np.uint32(8)) | (c2 << np.uint32(16))
    packed_q3[:, 2, :] = (c2 >> np.uint32(16)) | (c3 << np.uint32(8))
    packed_q3[:, 3, :] = c4 | (c5 << np.uint32(24))
    packed_q3[:, 4, :] = (c5 >> np.uint32(8)) | (c6 << np.uint32(16))
    packed_q3[:, 5, :] = (c6 >> np.uint32(16)) | (c7 << np.uint32(8))
    return q3_scales, np.ascontiguousarray(packed_q3).tobytes()


def write_record(output: BinaryIO, source: mmap.mmap, record: int) -> None:
    source_base = q4.HEADER_BYTES + record * Q4_RECORD_BYTES
    target_start = output.tell()
    specs = (
        (q4.GATE_SCALE, q4.GATE_WEIGHT, q4.MOE_DIM, q4.DIM,
         Q3_GATE_SCALE, Q3_GATE_WEIGHT),
        (q4.UP_SCALE, q4.UP_WEIGHT, q4.MOE_DIM, q4.DIM,
         Q3_UP_SCALE, Q3_UP_WEIGHT),
        (q4.DOWN_SCALE, q4.DOWN_WEIGHT, q4.DIM, q4.MOE_DIM,
         Q3_DOWN_SCALE, Q3_DOWN_WEIGHT),
    )
    for scale, weight, rows, columns, wanted_scale, wanted_weight in specs:
        if output.tell() - target_start != wanted_scale:
            raise AssertionError("Q3 scale offset drift")
        scales, weights = repack_matrix(
            source, source_base, scale, weight, rows, columns // 64
        )
        output.write(scales)
        if output.tell() - target_start != wanted_weight:
            raise AssertionError("Q3 weight offset drift")
        output.write(weights)
    if output.tell() - target_start != Q3_DATA_BYTES:
        raise AssertionError("Q3 record data size drift")
    output.write(bytes(Q3_RECORD_BYTES - Q3_DATA_BYTES))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source
    final = args.output
    expected_q4 = q4.HEADER_BYTES + RECORDS * Q4_RECORD_BYTES
    expected_q3 = q4.HEADER_BYTES + RECORDS * Q3_RECORD_BYTES
    q4.validate_expert_header(source_path, expected_q4)
    if source_path.stat().st_size != expected_q4:
        raise ValueError("Q4 expert source has an unexpected size")
    if final.exists():
        validate_q3_header(final, expected_q3)
        if final.stat().st_size != expected_q3:
            raise ValueError("existing Q3 expert output is incomplete")
        print(f"keeping existing {final} ({expected_q3:,} bytes)")
        return

    partial = final.with_suffix(final.suffix + ".partial")
    completed = 0
    if partial.exists():
        validate_q3_header(partial, expected_q3)
        data = partial.stat().st_size - q4.HEADER_BYTES
        completed, remainder = divmod(data, Q3_RECORD_BYTES)
        if remainder:
            attempt = 1
            while final.with_suffix(final.suffix + f".partial{attempt}").exists():
                attempt += 1
            partial = final.with_suffix(final.suffix + f".partial{attempt}")
            completed = 0
        else:
            print(f"resuming exact Q3 record boundary {completed}/{RECORDS}", flush=True)

    mode = "r+b" if completed else "xb"
    with source_path.open("rb") as source_file, \
            mmap.mmap(source_file.fileno(), 0, access=mmap.ACCESS_READ) as source, \
            partial.open(mode, buffering=16 * 1024 * 1024) as output:
        if completed == 0:
            output.write(q3_header(expected_q3))
            output.write(bytes(q4.HEADER_BYTES - q4.EXPERT_HEADER.size))
        output.seek(q4.HEADER_BYTES + completed * Q3_RECORD_BYTES)
        for record in range(completed, RECORDS):
            write_record(output, source, record)
            if (record + 1) % q4.EXPERTS == 0:
                output.flush()
                os.fsync(output.fileno())
                print(f"Q3 experts layer {(record + 1) // q4.EXPERTS}/{q4.LAYERS}", flush=True)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != expected_q3:
        raise AssertionError("Q3 final size drift")
    partial.rename(final)
    validate_q3_header(final, expected_q3)
    print(f"Q3 experts complete: {final} ({expected_q3:,} bytes)")


if __name__ == "__main__":
    main()
