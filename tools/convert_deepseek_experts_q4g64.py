#!/usr/bin/env python3
"""Requantize DeepSeek-V4 native E2M1 experts to signed Q4G64T.

The alternate container is deliberately record-for-record and byte-for-byte
the same size as experts.ovx.  Existing outputs are never overwritten and a
full conversion resumes only after a complete 13,369,344-byte expert record.
Without --full this tool only converts explicitly requested probe records.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Dict, Iterable, Tuple

import numpy as np


HEADER_BYTES = 4096
DIMENSION = 4096
MOE_DIMENSION = 2048
LAYERS = 43
EXPERTS = 256
RECORDS = LAYERS * EXPERTS
Q4_GROUP = 64
Q4_FORMAT_TAG = 102
EXPERT_HEADER = struct.Struct("<8s8I12Q")
NATIVE_MAGIC = b"OVD4EXP\0"
Q4_MAGIC = b"OVD4Q4T\0"

WEIGHT_BYTES = 4 * 1024 * 1024
SCALE_BYTES = 256 * 1024
RECORD_BYTES = 13_369_344
W1_WEIGHT = 0
W1_SCALE = W1_WEIGHT + WEIGHT_BYTES
W3_WEIGHT = W1_SCALE + SCALE_BYTES
W3_SCALE = W3_WEIGHT + WEIGHT_BYTES
W2_WEIGHT = W3_SCALE + SCALE_BYTES
W2_SCALE = W2_WEIGHT + WEIGHT_BYTES
assert W2_SCALE + SCALE_BYTES == RECORD_BYTES

E2M1 = np.array(
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)


def f32_bf16(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(values, dtype="<f4").view("<u4")
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return np.ascontiguousarray(rounded >> np.uint32(16), dtype="<u2")


def bf16_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype="<u2").astype(np.uint32) << 16).view("<f4")


def ue8m0_f32(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.uint8).astype(np.uint32) << 23).view("<f4")


class Metrics:
    def __init__(self) -> None:
        self.source2 = 0.0
        self.error2 = 0.0
        self.dot = 0.0
        self.reconstructed2 = 0.0
        self.maximum_error = 0.0
        self.values = 0

    def add(self, source: np.ndarray, reconstructed: np.ndarray) -> None:
        src = np.asarray(source, dtype=np.float32)
        rec = np.asarray(reconstructed, dtype=np.float32)
        error = rec - src
        self.source2 += float(np.sum(src.astype(np.float64) ** 2))
        self.error2 += float(np.sum(error.astype(np.float64) ** 2))
        self.dot += float(np.sum(src.astype(np.float64) * rec.astype(np.float64)))
        self.reconstructed2 += float(np.sum(rec.astype(np.float64) ** 2))
        self.maximum_error = max(self.maximum_error, float(np.max(np.abs(error))))
        self.values += src.size

    def report(self) -> Dict[str, float]:
        relative_rms = math.sqrt(self.error2 / max(self.source2, 1e-300))
        cosine = self.dot / math.sqrt(max(self.source2 * self.reconstructed2, 1e-300))
        return {
            "relative_rms": relative_rms,
            "cosine": cosine,
            "maximum_error": self.maximum_error,
            "values": float(self.values),
        }


class NativeContainer:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("rb")
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.mapping) < HEADER_BYTES:
            raise ValueError(f"truncated native expert container: {path}")
        self.header = list(EXPERT_HEADER.unpack_from(self.mapping, 0))
        expected = HEADER_BYTES + RECORDS * RECORD_BYTES
        if (
            self.header[0] != NATIVE_MAGIC
            or self.header[1] != 1
            or self.header[2] != HEADER_BYTES
            or self.header[3] != DIMENSION
            or self.header[4] != MOE_DIMENSION
            or self.header[5] != LAYERS
            or self.header[6] != EXPERTS
            or self.header[8] != RECORD_BYTES
            or self.header[9] != HEADER_BYTES
            or self.header[11] != RECORDS
            or self.header[13] != expected
            or len(self.mapping) != expected
        ):
            raise ValueError(f"unsupported native expert container: {path}")

    def projection(self, record: int, weight_offset: int, scale_offset: int,
                   rows: int, columns: int) -> Tuple[np.ndarray, np.ndarray]:
        base = HEADER_BYTES + record * RECORD_BYTES
        weights = np.ndarray(
            (rows, columns // 2), dtype=np.uint8, buffer=self.mapping,
            offset=base + weight_offset,
        )
        scales = np.ndarray(
            (rows, columns // 32), dtype=np.uint8, buffer=self.mapping,
            offset=base + scale_offset,
        )
        return weights, scales

    def q4_header(self) -> bytes:
        fields = self.header.copy()
        fields[0] = Q4_MAGIC
        fields[-1] = Q4_FORMAT_TAG
        encoded = EXPERT_HEADER.pack(*fields)
        return encoded + bytes(HEADER_BYTES - len(encoded))

    def close(self) -> None:
        self.mapping.close()
        self.file.close()


def convert_projection(container: NativeContainer, record_index: int,
                       weight_offset: int, scale_offset: int,
                       rows: int, columns: int, destination: bytearray,
                       metrics: Metrics | None, chunk_rows: int = 32) -> None:
    native_weights, native_scales = container.projection(
        record_index, weight_offset, scale_offset, rows, columns)
    groups = columns // Q4_GROUP
    row_weight_bytes = columns // 2
    row_scale_bytes = groups * 2
    for first in range(0, rows, chunk_rows):
        last = min(rows, first + chunk_rows)
        packed = np.asarray(native_weights[first:last], dtype=np.uint8)
        values = np.empty((last - first, columns), dtype=np.float32)
        values[:, 0::2] = E2M1[packed & 15]
        values[:, 1::2] = E2M1[packed >> 4]
        scales32 = ue8m0_f32(native_scales[first:last])
        values *= np.repeat(scales32, 32, axis=1)

        grouped = values.reshape(last - first, groups, Q4_GROUP)
        scale = np.max(np.abs(grouped), axis=2) / np.float32(7.0)
        encoded_scale = f32_bf16(scale)
        actual_scale = bf16_f32(encoded_scale)
        quantized = np.rint(grouped / np.maximum(scale[:, :, None], np.float32(1e-30)))
        quantized = np.clip(quantized, -7, 7).astype(np.int8)
        if metrics is not None:
            reconstructed = quantized.astype(np.float32) * actual_scale[:, :, None]
            metrics.add(grouped, reconstructed)

        nibbles = quantized.view(np.uint8).reshape(last - first, groups, 8, 8)
        words = np.zeros((last - first, groups, 8), dtype="<u4")
        for nibble in range(8):
            words |= ((nibbles[:, :, :, nibble] & 15).astype(np.uint32)
                      << np.uint32(nibble * 4))
        transposed = np.ascontiguousarray(words.transpose(0, 2, 1))
        weight_begin = weight_offset + first * row_weight_bytes
        weight_end = weight_offset + last * row_weight_bytes
        destination[weight_begin:weight_end] = transposed.tobytes()
        scale_begin = scale_offset + first * row_scale_bytes
        scale_end = scale_offset + last * row_scale_bytes
        destination[scale_begin:scale_end] = encoded_scale.tobytes()


def convert_record(container: NativeContainer, record_index: int,
                   collect_metrics: bool = True) -> Tuple[bytes, Dict[str, Dict[str, float]]]:
    if record_index < 0 or record_index >= RECORDS:
        raise ValueError(f"record must be in [0,{RECORDS - 1}]: {record_index}")
    record = bytearray(RECORD_BYTES)
    reports: Dict[str, Dict[str, float]] = {}
    for name, weight, scale, rows, columns in (
        ("w1", W1_WEIGHT, W1_SCALE, MOE_DIMENSION, DIMENSION),
        ("w3", W3_WEIGHT, W3_SCALE, MOE_DIMENSION, DIMENSION),
        ("w2", W2_WEIGHT, W2_SCALE, DIMENSION, MOE_DIMENSION),
    ):
        metric = Metrics() if collect_metrics else None
        convert_projection(container, record_index, weight, scale, rows, columns,
                           record, metric)
        if metric is not None:
            reports[name] = metric.report()
    return bytes(record), reports


_WORKER_CONTAINER: NativeContainer | None = None


def initialize_worker(path: str) -> None:
    global _WORKER_CONTAINER
    _WORKER_CONTAINER = NativeContainer(Path(path))


def convert_worker(record_index: int) -> Tuple[int, bytes]:
    if _WORKER_CONTAINER is None:
        raise RuntimeError("Q4G64T worker was not initialized")
    encoded, _ = convert_record(_WORKER_CONTAINER, record_index, False)
    return record_index, encoded


def validate_q4_file(path: Path, expected: int) -> bool:
    if not path.is_file() or path.stat().st_size != expected:
        return False
    with path.open("rb") as source:
        fields = EXPERT_HEADER.unpack(source.read(EXPERT_HEADER.size))
    return fields[0] == Q4_MAGIC and fields[-1] == Q4_FORMAT_TAG and fields[13] == expected


def probe(container: NativeContainer, records: Iterable[int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    elapsed = 0.0
    completed = 0
    for record_index in records:
        destination = directory / f"record-{record_index:05d}.q4g64t"
        if destination.exists():
            raise FileExistsError(f"preserving existing probe: {destination}")
        start = time.perf_counter()
        encoded, reports = convert_record(container, record_index)
        with destination.open("xb", buffering=16 * 1024 * 1024) as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        seconds = time.perf_counter() - start
        elapsed += seconds
        completed += 1
        layer, expert = divmod(record_index, EXPERTS)
        print(f"probe record {record_index} layer={layer} expert={expert}: {seconds:.3f} s")
        for name, values in reports.items():
            print(f"  {name}: rel_rms={values['relative_rms']:.6f} "
                  f"cosine={values['cosine']:.8f} max_error={values['maximum_error']:.6g}")
    if completed:
        per_record = elapsed / completed
        print(f"probe average: {per_record:.3f} s/record")
        print(f"estimated full conversion: {per_record * RECORDS / 3600.0:.2f} h")
        print(f"full output: {HEADER_BYTES + RECORDS * RECORD_BYTES:,} bytes "
              f"({(HEADER_BYTES + RECORDS * RECORD_BYTES) / 2**30:.3f} GiB)")


def full_conversion(container: NativeContainer, destination: Path, workers: int) -> None:
    expected = HEADER_BYTES + RECORDS * RECORD_BYTES
    if destination.exists():
        if not validate_q4_file(destination, expected):
            raise ValueError(f"existing output is not complete Q4G64T: {destination}")
        print(f"keeping complete output: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    if partial.exists():
        size = partial.stat().st_size
        if size < HEADER_BYTES or (size - HEADER_BYTES) % RECORD_BYTES:
            raise ValueError(f"partial is not at a record boundary: {partial}")
        with partial.open("rb") as source:
            fields = EXPERT_HEADER.unpack(source.read(EXPERT_HEADER.size))
        if fields[0] != Q4_MAGIC or fields[-1] != Q4_FORMAT_TAG or fields[13] != expected:
            raise ValueError(f"partial has incompatible Q4G64T header: {partial}")
        completed = (size - HEADER_BYTES) // RECORD_BYTES
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("xb") as output:
            output.write(container.q4_header())
            output.flush()
            os.fsync(output.fileno())
        completed = 0
    print(f"resuming Q4G64T experts at {completed:,}/{RECORDS:,}")
    with partial.open("ab", buffering=16 * 1024 * 1024) as output:
        if workers == 1:
            records = ((index, convert_record(container, index, False)[0])
                       for index in range(completed, RECORDS))
            executor = None
        else:
            executor = ProcessPoolExecutor(max_workers=workers,
                initializer=initialize_worker, initargs=(str(container.path),))
            records = executor.map(convert_worker, range(completed, RECORDS), chunksize=1)
        try:
            for record_index, encoded in records:
                output.write(encoded)
                if (record_index + 1) % EXPERTS == 0:
                    output.flush()
                    os.fsync(output.fileno())
                    print(f"Q4G64T experts layer {(record_index + 1) // EXPERTS}/{LAYERS}",
                          flush=True)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    if partial.stat().st_size != expected:
        raise AssertionError("Q4G64T expert output size drift")
    partial.rename(destination)
    print(f"Q4G64T experts: {destination} ({expected:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--probe-dir", type=Path)
    parser.add_argument("--probe-record", action="append", type=int, default=[])
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.workers > 16:
        parser.error("--workers must be in [1,16]")
    if not arguments.full and not arguments.probe_record:
        parser.error("select --probe-record at least once, or explicitly pass --full")
    if arguments.probe_record and arguments.probe_dir is None:
        parser.error("--probe-record requires --probe-dir")
    container = NativeContainer(arguments.source)
    try:
        if arguments.probe_record:
            probe(container, arguments.probe_record, arguments.probe_dir)
        if arguments.full:
            full_conversion(container, arguments.output, arguments.workers)
    finally:
        container.close()


if __name__ == "__main__":
    main()
