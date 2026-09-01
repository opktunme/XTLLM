#!/usr/bin/env python3
"""Convert the official Qwen3.8-Flash-Next MTP block for the Vulkan runtime.

This converter is deliberately additive.  It never removes or truncates an
existing artifact: an interrupted non-record-aligned attempt is preserved and
the next attempt receives a numbered suffix.  Dense/shared MTP tensors use the
same Q4G64T/Q8/F32 ABIs as the main runtime.  Routed experts are written
directly as signed Q3G64T records from the official FP8 checkpoint.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import BinaryIO

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import convert_qwen38 as q


MAGIC_SHARED = b"OQ38MTP\0"
MAGIC_EXPERT = b"OQ38MEX\0"

Q3_RECORD_BYTES = 1_998_848
Q3_GATE_SCALE = 0
Q3_GATE_WEIGHT = 51_200
Q3_UP_SCALE = 665_600
Q3_UP_WEIGHT = 716_800
Q3_DOWN_SCALE = 1_331_200
Q3_DOWN_WEIGHT = 1_382_400
Q3_DATA_BYTES = 1_996_800


def plans() -> list[q.Plan]:
    p = "mtp.layers.0"
    result = [
        q.Plan("pre_embedding_norm", ("mtp.pre_fc_norm_embedding.weight",),
               q.F32, -1, (q.DIM,)),
        q.Plan("pre_hidden_norm", ("mtp.pre_fc_norm_hidden.weight",),
               q.F32, -1, (q.HC_DIM,)),
        q.Plan("fc_embedding", ("mtp.fc_embedding.weight",),
               q.Q4G64T, -1, (q.DIM, q.DIM)),
        q.Plan("fc_hidden", ("mtp.fc_hidden.weight",),
               q.Q4G64T, -1, (q.DIM, q.DIM)),
        q.Plan("final_hc_norm", ("mtp.hyper_connection_mixer.hc_norm.weight",),
               q.F32, -1, (q.HC_DIM,)),
        q.Plan("final_hc_down",
               ("mtp.hyper_connection_mixer.input_mix_weight_down.weight",),
               q.Q4G64T, -1, (q.HC_LOWRANK, q.HC_DIM)),
        q.Plan("final_hc_up",
               ("mtp.hyper_connection_mixer.input_mix_weight_up.weight",),
               q.Q4G64T, -1, (q.HC_DIM, q.HC_LOWRANK)),
    ]
    for block in ("attn", "mlp"):
        source = f"{p}.{block}_hyper_connection"
        result.extend([
            q.Plan(f"layers.0.{block}_hc_norm", (f"{source}.hc_norm.weight",),
                   q.F32, 0, (q.HC_DIM,)),
            q.Plan(f"layers.0.{block}_hc_down",
                   (f"{source}.input_mix_weight_down.weight",),
                   q.Q4G64T, 0, (q.HC_LOWRANK, q.HC_DIM)),
            q.Plan(f"layers.0.{block}_hc_up",
                   (f"{source}.input_mix_weight_up.weight",),
                   q.Q4G64T, 0, (q.HC_DIM, q.HC_LOWRANK)),
            q.Plan(f"layers.0.{block}_hc_inject",
                   (f"{source}.block_inject_weight.weight",),
                   q.Q4G64T, 0, (q.HC_COUNT, q.HC_DIM)),
        ])
    attention = f"{p}.self_attn"
    mlp = f"{p}.mlp"
    result.extend([
        q.Plan("layers.0.q_proj", (f"{attention}.q_proj.weight",),
               q.Q4G64T, 0, (12288, q.DIM)),
        q.Plan("layers.0.k_proj", (f"{attention}.k_proj.weight",),
               q.Q4G64T, 0, (512, q.DIM)),
        q.Plan("layers.0.v_proj", (f"{attention}.v_proj.weight",),
               q.Q4G64T, 0, (512, q.DIM)),
        q.Plan("layers.0.o_proj", (f"{attention}.o_proj.weight",),
               q.Q4G64T, 0, (q.DIM, 6144)),
        q.Plan("layers.0.q_norm", (f"{attention}.q_norm.weight",),
               q.F32, 0, (q.HEAD_DIM,)),
        q.Plan("layers.0.k_norm", (f"{attention}.k_norm.weight",),
               q.F32, 0, (q.HEAD_DIM,)),
        q.Plan("layers.0.router", (f"{mlp}.gate.weight",),
               q.Q8_ROW, 0, (q.EXPERTS, q.DIM)),
        q.Plan("layers.0.shared_gate_proj",
               (f"{mlp}.shared_expert.gate_proj.weight",),
               q.Q4G64T, 0, (q.MOE_DIM, q.DIM)),
        q.Plan("layers.0.shared_up_proj",
               (f"{mlp}.shared_expert.up_proj.weight",),
               q.Q4G64T, 0, (q.MOE_DIM, q.DIM)),
        q.Plan("layers.0.shared_down_proj",
               (f"{mlp}.shared_expert.down_proj.weight",),
               q.Q4G64T, 0, (q.DIM, q.MOE_DIM)),
        q.Plan("layers.0.shared_expert_gate",
               (f"{mlp}.shared_expert_gate.weight",),
               q.Q4G64T, 0, (1, q.DIM)),
    ])
    if len(result) != 26 or len({x.name for x in result}) != len(result):
        raise AssertionError("MTP shared plan drift")
    return result


def next_attempt(path: Path) -> Path:
    if not path.exists():
        return path
    attempt = 2
    while Path(str(path) + str(attempt)).exists():
        attempt += 1
    return Path(str(path) + str(attempt))


def validate_shared(path: Path) -> None:
    if path.stat().st_size < q.HEADER_BYTES:
        raise ValueError(f"truncated MTP shared container: {path}")
    with path.open("rb") as source:
        fields = q.SHARED_HEADER.unpack(source.read(q.SHARED_HEADER.size))
    qwords = fields[-10:]
    if (fields[0] != MAGIC_SHARED or fields[1] != q.VERSION or
            fields[2] != q.HEADER_BYTES or fields[4] != q.TENSOR_ENTRY.size or
            fields[7] != 1 or qwords[1] != 2 or qwords[3] != 26 or
            qwords[5] != path.stat().st_size):
        raise ValueError(f"incompatible MTP shared container: {path}")


def write_shared(outdir: Path, store: q.SafeStore,
                 ps: list[q.Plan]) -> Path:
    final = outdir / "mtp-q4g64.ovs"
    if final.exists():
        validate_shared(final)
        print(f"MTP shared: keeping existing {final}", flush=True)
        return final
    partial = next_attempt(outdir / "mtp-q4g64.ovs.partial")
    if partial.name != "mtp-q4g64.ovs.partial":
        print(f"MTP shared: preserving earlier partial; using {partial}", flush=True)
    groups = [
        (0, -1, [x for x in ps if x.group < 0]),
        (2, 0, [x for x in ps if x.group == 0]),
    ]
    table_offset = q.HEADER_BYTES
    data_offset = q.align_value(
        table_offset + len(ps) * q.TENSOR_ENTRY.size, 4096)
    results: list[q.Result] = []
    group_rows = []
    with partial.open("xb", buffering=16 * 1024 * 1024) as output:
        output.write(bytes(data_offset))
        first = 0
        for kind, index, group_plans in groups:
            q.align_file(output, 4096)
            begin = output.tell()
            for plan in group_plans:
                results.append(q.write_plan(output, store, plan))
                print(f"MTP shared tensor: {plan.name}", flush=True)
            q.align_file(output, 4096)
            end = output.tell()
            group_rows.append(
                (kind, index, first, len(group_plans), begin, end, 0, 0))
            first += len(group_plans)
        file_bytes = output.tell()
        output.seek(table_offset)
        for result in results:
            output.write(q.pack_tensor(result))
        ints = [
            q.DIM, q.MOE_DIM, 1, q.Q_HEADS, q.KV_HEADS, q.HEAD_DIM,
            q.LINEAR_QK_HEADS, q.LINEAR_V_HEADS, q.LINEAR_HEAD_DIM,
            q.ROPE_DIM, q.VOCAB, q.EXPERTS, q.TOP_K, 1,
            q.HC_COUNT, q.HC_LOWRANK, 1, 2048,
            q.LINEAR_QK_HEADS, q.LINEAR_HEAD_DIM, q.LINEAR_V_HEADS,
            q.MAX_POSITION, 0, 0, q.UINT32_MAX, 248046, 248044, 248045,
            248046, 248068, 248069, q.UINT32_MAX,
        ]
        floats = [1e-6, 1e-6, 1.0, 0.0, 10_000_000.0, 0.0,
                  0.0, 0.0, 0.0, float(q.MAX_POSITION), 0.0, 0.0]
        qwords = [
            q.SHARED_HEADER.size, len(groups), table_offset, len(results),
            data_offset, file_bytes, q.EXPERT_RECORD_BYTES, q.EXPERTS,
            4_000_000_000, 0,
        ]
        output.seek(0)
        output.write(q.SHARED_HEADER.pack(
            MAGIC_SHARED, q.VERSION, q.HEADER_BYTES, 1,
            q.TENSOR_ENTRY.size, *ints, *floats, *qwords))
        for row in group_rows:
            output.write(q.GROUP_ENTRY.pack(*row))
        output.flush()
        os.fsync(output.fileno())
    partial.rename(final)
    validate_shared(final)
    print(f"MTP shared complete: {final} ({file_bytes:,} bytes)", flush=True)
    return final


def pack_q3_matrix(output: BinaryIO, store: q.SafeStore, name: str,
                   rows: int, columns: int) -> tuple[int, int, int, int]:
    info = store.info(name)
    scale_name = name + "_scale_inv"
    scale_info = store.info(scale_name)
    if (info.dtype != "F8_E4M3" or info.shape != (rows, columns) or
            scale_info.dtype != "BF16" or
            scale_info.shape != ((rows + 127) // 128,
                                 (columns + 127) // 128)):
        raise ValueError(f"unexpected MTP FP8 expert tensor: {name}")
    values = store.array(name)
    inverse_scales = q.bf16_f32(store.array(scale_name))
    groups = columns // 64
    decoded_chunks: list[np.ndarray] = []
    maxima = np.empty((rows, groups), dtype=np.float32)
    for first in range(0, rows, 128):
        count = min(128, rows - first)
        block_first = first // 128
        decoded = q.fp8_f32(
            values[first:first + count],
            inverse_scales[block_first:block_first + 1],
        ).reshape(count, groups, 64)
        decoded_chunks.append(decoded)
        maxima[first:first + count] = np.max(np.abs(decoded), axis=2)
    scales = maxima / np.float32(3.0)
    scale_offset = output.tell()
    output.write(q.f32_bf16(scales).tobytes())
    scale_bytes = rows * groups * 2
    weight_offset = output.tell()
    for first, decoded in zip(range(0, rows, 128), decoded_chunks):
        count = len(decoded)
        denominator = np.maximum(
            scales[first:first + count, :, None], np.float32(1e-30))
        quantized = np.rint(decoded / denominator).astype(np.int8)
        np.clip(quantized, -3, 3, out=quantized)
        codes = (quantized.astype(np.int16) & 7).astype(np.uint32)
        codes = codes.reshape(count, groups, 8, 8)
        chunks = np.zeros((count, 8, groups), dtype=np.uint32)
        for nibble in range(8):
            chunks |= ((codes[:, :, :, nibble].transpose(0, 2, 1))
                       << np.uint32(3 * nibble))
        packed = np.empty((count, 6, groups), dtype="<u4")
        c0, c1, c2, c3, c4, c5, c6, c7 = (
            chunks[:, index, :] for index in range(8))
        packed[:, 0, :] = c0 | (c1 << np.uint32(24))
        packed[:, 1, :] = (c1 >> np.uint32(8)) | (c2 << np.uint32(16))
        packed[:, 2, :] = (c2 >> np.uint32(16)) | (c3 << np.uint32(8))
        packed[:, 3, :] = c4 | (c5 << np.uint32(24))
        packed[:, 4, :] = (c5 >> np.uint32(8)) | (c6 << np.uint32(16))
        packed[:, 5, :] = (c6 >> np.uint32(16)) | (c7 << np.uint32(8))
        output.write(np.ascontiguousarray(packed).tobytes())
    weight_bytes = rows * columns * 3 // 8
    if output.tell() - weight_offset != weight_bytes:
        raise AssertionError(f"MTP Q3 packed size drift: {name}")
    return weight_offset, weight_bytes, scale_offset, scale_bytes


def expert_header(file_bytes: int) -> bytes:
    return q.EXPERT_HEADER.pack(
        MAGIC_EXPERT, q.VERSION, q.HEADER_BYTES, q.DIM, q.MOE_DIM,
        1, q.EXPERTS, 0, Q3_RECORD_BYTES,
        q.HEADER_BYTES, file_bytes, q.EXPERTS, q.EXPERTS, file_bytes,
        Q3_GATE_WEIGHT, Q3_GATE_SCALE, Q3_UP_WEIGHT, Q3_UP_SCALE,
        Q3_DOWN_WEIGHT, Q3_DOWN_SCALE, 0,
    )


def validate_experts(path: Path, expected: int) -> None:
    if path.stat().st_size != expected:
        raise ValueError(f"incomplete MTP Q3 expert container: {path}")
    with path.open("rb") as source:
        fields = q.EXPERT_HEADER.unpack(source.read(q.EXPERT_HEADER.size))
    if fields != q.EXPERT_HEADER.unpack(expert_header(expected)):
        raise ValueError(f"incompatible MTP Q3 expert container: {path}")


def write_experts(outdir: Path, store: q.SafeStore) -> Path:
    final = outdir / "mtp-experts-q3g64.ovx"
    expected = q.HEADER_BYTES + q.EXPERTS * Q3_RECORD_BYTES
    if final.exists():
        validate_experts(final, expected)
        print(f"MTP Q3 experts: keeping existing {final}", flush=True)
        return final
    partial = outdir / "mtp-experts-q3g64.ovx.partial"
    completed = 0
    if partial.exists():
        data = partial.stat().st_size - q.HEADER_BYTES
        if data >= 0:
            completed, remainder = divmod(data, Q3_RECORD_BYTES)
        else:
            remainder = 1
        if remainder:
            preserved = partial
            partial = next_attempt(partial)
            completed = 0
            print(f"MTP Q3 experts: preserving interrupted {preserved}; "
                  f"using {partial}", flush=True)
        else:
            with partial.open("rb") as source:
                fields = q.EXPERT_HEADER.unpack(
                    source.read(q.EXPERT_HEADER.size))
            if fields != q.EXPERT_HEADER.unpack(expert_header(expected)):
                raise ValueError(f"incompatible MTP Q3 partial: {partial}")
            print(f"MTP Q3 experts: resuming {completed}/{q.EXPERTS}",
                  flush=True)
    mode = "r+b" if completed else "xb"
    with partial.open(mode, buffering=16 * 1024 * 1024) as output:
        if completed == 0:
            output.write(expert_header(expected))
            output.write(bytes(q.HEADER_BYTES - q.EXPERT_HEADER.size))
        output.seek(q.HEADER_BYTES + completed * Q3_RECORD_BYTES)
        for expert in range(completed, q.EXPERTS):
            prefix = f"mtp.layers.0.mlp.experts.{expert}"
            start = output.tell()
            specs = [
                (f"{prefix}.gate_proj.weight", q.MOE_DIM, q.DIM,
                 Q3_GATE_WEIGHT, Q3_GATE_SCALE),
                (f"{prefix}.up_proj.weight", q.MOE_DIM, q.DIM,
                 Q3_UP_WEIGHT, Q3_UP_SCALE),
                (f"{prefix}.down_proj.weight", q.DIM, q.MOE_DIM,
                 Q3_DOWN_WEIGHT, Q3_DOWN_SCALE),
            ]
            for name, rows, columns, wanted_weight, wanted_scale in specs:
                weight, _, scale, _ = pack_q3_matrix(
                    output, store, name, rows, columns)
                if (weight - start != wanted_weight or
                        scale - start != wanted_scale):
                    raise AssertionError(
                        f"MTP Q3 record offset drift: expert={expert} {name}")
            if output.tell() - start != Q3_DATA_BYTES:
                raise AssertionError("MTP Q3 record data size drift")
            output.write(bytes(Q3_RECORD_BYTES - Q3_DATA_BYTES))
            if (expert + 1) % 16 == 0:
                output.flush()
                os.fsync(output.fileno())
                print(f"MTP Q3 experts {expert + 1}/{q.EXPERTS}", flush=True)
        output.flush()
        os.fsync(output.fileno())
    if partial.stat().st_size != expected:
        raise AssertionError("MTP Q3 final size drift")
    partial.rename(final)
    validate_experts(final, expected)
    print(f"MTP Q3 experts complete: {final} ({expected:,} bytes)",
          flush=True)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    store = q.SafeStore(args.source)
    ps = plans()
    for plan in ps:
        q.validate_plan(store, plan)
    write_shared(args.output, store, ps)
    write_experts(args.output, store)


if __name__ == "__main__":
    main()
