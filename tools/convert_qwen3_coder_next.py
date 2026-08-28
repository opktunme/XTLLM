#!/usr/bin/env python3
"""Convert official Qwen3-Coder-Next-FP8 to the XTLLM Q4 Vulkan containers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
from typing import BinaryIO, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import convert_qwen38 as b

MODEL_ID = "Qwen/Qwen3-Coder-Next-FP8"
MODEL_REVISION = "da6e2ed27304dd39abadd9c82ef50e8de67bdd4c"
CHECKPOINT_BYTES = 80_362_904_064
TENSORS = 148_383
SHARDS = 40
DIM, MOE_DIM, LAYERS, EXPERTS, TOP_K = 2048, 512, 48, 512, 10
VOCAB, BASE_VOCAB, MERGES = 151936, 151643, 151387
Q_HEADS, KV_HEADS, HEAD_DIM, ROPE_DIM = 16, 2, 256, 64
LINEAR_KEY_HEADS, LINEAR_VALUE_HEADS, LINEAR_HEAD_DIM = 16, 32, 128
FULL_LAYERS = tuple(range(3, LAYERS, 4))
MAX_POSITION = 262144
MAIN_PARAMETERS, ACTIVE_PARAMETERS = 80_000_000_000, 3_300_000_000
MAGIC_SHARED, MAGIC_EXPERT = b"OQN3SHR\0", b"OQN3EXP\0"

GATE_SCALE, GATE_WEIGHT = 0, 32768
UP_SCALE, UP_WEIGHT = 557056, 589824
DOWN_SCALE, DOWN_WEIGHT = 1114112, 1146880
EXPERT_RECORD_BYTES = 1671168

Plan, Result = b.Plan, b.Result


def plans() -> List[Plan]:
    out = [
        Plan("embed", ("model.embed_tokens.weight",), b.Q8_ROW, -1, (VOCAB, DIM)),
        Plan("lm_head", ("lm_head.weight",), b.Q8_ROW, -1, (VOCAB, DIM)),
        Plan("final_norm", ("model.norm.weight",), b.F32, -1, (DIM,)),
    ]
    for layer in range(LAYERS):
        s, r = f"model.layers.{layer}", f"layers.{layer}"
        out += [
            Plan(f"{r}.input_norm", (f"{s}.input_layernorm.weight",), b.F32, layer, (DIM,)),
            Plan(f"{r}.post_norm", (f"{s}.post_attention_layernorm.weight",), b.F32, layer, (DIM,)),
        ]
        if layer in FULL_LAYERS:
            a = f"{s}.self_attn"
            out += [
                Plan(f"{r}.q_proj", (f"{a}.q_proj.weight",), b.Q4G64T, layer, (8192, DIM)),
                Plan(f"{r}.k_proj", (f"{a}.k_proj.weight",), b.Q4G64T, layer, (512, DIM)),
                Plan(f"{r}.v_proj", (f"{a}.v_proj.weight",), b.Q4G64T, layer, (512, DIM)),
                Plan(f"{r}.o_proj", (f"{a}.o_proj.weight",), b.Q4G64T, layer, (DIM, 4096)),
                Plan(f"{r}.q_norm", (f"{a}.q_norm.weight",), b.F32, layer, (HEAD_DIM,)),
                Plan(f"{r}.k_norm", (f"{a}.k_norm.weight",), b.F32, layer, (HEAD_DIM,)),
            ]
        else:
            a = f"{s}.linear_attn"
            qkvz = f"{a}.in_proj_qkvz.weight"
            ba = f"{a}.in_proj_ba.weight"
            out += [
                Plan(f"{r}.gdn_qkv", (qkvz,), b.Q4G64T, layer, (8192, DIM), "qkvz_qkv"),
                Plan(f"{r}.gdn_z", (qkvz,), b.Q4G64T, layer, (4096, DIM), "qkvz_z"),
                Plan(f"{r}.ab_proj", (ba,), b.Q4G64T, layer, (64, DIM), "ba_to_ab"),
                Plan(f"{r}.gdn_out", (f"{a}.out_proj.weight",), b.Q4G64T, layer, (DIM, 4096)),
                Plan(f"{r}.conv", (f"{a}.conv1d.weight",), b.F32, layer, (8192, 4), "conv"),
                Plan(f"{r}.delta_params", (f"{a}.A_log", f"{a}.dt_bias", f"{a}.norm.weight"),
                     b.F32, layer, (192,), "concat_vector"),
            ]
        m = f"{s}.mlp"
        out += [
            Plan(f"{r}.router", (f"{m}.gate.weight",), b.Q8_ROW, layer, (EXPERTS, DIM)),
            Plan(f"{r}.shared_gate_proj", (f"{m}.shared_expert.gate_proj.weight",), b.Q4G64T, layer, (MOE_DIM, DIM)),
            Plan(f"{r}.shared_up_proj", (f"{m}.shared_expert.up_proj.weight",), b.Q4G64T, layer, (MOE_DIM, DIM)),
            Plan(f"{r}.shared_down_proj", (f"{m}.shared_expert.down_proj.weight",), b.Q4G64T, layer, (DIM, MOE_DIM)),
            Plan(f"{r}.shared_expert_gate", (f"{m}.shared_expert_gate.weight",), b.Q4G64T, layer, (1, DIM)),
        ]
    assert len(out) == 627 and len({x.name for x in out}) == 627
    return out


def decoded(store: b.SafeStore, name: str) -> np.ndarray:
    info = store.info(name)
    if info.dtype in ("BF16", "F32"):
        return b.decode_array(store, name)
    if info.dtype == "F8_E4M3":
        return b.fp8_f32(store.array(name), b.bf16_f32(store.array(name + "_scale_inv")))
    raise ValueError(f"unsupported dtype {info.dtype}: {name}")


def direct_chunks(store: b.SafeStore, name: str, chunk: int = 128):
    info = store.info(name)
    rows, columns = info.shape
    if info.dtype == "F8_E4M3":
        return b.fp8_tensor_chunks(store, name, rows, columns, chunk)
    def iterate():
        value = store.array(name)
        for first in range(0, rows, chunk):
            yield first, b.bf16_f32(value[first:first + chunk])
    return iterate


def transformed_chunks(store: b.SafeStore, plan: Plan):
    name = plan.sources[0]
    if plan.transform == "direct":
        return direct_chunks(store, name)
    source = decoded(store, name)
    if plan.transform == "qkvz_qkv":
        pieces = []
        for offset, count in ((0, 128), (128, 128), (256, 256)):
            pieces.extend(source[h * 768 + offset:h * 768 + offset + count]
                          for h in range(16))
        value = np.concatenate(pieces, axis=0)
    elif plan.transform == "qkvz_z":
        value = np.concatenate([source[h * 768 + 512:h * 768 + 768]
                                for h in range(16)], axis=0)
    elif plan.transform == "ba_to_ab":
        # Checkpoint order is [b0,b1,a0,a1] per key head. Runtime wants all a,
        # followed by all b, matching the authoritative delta-rule equations.
        a = np.concatenate([source[h * 4 + 2:h * 4 + 4] for h in range(16)], axis=0)
        beta = np.concatenate([source[h * 4:h * 4 + 2] for h in range(16)], axis=0)
        value = np.concatenate((a, beta), axis=0)
    else:
        raise ValueError(plan.transform)
    if value.shape != plan.shape:
        raise ValueError(f"transform shape drift {plan.name}: {value.shape}")
    def iterate():
        for first in range(0, len(value), 128):
            yield first, value[first:first + 128]
    return iterate


def write_plan(output: BinaryIO, store: b.SafeStore, plan: Plan) -> Result:
    b.align_file(output, 64)
    if plan.format == b.F32:
        start = output.tell()
        if plan.transform == "concat_vector":
            for name in plan.sources:
                output.write(np.ascontiguousarray(decoded(store, name).reshape(-1), dtype="<f4").tobytes())
        else:
            value = decoded(store, plan.sources[0]).reshape(plan.shape)
            output.write(np.ascontiguousarray(value, dtype="<f4").tobytes())
        return Result(plan, start, b.product(plan.shape) * 4)
    if plan.format == b.Q8_ROW:
        return b.write_q8(output, store, plan)
    rows, columns = plan.shape
    weight, wb, scale, sb = b.pack_q4g64(output, transformed_chunks(store, plan), rows, columns,
                                         cache_decoded=plan.transform != "direct")
    return Result(plan, weight, wb, scale, sb)


def shared_header(file_bytes: int, tensor_count: int, data_offset: int, groups: int) -> bytes:
    return b.SHARED_HEADER.pack(
        MAGIC_SHARED, 1, b.HEADER_BYTES, 1, b.TENSOR_ENTRY.size,
        DIM, MOE_DIM, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM,
        LINEAR_KEY_HEADS, LINEAR_VALUE_HEADS, LINEAR_HEAD_DIM, ROPE_DIM,
        VOCAB, EXPERTS, TOP_K, 1,
        4, 4, 0, 0, LINEAR_KEY_HEADS, LINEAR_HEAD_DIM, LINEAR_VALUE_HEADS, MAX_POSITION,
        0, 0, b.UINT32_MAX, 151645, 151643, 151644, 151645, 151667, 151668, b.UINT32_MAX,
        1e-6, 1e-6, 1.0, 0.0, 5_000_000.0, 0.0,
        0.0, 0.0, 0.0, float(MAX_POSITION), 0.0, 0.0,
        b.SHARED_HEADER.size, groups, b.HEADER_BYTES, tensor_count, data_offset,
        file_bytes, EXPERT_RECORD_BYTES, LAYERS * EXPERTS,
        MAIN_PARAMETERS, ACTIVE_PARAMETERS)


def write_shared(output_dir: Path, store: b.SafeStore, all_plans: Sequence[Plan]) -> Path:
    final, partial = output_dir / "model-q4g64.ovs", output_dir / "model-q4g64.ovs.partial"
    if final.exists():
        print(f"shared: keeping {final}"); return final
    groups = [(0, -1, [x for x in all_plans if x.group < 0])]
    groups += [(1, layer, [x for x in all_plans if x.group == layer]) for layer in range(LAYERS)]
    data_offset = b.align_value(b.HEADER_BYTES + len(all_plans) * b.TENSOR_ENTRY.size, 4096)
    results, group_rows = [], []
    with partial.open("xb", buffering=16 << 20) as output:
        output.write(bytes(data_offset)); first = 0
        for kind, index, entries in groups:
            b.align_file(output, 4096); begin = output.tell()
            for plan in entries: results.append(write_plan(output, store, plan))
            b.align_file(output, 4096); end = output.tell()
            group_rows.append((kind, index, first, len(entries), begin, end, 0, 0)); first += len(entries)
            print(f"shared {'global' if index < 0 else f'layer {index+1}/{LAYERS}'}: {end-begin:,} bytes", flush=True)
        file_bytes = output.tell(); output.seek(b.HEADER_BYTES)
        for result in results: output.write(b.pack_tensor(result))
        output.seek(0); output.write(shared_header(file_bytes, len(results), data_offset, len(groups)))
        for row in group_rows: output.write(b.GROUP_ENTRY.pack(*row))
        output.flush(); os.fsync(output.fileno())
    partial.rename(final); print(f"shared: {final} ({file_bytes:,} bytes)"); return final


def expert_header(file_bytes: int) -> bytes:
    records = LAYERS * EXPERTS
    return b.EXPERT_HEADER.pack(MAGIC_EXPERT, 1, b.HEADER_BYTES, DIM, MOE_DIM, LAYERS, EXPERTS, 0,
        EXPERT_RECORD_BYTES, b.HEADER_BYTES, file_bytes, records, records, file_bytes,
        GATE_WEIGHT, GATE_SCALE, UP_WEIGHT, UP_SCALE, DOWN_WEIGHT, DOWN_SCALE, 0)


def write_experts(output_dir: Path, store: b.SafeStore, flush_records: int) -> Path:
    final, partial = output_dir / "experts-q4g64.ovx", output_dir / "experts-q4g64.ovx.partial"
    records = LAYERS * EXPERTS; expected = b.HEADER_BYTES + records * EXPERT_RECORD_BYTES
    if final.exists(): print(f"experts: keeping {final}"); return final
    completed = 0
    if partial.exists():
        completed, remainder = divmod(partial.stat().st_size - b.HEADER_BYTES, EXPERT_RECORD_BYTES)
        if remainder:
            with partial.open("r+b") as f: f.truncate(b.HEADER_BYTES + completed * EXPERT_RECORD_BYTES)
        print(f"experts: resuming at {completed}/{records}", flush=True)
    with partial.open("r+b" if partial.exists() else "xb", buffering=16 << 20) as output:
        if not completed: output.write(expert_header(expected)); output.write(bytes(b.HEADER_BYTES - b.EXPERT_HEADER.size))
        output.seek(b.HEADER_BYTES + completed * EXPERT_RECORD_BYTES)
        for record in range(completed, records):
            layer, expert = divmod(record, EXPERTS); start = output.tell()
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            specs = (("gate_proj", MOE_DIM, DIM, GATE_WEIGHT, GATE_SCALE),
                     ("up_proj", MOE_DIM, DIM, UP_WEIGHT, UP_SCALE),
                     ("down_proj", DIM, MOE_DIM, DOWN_WEIGHT, DOWN_SCALE))
            for suffix, rows, cols, want_w, want_s in specs:
                name = f"{prefix}.{suffix}.weight"
                w, wb, s, sb = b.pack_q4g64(output, b.fp8_tensor_chunks(store, name, rows, cols),
                                             rows, cols, cache_decoded=True)
                if w-start != want_w or s-start != want_s: raise AssertionError("expert record layout drift")
            if output.tell()-start != EXPERT_RECORD_BYTES: raise AssertionError("expert record size drift")
            if (record+1) % flush_records == 0 or expert == EXPERTS-1:
                output.flush(); os.fsync(output.fileno())
            if expert == EXPERTS-1: print(f"experts layer {layer+1}/{LAYERS}", flush=True)
    partial.rename(final); print(f"experts: {final} ({expected:,} bytes)"); return final


def write_tokenizer(metadata_dir: Path, output_dir: Path) -> Path:
    final = output_dir / "tokenizer.ovb"
    if final.exists(): print(f"tokenizer: keeping {final}"); return final
    tok = json.loads((metadata_dir / "tokenizer.json").read_text("utf-8"))
    cfg = json.loads((metadata_dir / "tokenizer_config.json").read_text("utf-8"))
    model, vocabulary = tok["model"], tok["model"]["vocab"]
    if len(vocabulary) != BASE_VOCAB or len(model["merges"]) != MERGES: raise ValueError("tokenizer counts changed")
    pieces: List[bytes|None] = [None] * VOCAB; flags = [0] * VOCAB; decoder = b.byte_decoder()
    for piece, token in vocabulary.items(): pieces[int(token)] = bytes(decoder[c] for c in piece)
    added = {int(x["id"]): x for x in tok["added_tokens"]}
    for key, value in cfg.get("added_tokens_decoder", {}).items():
        combined = dict(added.get(int(key), {})); combined.update(value); added[int(key)] = combined
    for token, entry in added.items(): pieces[token] = str(entry["content"]).encode(); flags[token] = 1 | (2 if entry.get("special") else 0)
    for token in range(VOCAB):
        if pieces[token] is None: pieces[token] = f"<|unused_{token}|>".encode(); flags[token] |= 4
    merges=[]
    for rank, encoded in enumerate(model["merges"]):
        left,right = encoded if isinstance(encoded,list) else encoded.split(" ",1)
        merges.append((int(vocabulary[left]),int(vocabulary[right]),int(vocabulary[left+right]),rank))
    metadata=json.dumps({"model_id":MODEL_ID,"revision":MODEL_REVISION,"chat_template":(metadata_dir/"chat_template.jinja").read_text("utf-8"),"generation_suffix":"<|im_start|>assistant\n"},separators=(",",":"),ensure_ascii=False).encode()
    token_table=b.TOKENIZER_HEADER.size; merge_table=b.align_value(token_table+VOCAB*b.TOKEN_ENTRY.size,64)
    pieces_offset=b.align_value(merge_table+MERGES*b.MERGE_ENTRY.size,64); pieces_bytes=sum(map(len,pieces))
    metadata_offset=b.align_value(pieces_offset+pieces_bytes,64); file_bytes=metadata_offset+len(metadata)
    header=b.TOKENIZER_HEADER.pack(b.MAGIC_TOKENIZER,2,b.TOKENIZER_HEADER.size,VOCAB,BASE_VOCAB,MERGES,
        b.UINT32_MAX,151645,151643,151643,151644,151645,151667,151668,b.UINT32_MAX,b.UINT32_MAX,b.UINT32_MAX,
        b.TOKEN_ENTRY.size,b.MERGE_ENTRY.size,len(added),0,token_table,merge_table,pieces_offset,pieces_bytes,metadata_offset,len(metadata),file_bytes,0)
    partial=output_dir/"tokenizer.ovb.partial"; cursor=pieces_offset
    with partial.open("xb",buffering=8<<20) as output:
        output.write(header)
        for piece,flag in zip(pieces,flags): output.write(b.TOKEN_ENTRY.pack(cursor,len(piece),flag)); cursor+=len(piece)
        b.align_file(output,64)
        for merge in merges: output.write(b.MERGE_ENTRY.pack(*merge))
        b.align_file(output,64)
        for piece in pieces: output.write(piece)
        b.align_file(output,64); output.write(metadata); output.flush(); os.fsync(output.fileno())
    partial.rename(final); print(f"tokenizer: {final} ({file_bytes:,} bytes)"); return final


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--metadata",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--inspect",action="store_true"); p.add_argument("--flush-records",type=int,default=64); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((a.metadata/"config.json").read_text())
    for key,want in {"model_type":"qwen3_next","hidden_size":DIM,"num_hidden_layers":LAYERS,"num_experts":EXPERTS,"num_experts_per_tok":TOP_K,"vocab_size":VOCAB}.items():
        if config.get(key)!=want: raise ValueError(f"config {key} changed: {config.get(key)}")
    store=b.SafeStore(a.source,CHECKPOINT_BYTES,TENSORS,SHARDS,75_000_000_000,"Qwen3-Coder-Next-FP8")
    all_plans=plans()
    for plan in all_plans:
        for name in plan.sources: store.info(name)
    print(f"validated {MODEL_ID}@{MODEL_REVISION}: {len(store.infos):,} tensors",flush=True)
    if a.inspect: return
    write_tokenizer(a.metadata,a.output); write_shared(a.output,store,all_plans); write_experts(a.output,store,a.flush_records)

if __name__ == "__main__": main()
