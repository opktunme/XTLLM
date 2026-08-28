#!/usr/bin/env python3
"""Convert official LongCat-Flash-Lite-Sparse BF16 weights for XTLLM Vulkan."""
from __future__ import annotations
import argparse, json, os, struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Sequence
import numpy as np
import convert_qwen38 as b

MODEL_ID="meituan-longcat/LongCat-Flash-Lite-Sparse"
MODEL_REVISION="35de83a18b5838ec1657edb3f885a59bbdd7888f"
CHECKPOINT_BYTES=138_288_833_280; TENSORS=11_220; SHARDS=26
DIM=3072; MOE_DIM=1024; DENSE_DIM=6144; LAYERS=14; EXPERTS=256; TOTAL_EXPERTS=384; TOP_K=12
VOCAB=131072; MERGES=130592; Q_HEADS=32; Q_LORA=1536; KV_LORA=512
QK_NOPE=128; ROPE_DIM=64; VALUE_DIM=128; MAX_POSITION=983_040
MAGIC_SHARED=b"OLCFSHR\0"; MAGIC_EXPERT=b"OLCFEXP\0"; MAGIC_OE=b"OLCFOE1\0"
MAIN_PARAMETERS=68_500_000_000; ACTIVE_PARAMETERS=3_000_000_000
GATE_SCALE,GATE_WEIGHT=0,98304; UP_SCALE,UP_WEIGHT=1671168,1769472
DOWN_SCALE,DOWN_WEIGHT=3342336,3440640; EXPERT_RECORD_BYTES=5013504
OE_TABLES=12; OE_DIM=256; OE_BASE_ROWS=78*VOCAB
OE_HEADER=struct.Struct("<8s8I40Q")
Plan,Result=b.Plan,b.Result

def plans(matrix_format=b.Q4G64T):
    out=[Plan("embed",("model.embed_tokens.weight",),b.Q8_ROW,-1,(VOCAB,DIM)),
         Plan("lm_head",("lm_head.weight",),b.Q8_ROW,-1,(VOCAB,DIM)),
         Plan("final_norm",("model.norm.weight",),b.F32,-1,(DIM,),"norm")]
    for i in range(OE_TABLES):
        out.append(Plan(f"oe_proj.{i}",(f"model.oe_embed_proj{i}.weight",),matrix_format,-1,(DIM,OE_DIM)))
    for layer in range(LAYERS):
        s,r=f"model.layers.{layer}",f"layers.{layer}"
        for sub in range(2):
            out += [
              Plan(f"{r}.input_norm.{sub}",(f"{s}.input_layernorm.{sub}.weight",),b.F32,layer,(DIM,),"norm"),
              Plan(f"{r}.post_norm.{sub}",(f"{s}.post_attention_layernorm.{sub}.weight",),b.F32,layer,(DIM,),"norm")]
            a=f"{s}.self_attn.{sub}"
            out += [
              Plan(f"{r}.attn.{sub}.q_a",(f"{a}.q_a_proj.weight",),matrix_format,layer,(Q_LORA,DIM)),
              Plan(f"{r}.attn.{sub}.q_a_norm",(f"{a}.q_a_layernorm.weight",),b.F32,layer,(Q_LORA,),"q_norm"),
              Plan(f"{r}.attn.{sub}.q_b",(f"{a}.q_b_proj.weight",),matrix_format,layer,(Q_HEADS*(QK_NOPE+ROPE_DIM),Q_LORA)),
              Plan(f"{r}.attn.{sub}.kv_a",(f"{a}.kv_a_proj_with_mqa.weight",),matrix_format,layer,(KV_LORA+ROPE_DIM,DIM)),
              Plan(f"{r}.attn.{sub}.kv_a_norm",(f"{a}.kv_a_layernorm.weight",),b.F32,layer,(KV_LORA,),"kv_norm"),
              Plan(f"{r}.attn.{sub}.kv_b",(f"{a}.kv_b_proj.weight",),matrix_format,layer,(Q_HEADS*(QK_NOPE+VALUE_DIM),KV_LORA)),
              Plan(f"{r}.attn.{sub}.o",(f"{a}.o_proj.weight",),matrix_format,layer,(DIM,Q_HEADS*VALUE_DIM))]
            m=f"{s}.mlps.{sub}"
            out += [Plan(f"{r}.dense.{sub}.gate",(f"{m}.gate_proj.weight",),matrix_format,layer,(DENSE_DIM,DIM)),
                    Plan(f"{r}.dense.{sub}.up",(f"{m}.up_proj.weight",),matrix_format,layer,(DENSE_DIM,DIM)),
                    Plan(f"{r}.dense.{sub}.down",(f"{m}.down_proj.weight",),matrix_format,layer,(DIM,DENSE_DIM))]
        out += [Plan(f"{r}.router",(f"{s}.mlp.router.classifier.weight",),b.Q8_ROW,layer,(TOTAL_EXPERTS,DIM)),
                Plan(f"{r}.router_bias",(f"{s}.mlp.router.e_score_correction_bias",),b.F32,layer,(TOTAL_EXPERTS,))]
    assert len(out)==379 and len({x.name for x in out})==379
    return out

def chunks(store,name):
    values=store.array(name); rows=values.shape[0]
    def it():
        for first in range(0,rows,128): yield first,b.bf16_f32(values[first:first+128])
    return it

def write_q8(output:BinaryIO,store:b.SafeStore,plan:Plan):
    """Write row-wise Q8 from either BF16 model matrices or LongCat's F32 router."""
    values=store.array(plan.sources[0]); dtype=store.info(plan.sources[0]).dtype
    rows,columns=plan.shape; scales=np.empty(rows,dtype="<f4"); data_offset=output.tell()
    for first in range(0,rows,32):
        source=values[first:first+32]
        decoded=b.bf16_f32(source) if dtype=="BF16" else np.asarray(source,dtype="<f4")
        maximum=np.max(np.abs(decoded),axis=1)
        scale=np.maximum(maximum/np.float32(127.0),np.float32(1e-30))
        scales[first:first+len(decoded)]=scale
        quantized=np.rint(decoded/scale[:,None]).astype(np.int16)
        output.write(np.clip(quantized,-127,127).astype(np.int8).tobytes())
    data_bytes=rows*columns
    if output.tell()-data_offset!=data_bytes: raise AssertionError("Q8 matrix size drift")
    b.align_file(output,64); auxiliary_offset=output.tell(); output.write(scales.tobytes())
    return Result(plan,data_offset,data_bytes,auxiliary_offset,scales.nbytes)

def write_plan(output:BinaryIO,store:b.SafeStore,plan:Plan):
    b.align_file(output,64)
    if plan.format==b.F32:
        start=output.tell(); value=b.decode_array(store,plan.sources[0])
        if plan.transform=="norm": value=value-1.0
        elif plan.transform=="q_norm": value=value*(DIM/Q_LORA)**0.5-1.0
        elif plan.transform=="kv_norm": value=value*(DIM/KV_LORA)**0.5-1.0
        output.write(np.ascontiguousarray(value,dtype="<f4").tobytes()); return Result(plan,start,b.product(plan.shape)*4)
    if plan.format==b.Q8_ROW: return write_q8(output,store,plan)
    rows,cols=plan.shape; w,wb,s,sb=b.pack_q4g64(output,chunks(store,plan.sources[0]),rows,cols)
    return Result(plan,w,wb,s,sb)

def write_shared(outdir,store,all_plans):
    final,partial=outdir/"model-q4g64.ovs",outdir/"model-q4g64.ovs.partial"
    if final.exists(): print(f"shared: keeping {final}"); return
    groups=[(0,-1,[x for x in all_plans if x.group<0])]+[(1,l,[x for x in all_plans if x.group==l]) for l in range(LAYERS)]
    data_offset=b.align_value(b.HEADER_BYTES+len(all_plans)*b.TENSOR_ENTRY.size,4096); results=[]; group_rows=[]
    with partial.open("xb",buffering=16<<20) as output:
        output.write(bytes(data_offset)); first=0
        for kind,index,entries in groups:
            b.align_file(output,4096); begin=output.tell()
            for p in entries: results.append(write_plan(output,store,p))
            b.align_file(output,4096); end=output.tell(); group_rows.append((kind,index,first,len(entries),begin,end,0,0)); first+=len(entries)
            print(f"shared {'global' if index<0 else f'layer {index+1}/{LAYERS}'}: {end-begin:,} bytes",flush=True)
        file_bytes=output.tell(); output.seek(b.HEADER_BYTES)
        for x in results: output.write(b.pack_tensor(x))
        header=b.SHARED_HEADER.pack(MAGIC_SHARED,1,b.HEADER_BYTES,1,b.TENSOR_ENTRY.size,
          DIM,MOE_DIM,LAYERS,Q_HEADS,1,QK_NOPE+ROPE_DIM,1,1,VALUE_DIM,ROPE_DIM,
          VOCAB,EXPERTS,TOP_K,1,2,4,0,0,Q_LORA,KV_LORA,TOTAL_EXPERTS,MAX_POSITION,0,0,
          1,2,3,47,48,36,37,0xffffffff,
          1e-5,1e-5,6.0,0.0,1_000_000.0,0.0,2.0**0.5,6.0**0.5,0.0,float(MAX_POSITION),0.0,0.0,
          b.SHARED_HEADER.size,len(groups),b.HEADER_BYTES,len(results),data_offset,file_bytes,EXPERT_RECORD_BYTES,LAYERS*EXPERTS,MAIN_PARAMETERS,ACTIVE_PARAMETERS)
        output.seek(0); output.write(header)
        for row in group_rows: output.write(b.GROUP_ENTRY.pack(*row))
        output.flush(); os.fsync(output.fileno())
    partial.rename(final); print(f"shared: {final} ({file_bytes:,} bytes)")

def expert_header(file_bytes):
    records=LAYERS*EXPERTS
    return b.EXPERT_HEADER.pack(MAGIC_EXPERT,1,b.HEADER_BYTES,DIM,MOE_DIM,LAYERS,EXPERTS,0,EXPERT_RECORD_BYTES,
      b.HEADER_BYTES,file_bytes,records,records,file_bytes,GATE_WEIGHT,GATE_SCALE,UP_WEIGHT,UP_SCALE,DOWN_WEIGHT,DOWN_SCALE,0)

def write_experts(outdir,store,flush_records):
    final,partial=outdir/"experts-q4g64.ovx",outdir/"experts-q4g64.ovx.partial"; records=LAYERS*EXPERTS; expected=b.HEADER_BYTES+records*EXPERT_RECORD_BYTES
    if final.exists(): print(f"experts: keeping {final}"); return
    completed=0
    if partial.exists():
        completed,rem=divmod(partial.stat().st_size-b.HEADER_BYTES,EXPERT_RECORD_BYTES)
        if rem:
            with partial.open("r+b") as f:f.truncate(b.HEADER_BYTES+completed*EXPERT_RECORD_BYTES)
        print(f"experts: resuming {completed}/{records}",flush=True)
    with partial.open("r+b" if partial.exists() else "xb",buffering=16<<20) as output:
        if not completed: output.write(expert_header(expected)); output.write(bytes(b.HEADER_BYTES-b.EXPERT_HEADER.size))
        output.seek(b.HEADER_BYTES+completed*EXPERT_RECORD_BYTES)
        for record in range(completed,records):
            layer,expert=divmod(record,EXPERTS); start=output.tell(); prefix=f"model.layers.{layer}.mlp.experts.{expert}"
            for suffix,rows,cols,ww,ws in (("gate_proj",MOE_DIM,DIM,GATE_WEIGHT,GATE_SCALE),("up_proj",MOE_DIM,DIM,UP_WEIGHT,UP_SCALE),("down_proj",DIM,MOE_DIM,DOWN_WEIGHT,DOWN_SCALE)):
                w,wb,s,sb=b.pack_q4g64(output,chunks(store,f"{prefix}.{suffix}.weight"),rows,cols)
                if w-start!=ww or s-start!=ws: raise AssertionError("expert layout drift")
            if output.tell()-start!=EXPERT_RECORD_BYTES: raise AssertionError("record size drift")
            if (record+1)%flush_records==0 or expert==EXPERTS-1: output.flush();os.fsync(output.fileno())
            if expert==EXPERTS-1:
                # Drop mmap residency between layers.  The official BF16 shards are
                # much larger than the derived store and otherwise needlessly crowd
                # the machine while conversion advances.
                for mapping in store.maps.values(): mapping.close()
                for source in store.files.values(): source.close()
                store.maps.clear(); store.files.clear()
                print(f"experts layer {layer+1}/{LAYERS}",flush=True)
    partial.rename(final); print(f"experts: {final} ({expected:,} bytes)")

def write_oe(outdir,store):
    final,partial=outdir/"ngram-bf16.ove",outdir/"ngram-bf16.ove.partial"
    if final.exists(): print(f"ngram: keeping {final}"); return
    offsets=[]; rows=[]
    with partial.open("xb",buffering=64<<20) as output:
        output.write(bytes(4096))
        for i in range(OE_TABLES):
            name=f"model.oe_embed_tokens{i}.weight"; info=store.info(name); expected_rows=OE_BASE_ROWS+2*i+1
            if info.dtype!="BF16" or info.shape!=(expected_rows,OE_DIM): raise ValueError(f"unexpected ngram table {i}: {info.dtype} {info.shape}")
            b.align_file(output,4096); offsets.append(output.tell()); rows.append(expected_rows)
            values=store.array(name)
            for first in range(0,expected_rows,8192): output.write(np.ascontiguousarray(values[first:first+8192],dtype="<u2").tobytes())
            del values
            for mapping in store.maps.values(): mapping.close()
            for source in store.files.values(): source.close()
            store.maps.clear(); store.files.clear()
            print(f"ngram table {i+1}/{OE_TABLES}",flush=True)
        file_bytes=output.tell(); q=list(offsets)+[0]*(20-len(offsets))+list(rows)+[0]*(20-len(rows))
        output.seek(0); output.write(OE_HEADER.pack(MAGIC_OE,1,4096,OE_TABLES,OE_DIM,VOCAB,OE_BASE_ROWS,512,0,*q)); output.flush();os.fsync(output.fileno())
    partial.rename(final); print(f"ngram: {final} ({file_bytes:,} bytes)")

def write_tokenizer(meta,outdir):
    final=outdir/"tokenizer.ovb"
    if final.exists(): print(f"tokenizer: keeping {final}"); return
    tok=json.loads((meta/"tokenizer.json").read_text("utf-8")); model=tok["model"]; vocab=model["vocab"]
    if len(vocab)!=VOCAB or len(model["merges"])!=MERGES: raise ValueError("tokenizer counts changed")
    decoder=b.byte_decoder(); pieces=[None]*VOCAB;flags=[0]*VOCAB
    for piece,token in vocab.items(): pieces[int(token)]=bytes(decoder[c] for c in piece)
    for x in tok.get("added_tokens",[]): pieces[int(x["id"])]=x["content"].encode(); flags[int(x["id"])]=1|(2 if x.get("special") else 0)
    if any(x is None for x in pieces): raise ValueError("LongCat tokenizer has sparse vocabulary")
    merges=[]
    for rank,x in enumerate(model["merges"]):
        l,r=x if isinstance(x,list) else x.split(" ",1); merges.append((int(vocab[l]),int(vocab[r]),int(vocab[l+r]),rank))
    metadata=json.dumps({"model_id":MODEL_ID,"revision":MODEL_REVISION,"prompt":"<longcat_user>{text}<longcat_assistant>"},separators=(",",":")).encode()
    tt=b.TOKENIZER_HEADER.size; mt=b.align_value(tt+VOCAB*b.TOKEN_ENTRY.size,64); po=b.align_value(mt+MERGES*b.MERGE_ENTRY.size,64); pb=sum(map(len,pieces)); mo=b.align_value(po+pb,64); fb=mo+len(metadata)
    header=b.TOKENIZER_HEADER.pack(b.MAGIC_TOKENIZER,2,b.TOKENIZER_HEADER.size,VOCAB,VOCAB,MERGES,1,2,3,3,47,48,36,37,0xffffffff,0xffffffff,0xffffffff,b.TOKEN_ENTRY.size,b.MERGE_ENTRY.size,224,0,tt,mt,po,pb,mo,len(metadata),fb,0)
    partial=outdir/"tokenizer.ovb.partial";cursor=po
    with partial.open("xb",buffering=8<<20) as output:
        output.write(header)
        for piece,flag in zip(pieces,flags):output.write(b.TOKEN_ENTRY.pack(cursor,len(piece),flag));cursor+=len(piece)
        b.align_file(output,64)
        for x in merges:output.write(b.MERGE_ENTRY.pack(*x))
        b.align_file(output,64)
        for x in pieces:output.write(x)
        b.align_file(output,64);output.write(metadata);output.flush();os.fsync(output.fileno())
    partial.rename(final);print(f"tokenizer: {final} ({fb:,} bytes)")

def main():
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--metadata",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--inspect",action="store_true");p.add_argument("--q8-shared",action="store_true");p.add_argument("--flush-records",type=int,default=32);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    store=b.SafeStore(a.source,CHECKPOINT_BYTES,TENSORS,SHARDS,65_000_000_000,"LongCat-Flash-Lite-Sparse"); all_plans=plans(b.Q8_ROW if a.q8_shared else b.Q4G64T)
    for plan in all_plans:
        for name in plan.sources:store.info(name)
    print(f"validated {MODEL_ID}@{MODEL_REVISION}: {len(store.infos):,} tensors",flush=True)
    if a.inspect:return
    write_tokenizer(a.metadata,a.output);write_shared(a.output,store,all_plans);write_experts(a.output,store,a.flush_records);write_oe(a.output,store)
if __name__=="__main__":main()
