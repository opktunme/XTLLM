#!/usr/bin/env python3
"""Build a LongCat shared container with Q4 dense FFNs and Q8 attention.

This is additive: it writes only a new output container and never modifies the
official checkpoint or an existing completed runtime file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import convert_longcat_flash_lite as lc
import convert_qwen38 as b


def hybrid_plans(attention_output_q4: bool):
    result = []
    for plan in lc.plans(b.Q8_ROW):
        q4 = ".dense." in plan.name
        q4 = q4 or (attention_output_q4 and ".attn." in plan.name and
                    plan.name.endswith(".o"))
        if q4:
            plan = b.Plan(plan.name, plan.sources, b.Q4G64T, plan.group,
                          plan.shape, plan.transform)
        result.append(plan)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--filename", default="model-hybrid-q4g64.ovs")
    parser.add_argument("--attention-output-q4", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    store = b.SafeStore(
        args.source, lc.CHECKPOINT_BYTES, lc.TENSORS, lc.SHARDS,
        65_000_000_000, "LongCat-Flash-Lite-Sparse")
    plans = hybrid_plans(args.attention_output_q4)
    for plan in plans:
        for name in plan.sources:
            store.info(name)
    q4 = sum(1 for plan in plans if plan.format == b.Q4G64T)
    print(f"validated hybrid plan: {q4} Q4 tensors, "
          f"{len(plans) - q4} retained Q8/F32 tensors", flush=True)
    lc.write_shared(args.output, store, plans, args.filename)


if __name__ == "__main__":
    main()
