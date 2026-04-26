"""Build a key-correct merged checkpoint for LlamaFactory DPO.

Background
----------
SFT was run with ``AutoModelForCausalLM`` against the multimodal base
``/models/models/Qwen3.5-9B``. That auto-class loads only the language-model
submodule, so the SFT shards contain keys like::

    model.embed_tokens.weight
    model.layers.0.input_layernorm.weight
    model.norm.weight
    lm_head.weight

When LlamaFactory loads the model with ``Qwen3_5ForConditionalGeneration``
(the full multimodal class), it expects::

    model.visual.<...>            # vision tower
    model.language_model.<...>    # language model with extra prefix

The previous merge.py in saves/Qwen3.5-9B/SFT/ produced a checkpoint where
the LLM keys got remapped (the loader successfully loads e.g.
``model.language_model.norm.weight``) but the visual keys ended up missing
or under the wrong prefix, hence the MISSING report and randomly initialised
vision tower (which both wastes memory and breaks deployment).

This script
-----------
1. Reads the visual shard layout straight from the base model. The base
   shards already use the correct ``model.visual.<...>`` keys.
2. Reads the SFT shards (single-shard or sharded; both supported) and
   rewrites every key to its multimodal name:
       model.X            -> model.language_model.X
       lm_head.X          -> language_model.lm_head.X
   keys that already start with ``model.language_model.`` or ``model.visual.``
   are left alone.
3. Drops every visual tensor from the SFT side (we trust the base copies).
4. Emits a single ``model.safetensors`` containing both halves, plus all
   non-weight artifacts (config, processor, tokenizer, chat template) copied
   from the base model so the multimodal config is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file
import torch


_VISUAL_HINTS = ("visual", "vision", "patch_embed", "merger")

# Keep MTP tensors by default. Different upstream revisions may or may not
# materialize these modules in the target class; dropping them unconditionally
# can produce missing randomly-initialized fp32 params and trigger mixed-dtype
# failures under DeepSpeed ZeRO-3.
_DROP_HINTS = ("mtp",)


def is_visual_key(key: str) -> bool:
    lk = key.lower()
    return any(h in lk for h in _VISUAL_HINTS)


def should_drop_key(key: str, drop_mtp: bool) -> bool:
    if not drop_mtp:
        return False
    parts = key.lower().split(".")
    return any(h in parts for h in _DROP_HINTS)


def remap_sft_key(key: str) -> str:
    """Rewrite an SFT (CausalLM-only) key to its multimodal equivalent."""
    # Guard against accidental double-prefixing.
    if key.startswith("model.language_model.language_model."):
        return "model.language_model." + key[len("model.language_model.language_model.") :]

    if key.startswith("model.language_model.") or key.startswith("model.visual."):
        return key

    # Some merged checkpoints store language model keys without the outer
    # `model.` prefix; normalize them here.
    if key.startswith("language_model.lm_head."):
        return "lm_head." + key[len("language_model.lm_head.") :]
    if key.startswith("language_model.model."):
        return "model.language_model." + key[len("language_model.model.") :]
    if key.startswith("language_model."):
        return "model." + key

    # Some variants nest the inner model as model.model.*.
    if key.startswith("model.model."):
        return "model.language_model." + key[len("model.model.") :]

    if key.startswith("model.lm_head."):
        return "lm_head." + key[len("model.lm_head.") :]

    # lm_head stays at the top level for Qwen3_5ForConditionalGeneration.
    if key.startswith("lm_head."):
        return key
    if key.startswith("model."):
        # model.layers.X -> model.language_model.layers.X
        return "model.language_model." + key[len("model.") :]
    # Catch-all: prepend the language-model namespace.
    return "model.language_model." + key


def iter_safetensor_shards(directory: Path):
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            yield directory / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Path to the multimodal base model.")
    parser.add_argument("--sft", required=True, help="Path to the SFT (CausalLM) checkpoint.")
    parser.add_argument("--out", required=True, help="Output directory for the fixed merge.")
    parser.add_argument("--version", default="v2-bf16", help="Marker stored in fix_merge_info.json for cache validation.")
    parser.add_argument(
        "--drop-mtp",
        action="store_true",
        help="Drop MTP tensors from SFT checkpoint. Disabled by default for ZeRO-3 safety.",
    )
    args = parser.parse_args()

    base = Path(args.base)
    sft = Path(args.sft)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Copy non-weight artifacts (configs, tokenizer, chat template, processor)
    #    from base. These define the multimodal architecture.
    for name in os.listdir(base):
        src = base / name
        if not src.is_file():
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        if name.endswith(".bin") or name.endswith(".pt"):
            continue
        shutil.copy2(src, out / name)

    # Prefer SFT's tokenizer/template if it differs (matches what was actually trained).
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                 "generation_config.json", "special_tokens_map.json"):
        src = sft / name
        if src.is_file():
            shutil.copy2(src, out / name)

    # 2) Pull every visual tensor from the base model (correct keys).
    # Cast everything to bf16 to keep a single dtype across the checkpoint;
    # mixed dtypes (e.g. base float32 visual + bf16 LLM) trigger DeepSpeed
    # ZeRO-3's `assert len(set(t.dtype for t in tensors)) == 1` in defragment.
    visual_tensors = {}
    for shard in iter_safetensor_shards(base):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if is_visual_key(key):
                    visual_tensors[key] = f.get_tensor(key).to(torch.bfloat16)

    # 3) Pull every LLM tensor from the SFT checkpoint, remapping keys.
    llm_tensors = {}
    skipped_visual = 0
    skipped_mtp = 0
    for shard in iter_safetensor_shards(sft):
        with safe_open(shard, framework="pt") as f:
            for key in f.keys():
                if is_visual_key(key):
                    skipped_visual += 1
                    continue
                if should_drop_key(key, args.drop_mtp):
                    skipped_mtp += 1
                    continue
                new_key = remap_sft_key(key)
                llm_tensors[new_key] = f.get_tensor(key).to(torch.bfloat16)

    merged = {**visual_tensors, **llm_tensors}

    # Sanity checks to fail fast before writing a bad checkpoint.
    dtype_set = {t.dtype for t in merged.values()}
    if len(dtype_set) != 1:
        raise RuntimeError(f"[fix_merge] mixed dtypes detected in merged tensors: {sorted(str(d) for d in dtype_set)}")

    bad_double_prefix = [k for k in merged if "language_model.language_model." in k]
    if bad_double_prefix:
        preview = bad_double_prefix[:5]
        raise RuntimeError(
            "[fix_merge] detected doubly-prefixed language_model keys, e.g. "
            f"{preview}. Check SFT key namespace/remap logic."
        )

    print(f"[fix_merge] visual={len(visual_tensors)} llm={len(llm_tensors)} "
          f"skipped_sft_visual={skipped_visual} skipped_mtp={skipped_mtp} "
          f"total={len(merged)}")

    # 4) Write a single safetensors file. ~18GB for 9B bf16 — fine.
    out_path = out / "model.safetensors"
    save_file(merged, out_path, metadata={"format": "pt"})

    # Drop any stale shard index.
    stale_index = out / "model.safetensors.index.json"
    if stale_index.exists():
        stale_index.unlink()

    # Sanity: write a marker for traceability.
    (out / "fix_merge_info.json").write_text(
        json.dumps(
            {
                "version": args.version,
                "dtype": "bfloat16",
                "base": str(base),
                "sft": str(sft),
                "visual_tensors": len(visual_tensors),
                "llm_tensors": len(llm_tensors),
                "skipped_sft_visual": skipped_visual,
                "skipped_mtp": skipped_mtp,
                "drop_mtp": bool(args.drop_mtp),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[fix_merge] wrote {out_path}")


if __name__ == "__main__":
    main()
