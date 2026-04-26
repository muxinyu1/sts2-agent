"""DPO training with per-sample margin (memory-friendly + vLLM-ready).

Memory strategy
---------------
SFT was already tight on 8xH100; DPO loads BOTH a policy and a reference
model, so we keep the same memory footprint as SFT by:

* Loading only the **LLM half** of the multimodal base (``AutoModelForCausalLM``).
  The vision tower / projector are NOT loaded into GPU memory during training.
* After ``trainer.train()`` finishes, on rank-0 we run a one-shot post-merge
  step that takes the trained LLM safetensors and combines them with the
  vision-tower safetensors from the SFT-complete checkpoint, producing a
  fully self-contained directory that vLLM can serve directly.

This mirrors what you did manually after SFT (`*-complete` folder), but the
script now does it automatically — you don't need to remember to run a
merge.py afterwards.

DPO loss with per-sample margin (TRL native)
--------------------------------------------
    L_i = -log sigmoid( beta * Δr_i - margin_i )
where Δr_i is the implicit-reward gap between chosen and rejected.
Larger ``margin`` -> the chosen response must beat the rejected by more.
"""

import json
import os
import shutil

import torch
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)
from trl import DPOConfig, DPOTrainer


# ---- Dataset ---------------------------------------------------------------
def prepare_dataset(data_path: str, tokenizer, margin_scale: float = 0.1):
    """Render prompts with the chat template and pass margin through."""
    dataset = load_dataset("json", data_files=data_path, split="train")

    def process_func(example):
        system_prompt = str(example.get("system_prompt") or "")
        user_prompt = str(example.get("user_prompt") or "")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Raw HP-loss delta is in [0, ~80]; scale into a moderate range so
        # margin doesn't saturate the sigmoid relative to beta * Δr.
        raw_margin = float(example.get("margin", 0.0) or 0.0)
        margin = max(0.0, raw_margin) * margin_scale

        return {
            "prompt": prompt_text,
            "chosen": str(example.get("chosen") or ""),
            "rejected": str(example.get("rejected") or ""),
            "margin": margin,
        }

    return dataset.map(
        process_func,
        remove_columns=dataset.column_names,
        num_proc=8,
    )


# ---- Post-train: merge trained LLM weights with vision tower ---------------
_NON_LLM_HINTS = (
    "visual", "vision", "image", "video", "audio",
    "merger", "patch_embed", "mm_projector", "projector",
)


def _is_non_llm_key(key: str) -> bool:
    lk = key.lower()
    return any(h in lk for h in _NON_LLM_HINTS)


def _iter_safetensor_shards(directory: str):
    """Yield (filename, full_path) for every .safetensors shard in a dir."""
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors"):
            yield name, os.path.join(directory, name)


def merge_llm_with_vision(trained_llm_dir: str, complete_dir: str, out_dir: str):
    """Combine the DPO-trained LLM weights with the SFT-complete vision tower.

    * From ``trained_llm_dir`` we keep tensors whose key looks like a
      language-model parameter (i.e. NOT matching any non-LLM hint).
    * From ``complete_dir`` we keep every tensor whose key DOES match a
      non-LLM hint (the vision tower / projector).
    * We then rewrite a single sharded safetensors set under ``out_dir`` and
      copy every non-weight artifact from ``complete_dir`` (config,
      processor configs, tokenizer, chat template, ...). The result is a
      stand-alone, vLLM-deployable model.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) Copy non-weight artifacts (configs, tokenizer, processor, etc.) from
    #    the complete (vision-aware) base. Skip safetensors and shard index;
    #    we'll regenerate those.
    for name in os.listdir(complete_dir):
        src = os.path.join(complete_dir, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(".safetensors") or name.endswith(".safetensors.index.json"):
            continue
        if name.endswith(".bin") or name.endswith(".pt"):
            continue  # legacy weight formats; we only emit safetensors
        shutil.copy2(src, os.path.join(out_dir, name))

    # 2) Pull every non-LLM tensor from the complete checkpoint.
    vision_tensors = {}
    for _, path in _iter_safetensor_shards(complete_dir):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if _is_non_llm_key(key):
                    vision_tensors[key] = f.get_tensor(key)

    # 3) Pull every LLM tensor from the trained directory.
    llm_tensors = {}
    for _, path in _iter_safetensor_shards(trained_llm_dir):
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if not _is_non_llm_key(key):
                    llm_tensors[key] = f.get_tensor(key)

    merged = {**vision_tensors, **llm_tensors}
    print(f"[merge] llm tensors: {len(llm_tensors)}  "
          f"vision tensors: {len(vision_tensors)}  "
          f"total: {len(merged)}")

    # 4) Write a single shard (simple + reliable). For 9B bf16 this is ~18GB,
    #    which safetensors handles without issue.
    out_path = os.path.join(out_dir, "model.safetensors")
    save_file(merged, out_path, metadata={"format": "pt"})

    # Drop any stale shard index left over from copying.
    stale_index = os.path.join(out_dir, "model.safetensors.index.json")
    if os.path.exists(stale_index):
        os.remove(stale_index)

    print(f"[merge] wrote {out_path}")


def main():
    output_dir = "./saves/Qwen3.5-9B/DPO"
    base_model_id = "/models/models/Qwen3.5-9B"
    # SFT 产出的 *-complete 含完整视觉塔，作为视觉权重的来源 + 配置模板。
    sft_complete = "./saves/Qwen3.5-9B/SFT/checkpoint-1068-complete"
    data_path = "./distill/dpo_pairs.jsonl"

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # DPO requires pad_token != eos_token: if they are the same, real EOS tokens
    # at the end of chosen/rejected responses get indistinguishably masked as
    # padding, and the model never learns when to stop generating.
    # Qwen tokenizers expose <|endoftext|> (id=0 typically) which is safe to use
    # as a pure padding sentinel since the chat template uses <|im_end|> as EOS.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        if endoftext_id != tokenizer.unk_token_id and endoftext_id != tokenizer.eos_token_id:
            tokenizer.pad_token_id = endoftext_id
        else:
            # Fallback: add a dedicated pad token so it never collides with EOS.
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    # DPO: prompts must be left-padded so response tokens stay right-aligned.
    tokenizer.padding_side = "left"

    train_dataset = prepare_dataset(data_path, tokenizer)

    # Load only the LLM half. For multimodal Qwen, AutoModelForCausalLM
    # picks up the inner causal-LM submodule, leaving the vision tower out
    # of GPU memory — same footprint as SFT.
    common_load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    policy_path = sft_complete if os.path.isdir(sft_complete) else base_model_id
    model = AutoModelForCausalLM.from_pretrained(policy_path, **common_load_kwargs)
    ref_model = AutoModelForCausalLM.from_pretrained(policy_path, **common_load_kwargs)
    # If a new pad token was added, resize embeddings to match the tokenizer vocab.
    model.resize_token_embeddings(len(tokenizer))
    ref_model.resize_token_embeddings(len(tokenizer))
    for p in ref_model.parameters():
        p.requires_grad = False

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-7,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,  # keep the `margin` column
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        deepspeed="./distill/ds_z2_config.json",
        # DPO-specific
        beta=0.1,
        loss_type="sigmoid",  # standard DPO; per-sample margin subtracted in-loss
        max_length=4096,
        max_prompt_length=3072,
    )

    # TRL compatibility: some model classes (e.g. Qwen3_5ForCausalLM loaded via
    # trust_remote_code) may not carry the `warnings_issued` dict that TRL's
    # Trainer uses internally when gradient_checkpointing is enabled.
    for _m in (model, ref_model):
        if not hasattr(_m, "warnings_issued"):
            _m.warnings_issued = {}

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    # Step 1: dump the (LLM-only) trained weights via Trainer (handles
    # DeepSpeed gather across ranks for us).
    trained_llm_dir = os.path.join(output_dir, "trained_llm")
    trainer.save_model(trained_llm_dir)
    tokenizer.save_pretrained(trained_llm_dir)

    # Step 2: on rank-0 only, merge with the vision tower from the
    # SFT-complete checkpoint into a final, vLLM-deployable directory.
    if trainer.is_world_process_zero():
        final_dir = os.path.join(output_dir, "final")
        merge_llm_with_vision(
            trained_llm_dir=trained_llm_dir,
            complete_dir=sft_complete,
            out_dir=final_dir,
        )
        # Make sure tokenizer + processor exist in the final dir too.
        tokenizer.save_pretrained(final_dir)
        try:
            processor = AutoProcessor.from_pretrained(sft_complete, trust_remote_code=True)
            processor.save_pretrained(final_dir)
        except Exception as e:
            print(f"[merge] processor save skipped: {e}")
        print(f"[done] vLLM-deployable checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
