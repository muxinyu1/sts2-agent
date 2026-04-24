import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from dataclasses import dataclass


@dataclass
class WeightedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        weights = [feature.pop("weight") for feature in features]
    
        batch = super().__call__(features, return_tensors=return_tensors)

        batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch

class WeightedSFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        sample_weights = inputs.pop("weight").to(model.device)

        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        token_losses = token_losses.view(shift_labels.size())
        
        mask = (shift_labels != -100).to(logits.dtype)
        sample_losses = (token_losses * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        
        weighted_loss = (sample_losses * sample_weights).mean()
        
        return (weighted_loss, outputs) if return_outputs else weighted_loss


def prepare_dataset(data_path, tokenizer, max_seq_len=4096):
    dataset = load_dataset("json", data_files=data_path, split="train")

    def to_id_list(tokenized):
        # Some tokenizer paths may return tokenizers.Encoding instead of list[int].
        if hasattr(tokenized, "ids"):
            tokenized = tokenized.ids
        elif isinstance(tokenized, dict) and "input_ids" in tokenized:
            tokenized = tokenized["input_ids"]

        if isinstance(tokenized, torch.Tensor):
            tokenized = tokenized.tolist()
        elif isinstance(tokenized, tuple):
            tokenized = list(tokenized)

        if tokenized and isinstance(tokenized[0], list):
            tokenized = tokenized[0]

        return [int(x) for x in tokenized]
    
    def process_func(example):
        system_prompt = example.get("system_prompt", "")
        user_prompt = example.get("user_prompt", "")
        response = example.get("response", "")

        if not isinstance(system_prompt, str):
            system_prompt = str(system_prompt)
        if not isinstance(user_prompt, str):
            user_prompt = str(user_prompt)
        if not isinstance(response, str):
            response = str(response)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        prompt_ids = to_id_list(prompt_ids)
        
        
        full_messages = messages + [{"role": "assistant", "content": response}]
        input_ids = tokenizer.apply_chat_template(
            full_messages, tokenize=True
        )
        input_ids = to_id_list(input_ids)
        
        # 截断
        input_ids = input_ids[:max_seq_len]
        
        
        labels = input_ids.copy()
        prompt_len = len(prompt_ids)
        
        labels[:min(prompt_len, max_seq_len)] = [-100] * min(prompt_len, max_seq_len)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "weight": float(example.get("weight", 1.0))
        }

    
    processed_dataset = dataset.map(
        process_func,
        remove_columns=dataset.column_names,
        num_proc=16,
        writer_batch_size=32
    )
    return processed_dataset


def main():
    output_dir = "./saves/Qwen3.5-9B/SFT"
    model_id = "/models/models/Qwen3.5-9B"
    data_path = "./distill/sft_weighted.jsonl"

    os.makedirs(output_dir, exist_ok=True)
    
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    
    train_dataset = prepare_dataset(data_path, tokenizer)
    
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    )
    
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,                       
        remove_unused_columns=False,    
        report_to="tensorboard",
        logging_dir=os.path.join(output_dir, "logs"),
        deepspeed="./distill/ds_z2_config.json"
    )
    
    
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=WeightedDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8),
    )
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()