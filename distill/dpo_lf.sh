#!/usr/bin/env bash
set -e

export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_BLOCKING_WAIT=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ---------------------------------------------------------------------------
# Step 1: Pre-scale margins
# dpo_pairs.jsonl stores raw HP-loss deltas (integers, range ~[0, 80]).
# LlamaFactory uses the margin column directly in the DPO loss:
#   loss = -log sigmoid(pref_beta * delta_r - margin)
# We scale by 0.1 so that margins stay in ~[0, 8], matching the original
# dpo.py behaviour (margin_scale=0.1 with pref_beta=0.1).
# ---------------------------------------------------------------------------
echo "[dpo_lf] Scaling margins: distill/dpo_pairs.jsonl -> distill/dpo_pairs_lf.jsonl"
python - <<'EOF'
import json

src = "./distill/dpo_pairs.jsonl"
dst = "./distill/dpo_pairs_lf.jsonl"

with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        rec = json.loads(line)
        rec["margin"] = round(float(rec.get("margin") or 0.0) * 0.1, 4)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Written: {dst}")
EOF

# ---------------------------------------------------------------------------
# Step 2: Launch LlamaFactory DPO training
# ---------------------------------------------------------------------------
echo "[dpo_lf] Starting LlamaFactory DPO training..."
llamafactory-cli train ./distill/dpo_lf.yaml
