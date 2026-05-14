#!/usr/bin/env python3
"""
Phase 1B: Train the Bq/s (Base Query/Schema) Cross-Encoder
===========================================================
Three-Tiered Confidence Cascade Architecture — Spider Dataset

Objective:
    Fine-tune microsoft/deberta-v3-base on the spider_bqs_train_pairs.jsonl data
    generated in Phase 1A. This is a Binary Sequence Classification task:
        Input:  [Query] [SEP] [Schema Context]
        Output: Sigmoid score (0.0 → distractor, 1.0 → correct schema)
    Loss:   Binary Cross-Entropy (BCEWithLogitsLoss, applied via num_labels=1)

Architecture Notes:
    - Uses the HuggingFace Trainer API directly (NOT sentence_transformers.fit())
      to avoid known AutoProcessor + CrossEncoderTrainer crashes on DeBERTa-v3.
    - bf16=True instead of fp16 — DeBERTa-v3 uses disentangled attention heads
      that are numerically unstable under fp16 on Ampere/Ada/Hopper GPUs.
    - max_grad_norm=1.0 prevents NaN gradient explosions during early training.

Input:
    data/spider_bqs_train_pairs.jsonl     — Pairs from Phase 1A (22,000 samples)

Output:
    models/spider_bqs_cross_encoder/      — Saved model + tokenizer

Metrics Reported:
    Final Training Loss, Final Validation Loss (per epoch)
"""

import json
import os
import warnings
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Suppress harmless DeBERTa-v3 LayerNorm gamma/beta → weight/bias renaming warnings
warnings.filterwarnings("ignore", message=".*Some weights of.*were not used.*")
warnings.filterwarnings("ignore", message=".*were not initialized from the model checkpoint.*")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "spider_bqs_train_pairs.jsonl")
MODEL_NAME     = "roberta-base"
OUTPUT_DIR     = os.path.join(BASE_DIR, "models", "spider_bqs_cross_encoder_roberta")
MAX_LENGTH     = 512
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------
class PairDataset(torch.utils.data.Dataset):
    """Wraps tokenized encodings and float labels for HuggingFace Trainer."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert each encoding value at this index to a tensor
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Label must be float for BCEWithLogitsLoss (num_labels=1)
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"Loading training pairs from:\n  {DATA_PATH}\n")
queries, schemas, labels = [], [], []

with open(DATA_PATH, "r") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        queries.append(rec["query"])
        schemas.append(rec["schema_context"])
        labels.append(rec["label"])

print(f"Total pairs loaded : {len(queries)}")
print(f"  Positives        : {sum(1 for l in labels if l == 1.0)}")
print(f"  Negatives        : {sum(1 for l in labels if l == 0.0)}\n")


# ---------------------------------------------------------------------------
# 2. Train / Validation split (90 / 10)
# ---------------------------------------------------------------------------
(train_q, val_q,
 train_s, val_s,
 train_l, val_l) = train_test_split(
    queries, schemas, labels,
    test_size=0.10,
    random_state=RANDOM_SEED,
    stratify=[int(l) for l in labels],   # Preserve class balance in both splits
)
print(f"Train pairs      : {len(train_q)}")
print(f"Validation pairs : {len(val_q)}\n")


# ---------------------------------------------------------------------------
# 3. Tokenise
# ---------------------------------------------------------------------------
print(f"Tokenising with {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# DeBERTa cross-encoder expects the two sequences as a pair:
#   sequence_a = user query
#   sequence_b = schema context text
train_enc = tokenizer(
    train_q, train_s,
    truncation=True, padding=True, max_length=MAX_LENGTH
)
val_enc = tokenizer(
    val_q, val_s,
    truncation=True, padding=True, max_length=MAX_LENGTH
)
print("Tokenisation complete.\n")

train_dataset = PairDataset(train_enc, train_l)
val_dataset   = PairDataset(val_enc,   val_l)


# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
print(f"Loading base model: {MODEL_NAME}")
# num_labels=1 → single logit output → BCEWithLogitsLoss applied automatically
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
print("Model loaded.\n")


# ---------------------------------------------------------------------------
# 5. Training Arguments
# ---------------------------------------------------------------------------
# Key stability fixes (same as the proven spider_execution_router training):
#   bf16=True         — DeBERTa-v3 crashes with fp16; bf16 is numerically safe
#   max_grad_norm=1.0 — Clips explosive gradients during early epochs
#   lr=2e-5           — Standard fine-tuning LR for DeBERTa-base
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = 16,           # deberta-base fits comfortably on RTX 5090
    per_device_eval_batch_size  = 32,
    gradient_accumulation_steps = 2,            # Effective batch size = 32
    warmup_steps                = 100,
    learning_rate               = 2e-5,
    max_grad_norm               = 1.0,          # FIX: prevents NaN gradient explosions
    fp16                        = False,
    bf16                        = True,         # FIX: safe on DeBERTa-v3 + Hopper GPU
    logging_dir                 = os.path.join(OUTPUT_DIR, "logs"),  # kept for compatibility
    logging_steps               = 50,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "none",       # No wandb / tensorboard
    seed                        = RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# 6. Train
# ---------------------------------------------------------------------------
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = val_dataset,
)

print("=" * 55)
print("  PHASE 1B — Training Bq/s Cross-Encoder")
print("=" * 55)
print(f"  Base Model  : {MODEL_NAME}")
print(f"  Train pairs : {len(train_dataset)}")
print(f"  Val pairs   : {len(val_dataset)}")
print(f"  Epochs      : {training_args.num_train_epochs}")
print(f"  Batch size  : {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} grad accum")
print(f"  LR          : {training_args.learning_rate}")
print("=" * 55 + "\n")

trainer.train()

# ---------------------------------------------------------------------------
# 7. Save model and tokenizer
# ---------------------------------------------------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Bq/s Cross-Encoder saved to: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 8. Print final loss summary
# ---------------------------------------------------------------------------
history = trainer.state.log_history

# Build per-epoch stats dict — cast all values to float (Trainer stores them as str)
epoch_eval: dict[int, float] = {}
for entry in history:
    if "eval_loss" in entry:
        ep = int(float(entry.get("epoch", 0)))
        epoch_eval[ep] = float(entry["eval_loss"])

# Grab overall final train_loss from the last summary entry
final_train_loss = next(
    (float(e["train_loss"]) for e in reversed(history) if "train_loss" in e), None
)

print("\n" + "=" * 55)
print("  PHASE 1B — Training Complete: Loss Summary")
print("=" * 55)
for ep, eval_l in sorted(epoch_eval.items()):
    print(f"  Epoch {ep}  →  Val Loss: {eval_l:.4f}")
if final_train_loss is not None:
    print(f"  Overall Train Loss (3 epochs): {final_train_loss:.4f}")
print("=" * 55)
