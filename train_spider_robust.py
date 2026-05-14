#!/usr/bin/env python3
"""
Spider Cross-Encoder Training (Robust HuggingFace Trainer API)
==============================================================
Uses raw transformers Trainer instead of sentence_transformers.CrossEncoder.fit()
to avoid:
  1. AutoProcessor crash on DeBERTa (newer sentence-transformers >=5.x bug)
  2. _nested_gather AttributeError in CrossEncoderTrainer
  3. NaN grad_norm from gradient explosion (fixed by max_grad_norm + lower LR)
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DATA_PATH  = "results/spider_cross_encoder_training_data.jsonl"
MODEL_NAME = "microsoft/deberta-v3-small"
OUTPUT_DIR = "models/spider_execution_router"
MAX_LENGTH = 512   # DeBERTa-v3-small hard limit

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. LOAD AND FLATTEN DATA ---
def load_data(filepath):
    queries, evidences, labels = [], [], []
    with open(filepath, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]

    for record in records:
        query   = record["user_query"]
        gold_db = record["gold_db"]

        for ctx in record.get("candidate_contexts", []):
            db_id  = ctx["db_id"]
            status = ctx.get("execution_status", "MISSING")
            error  = ctx.get("execution_error") or "None"
            sql    = ctx.get("sql", "")

            # Exact format used in evaluate_spider_pipeline.py
            evidence_text = f"Status: {status} | Error: {error} | Database: {db_id} | SQL: {sql}"

            label = 1.0 if db_id == gold_db else 0.0

            queries.append(query)
            evidences.append(evidence_text)
            labels.append(label)

    return queries, evidences, labels


print("Loading and flattening data...")
queries, evidences, labels = load_data(DATA_PATH)
print(f"Total pairs: {len(queries)} | Positives: {sum(1 for l in labels if l == 1.0)}")

# 90/10 split
train_q, val_q, train_e, val_e, train_l, val_l = train_test_split(
    queries, evidences, labels, test_size=0.1, random_state=42
)
print(f"Training: {len(train_q)} | Validation: {len(val_q)}")


# --- 3. TOKENISATION ---
print(f"Tokenising with {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_enc = tokenizer(train_q, train_e, truncation=True, padding=True, max_length=MAX_LENGTH)
val_enc   = tokenizer(val_q,   val_e,   truncation=True, padding=True, max_length=MAX_LENGTH)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = PairDataset(train_enc, train_l)
val_dataset   = PairDataset(val_enc,   val_l)


# --- 4. MODEL ---
print("Loading base model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)


# --- 5. TRAINING ARGUMENTS ---
# Key fixes vs. the broken sentence_transformers version:
#   - max_grad_norm=1.0  → kills NaN gradient explosions
#   - learning_rate=1e-5 → safer than 2e-5 for DeBERTa
#   - fp16=True          → cuts VRAM, keeps speed
#   - gradient_accumulation_steps=4 → effective batch = 4
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    learning_rate=1e-5,
    max_grad_norm=1.0,              # FIX: prevents NaN grad explosion
    fp16=False,
    bf16=True,                      # FIX: DeBERTa-v3 crashes with fp16; bf16 works correctly
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",               # No wandb/tensorboard needed
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# --- 6. TRAIN ---
print("\nStarting training...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\n✅ Spider model saved to: {OUTPUT_DIR}")