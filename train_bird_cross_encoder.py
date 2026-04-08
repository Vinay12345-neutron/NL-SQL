#!/usr/bin/env python3
import json
import os
import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DATA_PATH = "results/bird_cross_encoder_training_data.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = "models/bird_execution_router"
BATCH_SIZE = 4  # Pushing this high because the RTX 5090 has massive VRAM
EPOCHS = 4

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. FORMAT THE DATA ---
def prepare_training_data(filepath):
    examples = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at {filepath}. Did you run the merge script?")
        
    with open(filepath, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    for record in records:
        query = record["user_query"]
        gold_db = record["gold_db"]
        
        for ctx in record["candidate_contexts"]:
            db_id = ctx["db_id"]
            status = ctx["execution_status"]
            error = ctx["execution_error"] or "None"
            sql = ctx["sql"]
            
            # The Evidence String (What the model reads to make its decision)
            evidence_text = f"Status: {status} | Error: {error} | Database: {db_id} | SQL: {sql}"
            
            # Label: 1.0 for the correct DB, 0.0 for the distractors
            label = 1.0 if db_id == gold_db else 0.0
            examples.append(InputExample(texts=[query, evidence_text], label=label))
            
    return examples

print("Loading and formatting BIRD training data...")
all_examples = prepare_training_data(DATA_PATH)

# --- 3. SPLIT DATA (90% Train / 10% Validation) ---
train_examples, val_examples = train_test_split(all_examples, test_size=0.1, random_state=42)
print(f"Created {len(train_examples)} training pairs and {len(val_examples)} validation pairs.")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# --- 4. INITIALIZE MODEL ---
print(f"Loading base model: {MODEL_NAME} onto RTX 5090...")
# num_labels=1 tells it to output a continuous score (0.0 to 1.0)
model = CrossEncoder(MODEL_NAME, num_labels=1, max_length=1024)

# --- 5. SETUP EVALUATOR ---
# This will test the model after every epoch to ensure it's actually learning
val_evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='bird-val')

# --- 6. TRAIN! ---
print(f"\nStarting training for {EPOCHS} epochs...")
model.fit(
    train_dataloader=train_dataloader,
    evaluator=val_evaluator,
    epochs=EPOCHS,
    warmup_steps=100,
    output_path=OUTPUT_DIR,
    optimizer_params={'lr': 2e-5},
    show_progress_bar=True
)

print(f"\n✅ Training complete! Your custom model is saved to {OUTPUT_DIR}")