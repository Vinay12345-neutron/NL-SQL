import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# --- 1. CONFIGURATION ---
DATA_PATH = "results/bird_cross_encoder_training_data.jsonl"
MODEL_NAME = "microsoft/deberta-v3-large"
OUTPUT_DIR = "models/bird_execution_router"
MAX_LENGTH = 1024

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. FORMAT THE DATA ---
def load_data(filepath):
    queries, evidences, labels = [], [], []
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
            
            evidence_text = f"Status: {status} | Error: {error} | Database: {db_id} | SQL: {sql}"
            label = 1.0 if db_id == gold_db else 0.0
            
            queries.append(query)
            evidences.append(evidence_text)
            labels.append(label)
            
    return queries, evidences, labels

print("Loading data...")
queries, evidences, labels = load_data(DATA_PATH)

# Split data
train_q, val_q, train_e, val_e, train_l, val_l = train_test_split(queries, evidences, labels, test_size=0.1, random_state=42)

# --- 3. TOKENIZATION ---
print("Tokenizing data...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(train_q, train_e, truncation=True, padding=True, max_length=MAX_LENGTH)
val_encodings = tokenizer(val_q, val_e, truncation=True, padding=True, max_length=MAX_LENGTH)

class BIRDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BIRDDataset(train_encodings, train_l)
val_dataset = BIRDDataset(val_encodings, val_l)

# --- 4. INITIALIZE MODEL ---
print("Loading Model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)

# --- 5. ROBUST TRAINING ARGUMENTS (The VRAM Fix) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    per_device_train_batch_size=1,            # PHYSICAL BATCH SIZE: 1 (Impossible to OOM)
    gradient_accumulation_steps=8,            # MATHEMATICAL BATCH SIZE: 8
    per_device_eval_batch_size=4,
    warmup_steps=100,
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",                    # Fix for newer transformers warning
    save_strategy="epoch",
    fp16=True,                                # USE MIXED PRECISION (Cuts VRAM usage in half!)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# --- 6. TRAIN! ---
print("Starting robust training loop...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete!")