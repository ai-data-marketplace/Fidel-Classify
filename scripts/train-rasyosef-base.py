import os, gc, json, random, warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)


SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {DEVICE}")

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_cache()

MODEL_NAME   = "rasyosef/bert-small-amharic"
DATA_PATH    = "/kaggle/input/datasets/amanuelfisseha/amharic-multi-domain-text-classification-dataset/amharic_domain_classification_dataset_v2.jsonl"
OUTPUT_DIR   = "/kaggle/working/bert_small_amharic_model"
LOG_DIR      = "./logs_bert_small"
MAX_LEN      = 256
BATCH_SIZE   = 16        
GRAD_ACCUM   = 2
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
TEST_SIZE    = 0.20

LABELS = ["Education", "Health", "Religion", "Politics", "Law", "General", "Finance"]
NUM_LABELS  = len(LABELS)
LABEL2ID    = {l: i for i, l in enumerate(LABELS)}
ID2LABEL    = {i: l for i, l in enumerate(LABELS)}


records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            obj = json.loads(line)
            if obj.get("label") in LABEL2ID:
                records.append({"text": obj["text"], "label": obj["label"]})

df = pd.DataFrame(records)

df["label_id"] = df["label"].map(LABEL2ID)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["label_id"],
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train size : {len(train_df)}")
print(f"Test  size : {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class AmharicTextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts     = list(texts)
        self.labels    = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = AmharicTextDataset(
    train_df["text"].tolist(),
    train_df["label_id"].tolist(),
    tokenizer,
    MAX_LEN,
)
test_dataset = AmharicTextDataset(
    test_df["text"].tolist(),
    test_df["label_id"].tolist(),
    tokenizer,
    MAX_LEN,
)

print(f"\n[INFO] Loading model '{MODEL_NAME}' with {NUM_LABELS} classification heads …")
clear_cache()

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)
model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy"  : accuracy_score(labels, preds),
        "macro_f1"  : f1_score(labels, preds, average="macro",     zero_division=0),
        "macro_p"   : precision_score(labels, preds, average="macro", zero_division=0),
        "macro_r"   : recall_score(labels, preds, average="macro",  zero_division=0),
    }


USE_FP16 = torch.cuda.is_available()   # fp16 only on GPU

training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    logging_dir                 = LOG_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE * 2,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = LR,
    weight_decay                = WEIGHT_DECAY,
    warmup_ratio                = 0.1,
    fp16                        = USE_FP16,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    greater_is_better           = True,
    logging_steps               = 50,
    report_to                   = "none",
    seed                        = SEED,
    dataloader_num_workers      = 2,
    save_total_limit            = 2,
)

print(f"\nfp16 mixed-precision : {USE_FP16}")



trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = test_dataset,
    processing_class       = tokenizer,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("\n" + "="*60)
print("  Starting fine-tuning …")
print("="*60)
clear_cache()

train_result = trainer.train()

print("\nTraining complete.")
print(f" Train runtime   : {train_result.metrics['train_runtime']:.1f}s")
print(f" Samples / sec   : {train_result.metrics['train_samples_per_second']:.2f}")



trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nRunning final evaluation on test set …")
clear_cache()

eval_results = trainer.evaluate(test_dataset)
print("\nEvaluation metrics")
for k, v in eval_results.items():
    print(f"  {k:<40} {v:.4f}" if isinstance(v, float) else f"  {k:<40} {v}")



pred_output = trainer.predict(test_dataset)
y_pred = np.argmax(pred_output.predictions, axis=-1)
y_true = test_df["label_id"].tolist()

print("\nClassification Report")
print(classification_report(y_true, y_pred, target_names=LABELS, zero_division=0))


cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=LABELS,
    yticklabels=LABELS,
    linewidths=0.5,
    ax=ax,
)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title("Confusion Matrix – BERT-small-Amharic Classifier", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.show()


log_history = trainer.state.log_history
train_loss, eval_loss, eval_f1 = [], [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_loss.append((entry["step"], entry["loss"]))
    if "eval_loss" in entry:
        eval_loss.append((entry["epoch"], entry["eval_loss"]))
        eval_f1.append((entry["epoch"],   entry.get("eval_macro_f1", 0)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if train_loss:
    steps, losses = zip(*train_loss)
    axes[0].plot(steps, losses, color="steelblue", linewidth=1.5)
    axes[0].set_xlabel("Step"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")

if eval_loss and eval_f1:
    epochs_el, el = zip(*eval_loss)
    epochs_ef, ef = zip(*eval_f1)
    axes[1].plot(epochs_el, el, label="Eval Loss",     color="tomato",    marker="o")
    axes[1].plot(epochs_ef, ef, label="Macro F1",      color="seagreen",  marker="s")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Evaluation Metrics per Epoch")
    axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.show()



print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  Model        : {MODEL_NAME}")
print(f"  Classes      : {LABELS}")
print(f"  Train / Test : {len(train_dataset)} / {len(test_dataset)}")
print(f"  Accuracy     : {eval_results.get('eval_accuracy', 0):.4f}")
print(f"  Macro F1     : {eval_results.get('eval_macro_f1', 0):.4f}")
print(f"  Macro P      : {eval_results.get('eval_macro_p', 0):.4f}")
print(f"  Macro R      : {eval_results.get('eval_macro_r', 0):.4f}")
print(f"\n  Model saved to : {OUTPUT_DIR}")
print("="*60)
