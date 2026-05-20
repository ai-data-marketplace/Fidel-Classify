"""
Multi-Head AfriBERTa Training Script
=====================================
Three parallel output heads on castorini/afriberta_small:
  1. Language Head   – Binary  (0: Other/Mixed, 1: Amharic)
  2. Readability Head – Binary  (0: Broken/OCR,  1: Clear)
  3. Domain Head      – 7-class (matches reference architecture)

Strict hierarchical loss:
  • Language   → always active
  • Readability → gated by lang_check == 1
  • Domain      → gated by lang_check == 1 AND readability_check == 1

Target: Kaggle Dual T4 x2 via accelerate DDP.
Launch:  accelerate launch --multi_gpu --num_processes=2 scripts/train-multihead-afriberta.py
"""

import os, gc, json, random, warnings, math
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

from transformers import AutoTokenizer, AutoModel, AutoConfig, set_seed
from accelerate import Accelerator
from accelerate.utils import set_seed as acc_set_seed


SEED         = 42
MODEL_NAME   = "castorini/afriberta_small"



DATA_PATH    = "/kaggle/input/amharic-domain-sense-v2/final_training_v3.jsonl"

OUTPUT_DIR   = "/kaggle/working/multihead_afriberta"
LOG_DIR      = os.path.join(OUTPUT_DIR, "logs")
MAX_LEN      = 256
BATCH_SIZE   = 16
GRAD_ACCUM   = 2
EPOCHS       = 5
LR           = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
TEST_SIZE    = 0.20
PATIENCE     = 2  # early-stopping patience (epochs)

DOMAIN_LABELS = ["Education", "Health", "Religion", "Politics",
                 "Law", "General", "Finance"]
NUM_DOMAINS   = len(DOMAIN_LABELS)
DOMAIN2ID     = {l: i for i, l in enumerate(DOMAIN_LABELS)}
ID2DOMAIN     = {i: l for i, l in enumerate(DOMAIN_LABELS)}

IGNORE_INDEX  = -100


def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    acc_set_seed(seed)



# Model – Multi-Head AfriBERTa
class MultiHeadAfriBERTa(nn.Module):
    """
    Shared XLM-R backbone with three classification heads.
    The Domain Head replicates the dropout → dense → tanh → dropout → proj
    architecture of XLMRobertaClassificationHead used in the reference
    train-afriberta.py script (via AutoModelForSequenceClassification).
    """

    def __init__(self, model_name: str, num_domain_labels: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        config = self.backbone.config
        hidden   = config.hidden_size
        drop_p   = (getattr(config, "classifier_dropout", None)
                     or config.hidden_dropout_prob)

        # ── Language Head (binary) ──
        self.lang_head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(hidden, 2),
        )

        # ── Readability Head (binary) ──
        self.read_head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(hidden, 2),
        )

        # ── Domain Head (exact match to reference) ──
        self.domain_dropout = nn.Dropout(drop_p)
        self.domain_dense   = nn.Linear(hidden, hidden)
        self.domain_out     = nn.Linear(hidden, num_domain_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_tok = out.last_hidden_state[:, 0, :]        

        lang_logits = self.lang_head(cls_tok)            
        read_logits = self.read_head(cls_tok)             

        # Domain head – matches XLMRobertaClassificationHead
        x = self.domain_dropout(cls_tok)
        x = self.domain_dense(x)
        x = torch.tanh(x)
        x = self.domain_dropout(x)
        domain_logits = self.domain_out(x)                

        return lang_logits, read_logits, domain_logits



# Dataset
class MultiHeadDataset(Dataset):

    def __init__(self, texts, lang_labels, read_labels, domain_labels,
                 tokenizer, max_length: int = 256):
        self.texts         = list(texts)
        self.lang_labels   = list(lang_labels)
        self.read_labels   = list(read_labels)
        self.domain_labels = list(domain_labels)
        self.tokenizer     = tokenizer
        self.max_length    = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "lang_label":     torch.tensor(self.lang_labels[idx],   dtype=torch.long),
            "read_label":     torch.tensor(self.read_labels[idx],   dtype=torch.long),
            "domain_label":   torch.tensor(self.domain_labels[idx], dtype=torch.long),
        }



# Hierarchical Loss  (zero-corruption gating)
def hierarchical_loss(lang_logits, read_logits, domain_logits,
                      lang_labels, read_labels, domain_labels):
    """
    Strict gated loss:
      1. lang_loss   – always computed.
      2. read_loss   – only where lang_labels == 1.
      3. domain_loss – only where lang_labels == 1 AND read_labels == 1.
    Existing -100 values in labels are respected on top of the gating.
    """
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="mean")

    # ── Language: always active ──
    lang_loss = ce(lang_logits, lang_labels)

    # ── Readability: gated by lang == 1 ──
    gated_read = read_labels.clone()
    gated_read[lang_labels != 1] = IGNORE_INDEX
    n_read = (gated_read != IGNORE_INDEX).sum().item()
    read_loss = ce(read_logits, gated_read) if n_read > 0 else (lang_logits * 0).sum()

    # ── Domain: gated by lang == 1 AND read == 1 ──
    gated_dom = domain_labels.clone()
    gated_dom[(lang_labels != 1) | (read_labels != 1)] = IGNORE_INDEX
    n_dom = (gated_dom != IGNORE_INDEX).sum().item()
    domain_loss = ce(domain_logits, gated_dom) if n_dom > 0 else (lang_logits * 0).sum()

    total = lang_loss + read_loss + domain_loss
    return total, lang_loss, read_loss, domain_loss



# Evaluation helper
@torch.no_grad()
def evaluate(model, dataloader, accelerator):
    model.eval()
    all_lang_p, all_lang_t = [], []
    all_read_p, all_read_t = [], []
    all_dom_p,  all_dom_t  = [], []
    total_loss = 0.0
    n_batches  = 0

    for batch in dataloader:
        lang_log, read_log, dom_log = model(
            batch["input_ids"], batch["attention_mask"]
        )
        loss, _, _, _ = hierarchical_loss(
            lang_log, read_log, dom_log,
            batch["lang_label"], batch["read_label"], batch["domain_label"],
        )
        total_loss += loss.detach().float().item()
        n_batches  += 1

        # predictions
        lang_pred = lang_log.argmax(dim=-1)
        read_pred = read_log.argmax(dim=-1)
        dom_pred  = dom_log.argmax(dim=-1)

        # gather across GPUs
        lang_pred, lang_t = accelerator.gather_for_metrics(
            (lang_pred, batch["lang_label"]))
        read_pred, read_t = accelerator.gather_for_metrics(
            (read_pred, batch["read_label"]))
        dom_pred, dom_t   = accelerator.gather_for_metrics(
            (dom_pred, batch["domain_label"]))

        all_lang_p.append(lang_pred.cpu()); all_lang_t.append(lang_t.cpu())
        all_read_p.append(read_pred.cpu()); all_read_t.append(read_t.cpu())
        all_dom_p.append(dom_pred.cpu());   all_dom_t.append(dom_t.cpu())

    # concatenate
    lp = torch.cat(all_lang_p).numpy();  lt = torch.cat(all_lang_t).numpy()
    rp = torch.cat(all_read_p).numpy();  rt = torch.cat(all_read_t).numpy()
    dp = torch.cat(all_dom_p).numpy();   dt = torch.cat(all_dom_t).numpy()

    # filter valid (≠ -100) per head
    lv = lt != IGNORE_INDEX
    rv = rt != IGNORE_INDEX
    dv = dt != IGNORE_INDEX

    metrics = {"eval_loss": total_loss / max(n_batches, 1)}

    if lv.sum() > 0:
        metrics["lang_acc"] = accuracy_score(lt[lv], lp[lv])
        metrics["lang_f1"]  = f1_score(lt[lv], lp[lv], average="binary",
                                       zero_division=0)
    if rv.sum() > 0:
        metrics["read_acc"] = accuracy_score(rt[rv], rp[rv])
        metrics["read_f1"]  = f1_score(rt[rv], rp[rv], average="binary",
                                       zero_division=0)
    if dv.sum() > 0:
        metrics["domain_acc"]      = accuracy_score(dt[dv], dp[dv])
        metrics["domain_macro_f1"] = f1_score(dt[dv], dp[dv], average="macro",
                                              zero_division=0)

    model.train()
    return metrics, (lp, lt, lv), (rp, rt, rv), (dp, dt, dv)



# Plotting helpers
def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    epochs = [h["epoch"] for h in history]

    for ax, key, label, color in [
        (axes[0], "eval_loss",        "Eval Loss",       "tomato"),
        (axes[1], "domain_macro_f1",  "Domain Macro F1", "seagreen"),
        (axes[2], "lang_f1",          "Lang F1",         "steelblue"),
    ]:
        vals = [h.get(key, 0) for h in history]
        ax.plot(epochs, vals, marker="o", color=color, linewidth=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel(label); ax.set_title(label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()



# Main
def main():
    seed_everything(SEED)
    clear_cache()

    # ── Accelerator ──
    accelerator = Accelerator(
        gradient_accumulation_steps=GRAD_ACCUM,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="all",
        project_dir=LOG_DIR,
    )
    is_main = accelerator.is_main_process

    if is_main:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        print("=" * 60)
        print("  Multi-Head AfriBERTa Trainer (DDP)")
        print("=" * 60)
        print(f"  Model          : {MODEL_NAME}")
        print(f"  Devices        : {accelerator.num_processes}")
        print(f"  Mixed precision: {accelerator.mixed_precision}")
        print(f"  Grad accum     : {GRAD_ACCUM}")
        print("=" * 60)

    # ── Load data ──
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").strip()
            if not text:
                continue

            lang  = int(obj.get("lang_check", IGNORE_INDEX))
            read  = int(obj.get("readability_check", IGNORE_INDEX))

            # Domain label
            if "label" in obj and obj["label"] in DOMAIN2ID:
                dom = DOMAIN2ID[obj["label"]]
            elif "domain_id" in obj:
                dom = int(obj["domain_id"])
            else:
                dom = IGNORE_INDEX

            records.append({
                "text": text, "lang": lang, "read": read, "domain": dom,
            })

    df = pd.DataFrame(records)
    if is_main:
        print(f"\n[INFO] Total samples loaded: {len(df)}")
        print(f"  lang_check distribution:\n{df['lang'].value_counts().to_string()}")
        print(f"  read distribution:\n{df['read'].value_counts().to_string()}")
        dom_valid = df[df["domain"] != IGNORE_INDEX]
        print(f"  domain distribution ({len(dom_valid)} valid):")
        print(f"{dom_valid['domain'].value_counts().sort_index().to_string()}")

    # ── Stratify on lang_check for balanced split ──
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df["lang"],
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    if is_main:
        print(f"\n[INFO] Train size: {len(train_df)}")
        print(f"[INFO] Test  size: {len(test_df)}")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = MultiHeadDataset(
        train_df["text"].tolist(), train_df["lang"].tolist(),
        train_df["read"].tolist(), train_df["domain"].tolist(),
        tokenizer, MAX_LEN,
    )
    test_ds = MultiHeadDataset(
        test_df["text"].tolist(), test_df["lang"].tolist(),
        test_df["read"].tolist(), test_df["domain"].tolist(),
        tokenizer, MAX_LEN,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 2, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── Model ──
    model = MultiHeadAfriBERTa(MODEL_NAME, NUM_DOMAINS)

    if is_main:
        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n[INFO] Parameters – total: {total_p:,}  trainable: {train_p:,}")

    # ── Optimizer & Scheduler ──
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                     if not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                     if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)

    num_update_steps = math.ceil(
        len(train_loader) / GRAD_ACCUM) * EPOCHS
    warmup_steps = int(num_update_steps * WARMUP_RATIO)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=num_update_steps,
    )

    # ── Prepare with accelerate ──
    model, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, test_loader, scheduler,
    )

    # ── Training loop ──
    best_metric   = -1.0
    patience_left = PATIENCE
    history       = []
    global_step   = 0

    if is_main:
        print("\n" + "=" * 60)
        print("  Starting training …")
        print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                lang_log, read_log, dom_log = model(
                    batch["input_ids"], batch["attention_mask"])

                loss, l_loss, r_loss, d_loss = hierarchical_loss(
                    lang_log, read_log, dom_log,
                    batch["lang_label"], batch["read_label"],
                    batch["domain_label"],
                )

                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss  += loss.detach().float().item()
            epoch_steps += 1
            global_step += 1

            if is_main and global_step % 50 == 0:
                avg = epoch_loss / epoch_steps
                print(f"  [epoch {epoch}  step {global_step}]  "
                      f"loss={avg:.4f}  "
                      f"lang={l_loss.item():.4f}  "
                      f"read={r_loss.item():.4f}  "
                      f"dom={d_loss.item():.4f}")

        # ── Evaluate ──
        metrics, lang_data, read_data, dom_data = evaluate(
            model, test_loader, accelerator)
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / max(epoch_steps, 1)
        history.append(metrics)

        if is_main:
            print(f"\n  ── Epoch {epoch}/{EPOCHS} ──")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"    {k:<25} {v:.4f}")
                else:
                    print(f"    {k:<25} {v}")

        # ── Early stopping (on domain_macro_f1) ──
        current = metrics.get("domain_macro_f1", 0.0)
        if current > best_metric:
            best_metric   = current
            patience_left = PATIENCE
            # save best checkpoint
            if is_main:
                accelerator.wait_for_everyone()
                unwrapped = accelerator.unwrap_model(model)
                save_dir = os.path.join(OUTPUT_DIR, "best_model")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(unwrapped.state_dict(), os.path.join(save_dir, "model.pt"))
                tokenizer.save_pretrained(save_dir)
                # save config for reproducibility
                cfg = unwrapped.backbone.config
                cfg.save_pretrained(save_dir)
                print(f"Best model saved (domain_macro_f1={best_metric:.4f})")
        else:
            patience_left -= 1
            if is_main:
                print(f"No improvement (patience {patience_left}/{PATIENCE})")
            if patience_left <= 0:
                if is_main:
                    print("Early stopping triggered.")
                break

        clear_cache()

    # ── Final evaluation on best model ──
    if is_main:
        print("\n" + "=" * 60)
        print("  Final Evaluation (best checkpoint)")
        print("=" * 60)

    best_path = os.path.join(OUTPUT_DIR, "best_model", "model.pt")
    if os.path.exists(best_path):
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(torch.load(best_path, map_location="cpu"))

    final_metrics, lang_data, read_data, dom_data = evaluate(
        model, test_loader, accelerator)

    lp, lt, lv = lang_data
    rp, rt, rv = read_data
    dp, dt, dv = dom_data

    if is_main:
        # ── Language Head Report ──
        print("\n── Language Head ──")
        print(f"  Accuracy : {final_metrics.get('lang_acc', 0):.4f}")
        print(f"  F1       : {final_metrics.get('lang_f1', 0):.4f}")
        if lv.sum() > 0:
            print(classification_report(
                lt[lv], lp[lv],
                target_names=["Other/Mixed", "Amharic"],
                zero_division=0))

        # ── Readability Head Report ──
        print("── Readability Head ──")
        print(f"  Accuracy : {final_metrics.get('read_acc', 0):.4f}")
        print(f"  F1       : {final_metrics.get('read_f1', 0):.4f}")
        if rv.sum() > 0:
            print(classification_report(
                rt[rv], rp[rv],
                target_names=["Broken/OCR", "Clear"],
                zero_division=0))

        # ── Domain Head Report ──
        print("── Domain Head ──")
        print(f"  Accuracy : {final_metrics.get('domain_acc', 0):.4f}")
        print(f"  Macro F1 : {final_metrics.get('domain_macro_f1', 0):.4f}")
        if dv.sum() > 0:
            print(classification_report(
                dt[dv], dp[dv],
                target_names=DOMAIN_LABELS,
                zero_division=0))

            # Domain confusion matrix
            plot_confusion_matrix(
                dt[dv], dp[dv], DOMAIN_LABELS,
                "Confusion Matrix – Domain Head",
                os.path.join(OUTPUT_DIR, "domain_confusion_matrix.png"))
            print("  Confusion matrix saved.")

        # Language confusion matrix
        if lv.sum() > 0:
            plot_confusion_matrix(
                lt[lv], lp[lv], ["Other/Mixed", "Amharic"],
                "Confusion Matrix – Language Head",
                os.path.join(OUTPUT_DIR, "lang_confusion_matrix.png"))

        # Readability confusion matrix
        if rv.sum() > 0:
            plot_confusion_matrix(
                rt[rv], rp[rv], ["Broken/OCR", "Clear"],
                "Confusion Matrix – Readability Head",
                os.path.join(OUTPUT_DIR, "read_confusion_matrix.png"))

        # Training curves
        if history:
            plot_training_curves(
                history,
                os.path.join(OUTPUT_DIR, "training_curves.png"))
            print("  Training curves saved.")

        # ── Final Summary ──
        print("\n" + "=" * 60)
        print("  FINAL SUMMARY")
        print("=" * 60)
        print(f"  Model             : {MODEL_NAME}")
        print(f"  Domain classes    : {DOMAIN_LABELS}")
        print(f"  Train / Test      : {len(train_ds)} / {len(test_ds)}")
        print(f"  ---")
        print(f"  Lang Accuracy     : {final_metrics.get('lang_acc', 0):.4f}")
        print(f"  Lang F1           : {final_metrics.get('lang_f1', 0):.4f}")
        print(f"  Read Accuracy     : {final_metrics.get('read_acc', 0):.4f}")
        print(f"  Read F1           : {final_metrics.get('read_f1', 0):.4f}")
        print(f"  Domain Accuracy   : {final_metrics.get('domain_acc', 0):.4f}")
        print(f"  Domain Macro F1   : {final_metrics.get('domain_macro_f1', 0):.4f}")
        print(f"\n  Model saved to    : {OUTPUT_DIR}/best_model")
        print("=" * 60)

        # Save metrics to JSON
        with open(os.path.join(OUTPUT_DIR, "final_metrics.json"), "w") as f:
            json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v
                       for k, v in final_metrics.items()}, f, indent=2)

    accelerator.end_training()
    clear_cache()


if __name__ == "__main__":
    main()
