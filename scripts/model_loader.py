import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download

DOMAIN_LABELS = ["Education", "Health", "Religion", "Politics", "Law", "General", "Finance"]

class MultiHeadRoberta(nn.Module):
    """
    Shared RoBERTa backbone with three classification heads.
    Matches the architecture for the rasyosef/roberta-base-amharic model.
    """
    def __init__(self, config_path: str, num_domain_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(config_path)
        # Initialize architecture skeleton from config to avoid OSError searching for .bin files
        self.backbone = AutoModel.from_config(config)
        
        hidden = config.hidden_size
        drop_p = (getattr(config, "classifier_dropout", None) or 
                  getattr(config, "hidden_dropout_prob", 0.1))

        # Language Head (binary) 
        self.lang_head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(hidden, 2),
        )

        # Readability Head (binary)
        self.read_head = nn.Sequential(
            nn.Dropout(drop_p),
            nn.Linear(hidden, 2),
        )

        # Domain Head
        self.domain_dropout = nn.Dropout(drop_p)
        self.domain_dense   = nn.Linear(hidden, hidden)
        self.domain_out     = nn.Linear(hidden, num_domain_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls_tok = out.last_hidden_state[:, 0, :]

        lang_logits = self.lang_head(cls_tok)
        read_logits = self.read_head(cls_tok)

        # Domain head logic
        x = self.domain_dropout(cls_tok)
        x = self.domain_dense(x)
        x = torch.tanh(x)
        x = self.domain_dropout(x)
        domain_logits = self.domain_out(x)

        return lang_logits, read_logits, domain_logits


class TextQualityModel:
    def __init__(self, repo_id: str = "amanfisseha/multihead-rasyosef-amharic"):
        """
        Downloads model snapshot from HF Hub, initializes MultiHeadRoberta,
        and loads fine-tuned weights from model.pt.
        """
        print(f"Downloading/loading snapshot from {repo_id}...")
        snapshot_dir = snapshot_download(repo_id)

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
        
        # Use renamed class
        self.model = MultiHeadRoberta(config_path=snapshot_dir, num_domain_labels=len(DOMAIN_LABELS))
        
        model_path = os.path.join(snapshot_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected to find model.pt in {snapshot_dir}")
            
        print("Loading weights from model.pt...")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        if not text.strip():
            return {"error": "Empty input text"}

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            lang_logits, read_logits, domain_logits = self.model(input_ids, attention_mask)
            
            lang_probs = torch.softmax(lang_logits, dim=-1).squeeze(0)
            read_probs = torch.softmax(read_logits, dim=-1).squeeze(0)
            domain_probs = torch.softmax(domain_logits, dim=-1).squeeze(0)
            
            lang_pred = lang_logits.argmax(dim=-1).item()
            read_pred = read_logits.argmax(dim=-1).item()
            domain_pred = domain_logits.argmax(dim=-1).item()

        lang_conf = float(lang_probs[lang_pred])
        read_conf = float(read_probs[read_pred])
        domain_conf = float(domain_probs[domain_pred])

        # --- Quality Thresholds ---
        if lang_pred == 1 and lang_conf < 0.97:
            lang_label = "Other/Mixed"
            lang_conf = float(lang_probs[0])
        else:
            lang_label = "Amharic" if lang_pred == 1 else "Other/Mixed"

        if read_pred == 1 and read_conf < 0.95:
            read_label = "Broken/OCR"
            read_conf = float(read_probs[0])
        else:
            read_label = "Clear" if read_pred == 1 else "Broken/OCR"

        return {
            "language": {"label": lang_label, "confidence": lang_conf},
            "readability": {"label": read_label, "confidence": read_conf},
            "domain": {"label": DOMAIN_LABELS[domain_pred], "confidence": domain_conf}
        }


if __name__ == "__main__":
    try:
        model = TextQualityModel("amanfisseha/multihead-rasyosef-amharic")
        result = model.predict("ጠቅላይ ሚኒስትር ዐቢይ አሕመድ አገራዊ ምክክር እየተደረገ ያለው በአገሪቱ ሕገ መንግሥት ያለ ማብቂያ የተቀመጠውን የአገሪቱን መሪ የሥልጣን ዘመን ለመገደብ መሆኑን ተናገሩ። መንግሥት ሥልጣኑ በሕግ ተገድቦ ሕገ መንግሥት እንዲሻሻል እየጠየቀ ባለበት ሁኔታ በአገራዊ ምክክሩ አንወያይም ያሉ አካላትን ተችተዋል።")
        print("Prediction Result:", result)
    except Exception as e:
        print(f"Error initializing model or predicting: {e}")