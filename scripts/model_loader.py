import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from huggingface_hub import snapshot_download

DOMAIN_LABELS = ["Education", "Health", "Religion", "Politics", "Law", "General", "Finance"]

class MultiHeadAfriBERTa(nn.Module):
    """
    Shared XLM-R backbone with three classification heads.
    Matches the architecture from the training script exactly.
    """
    def __init__(self, config_path: str, num_domain_labels: int):
        super().__init__()
        # Load config from the snapshot directory
        config = AutoConfig.from_pretrained(config_path)
        # Initialize backbone without pretrained weights since we will load state_dict later
        self.backbone = AutoModel.from_config(config)
        
        hidden = config.hidden_size
        drop_p = (getattr(config, "classifier_dropout", None) or config.hidden_dropout_prob)

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


class TextQualityModel:
    def __init__(self, repo_id: str = "amanfisseha/multihead-afriberta"):
        """
        Downloads the model snapshot from Hugging Face Hub, initializes the architecture,
        loads the state dictionary, and sets up the tokenizer.
        """
        print(f"Downloading/loading snapshot from {repo_id}...")
        snapshot_dir = snapshot_download(repo_id)

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
        
        # Initialize the model architecture based on the downloaded config
        self.model = MultiHeadAfriBERTa(config_path=snapshot_dir, num_domain_labels=len(DOMAIN_LABELS))
        
        # Load the PyTorch weights
        model_path = os.path.join(snapshot_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Expected to find model.pt in {snapshot_dir}")
            
        print("Loading weights from model.pt...")
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        
        # Move to GPU if available and set to eval mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """
        Runs inference on the provided text and returns classifications for
        language, readability, and domain.
        """
        if not text.strip():
            return {"error": "Empty input text"}

        # Tokenize
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
            
            # Apply softmax to get probabilities
            lang_probs = torch.softmax(lang_logits, dim=-1).squeeze(0)
            read_probs = torch.softmax(read_logits, dim=-1).squeeze(0)
            domain_probs = torch.softmax(domain_logits, dim=-1).squeeze(0)
            
            # Predictions
            lang_pred = lang_logits.argmax(dim=-1).item()
            read_pred = read_logits.argmax(dim=-1).item()
            domain_pred = domain_logits.argmax(dim=-1).item()

        return {
            "language": {
                "label": "Amharic" if lang_pred == 1 else "Other/Mixed",
                "confidence": float(lang_probs[lang_pred])
            },
            "readability": {
                "label": "Clear" if read_pred == 1 else "Broken/OCR",
                "confidence": float(read_probs[read_pred])
            },
            "domain": {
                "label": DOMAIN_LABELS[domain_pred],
                "confidence": float(domain_probs[domain_pred])
            }
        }

if __name__ == "__main__":
    # Example usage
    try:
        model = TextQualityModel("amanfisseha/multihead-afriberta")
        result = model.predict("በአጠቃላይ የፖለቲካ መረጋጋት ለሀገር ኢኮኖሚ እድገት መሰረት ነው። ኢንቨስተሮች በሀገር ውስጥ መዋዕለ ንዋያቸውን ለማፍሰስ ሰላም ይፈልጋሉ። የህዝቡ የኑሮ ሁኔታ መሻሻል ደግሞ ለፖለቲካዊ መረጋጋት አስተዋፅኦ ያደርጋል። መንግስት በልማት ስራዎች ላይ የሚያደርገው ኢንቨስትመንት ለዜጎች የስራ እድል ከመፍጠሩም በላይ ድህነትን ለመቀነስ ይረዳል። ማህበራዊ ፍትህ ሲረጋገጥ እና ዜጎች በሀገራቸው ጉዳይ እኩል ተሳታፊ ሲሆኑ አንድነት ይጠናከራል። ነገር ግን የሃብት ክፍፍል ኢ-ፍትሃዊ ከሆነ ለግጭት መንስኤ ሊሆን ይችላል። ሁሉም የልማት ስራዎች የህዝቡን ፍላጎት መሰረት ያደረጉ መሆን አለባቸው።")
        print("Prediction Result:", result)
    except Exception as e:
        print(f"Error initializing model or predicting: {e}")
