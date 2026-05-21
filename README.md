# Amh-Domain-Sense

Amh-Domain-Sense is a dual-purpose system designed for Amharic text classification:
- **A Research Framework** for evaluating and comparing Amharic text classification architectures.
  - **Model Comparison**: Evaluating the performance of monolingual Amharic models (`rasyosef/roberta-base-amharic`) against multilingual baselines (AfriBERTa and XLM-RoBERTa).
- **A Production Engine** for backend platform inference.

## Core Functionality

The core of the system utilizes a **Multi-Head RoBERTa architecture** featuring hierarchical gating to ensure high-quality domain classification:

- **Language Gate**: Filters for Amharic text, ensuring that only relevant language data is processed.
- **Readability Gate**: Identifies clean versus broken/OCR text to maintain data quality.
- **Domain Head**: Classifies text into 7 specific domains: Education, Health, Religion, Politics, Law, General, and Finance.

## Folder Structure

```text
Amh-Domain-Sense/
├── data/
│   ├── final_training_v3.jsonl
│   └── test_inference.jsonl
├── notebooks/
├── results/
│   ├── afriberta/
│   ├── multihead-project-model/
│   ├── rasyosef_roberta/
│   └── xlm_roberta/
├── scripts/
│   ├── model_loader.py
│   ├── train-afriberta.py
│   ├── train-multihead-model.py
│   ├── train-rasyosef-base.py
│   └── train-xlm_roberta.py
├── .gitignore
└── requirements.txt
```

## Local Testing & Inference

To run domain classification locally, you can use the provided model loader utility.

1. **Run the model** by executing:
   ```bash
   python scripts/model_loader.py
   ```

2. **Test Custom Text**: To test custom Amharic text, open `scripts/model_loader.py` and modify the input string in the `result = model.predict("...")` variable located at the end of the script.

### Model Loading

The `scripts/model_loader.py` file is the primary utility for loading the model. It should be used as the entry point for integrating the model into other services or for standalone inference.

## Production Model

The hosted weights for the production model can be found on Hugging Face:
[amanfisseha/multihead-rasyosef-amharic](https://huggingface.co/amanfisseha/multihead-rasyosef-amharic)
