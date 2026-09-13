import os
import json
import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification
from data_preprocessing import load_and_preprocess_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_evaluation():
    os.makedirs("assets", exist_ok=True)
    _, val_dataset, tokenizer = load_and_preprocess_data()
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    if os.path.exists("pytorch_model.bin"):
        model.load_state_dict(torch.load("pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    
    # Benchmarking Latency
    sample_text = "Operating profit margins increased by 14% quarter-over-quarter due to strong regional demand."
    inputs = tokenizer(sample_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    
    with torch.no_grad():
        start_time = time.perf_counter()
        _ = model(**inputs)
        latency_ms = (time.perf_counter() - start_time) * 1000

        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    # Model Footprint
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)

    # Save metrics JSON
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "inference_latency_ms": round(float(latency_ms), 2),
        "model_footprint_mb": round(float(model_size_mb), 2)
    }
    
    with open("assets/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save Confusion Matrix Visual
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('BERT Financial Sentiment Confusion Matrix')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=300)
    plt.close()

    print("Evaluation Complete. Metrics and Confusion Matrix saved to /assets.")

if __name__ == "__main__":
    run_evaluation()