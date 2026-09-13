import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def benchmark_inference():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)
    model.eval()

    sample_text = "Operating profit margins increased by 14% quarter-over-quarter due to strong regional demand."
    inputs = tokenizer(sample_text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(**inputs)
        end = time.perf_counter()

    latency_ms = (end - start) * 1000
    prediction = torch.argmax(outputs.logits, dim=1).item()
    label_map = {0: "Negative", 1: "Positive"}

    print(f"Predicted Sentiment: {label_map[prediction]}")
    print(f"Inference Latency: {latency_ms:.2f} ms")

if __name__ == "__main__":
    benchmark_inference()