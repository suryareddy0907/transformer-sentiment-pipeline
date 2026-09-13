# Financial Sentiment Analysis Pipeline

A production-grade, modular PyTorch pipeline fine-tuning a bidirectional `bert-base-uncased` Transformer model for financial market binary sentiment classification (Positive vs. Negative).

## Project Architecture & Structure

```text
├── data_preprocessing.py  # Dataset loading, neutral-class filtering, and tokenization pipeline
├── train.py               # BERT-base model initialization, training loop, and evaluation
├── inference.py           # Single-sequence inference and latency benchmarking script
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

## Quickstart & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/financial-sentiment-pipeline.git](https://github.com/YOUR_USERNAME/financial-sentiment-pipeline.git)
   cd financial-sentiment-pipeline
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing & Training:**
   ```bash
   python data_preprocessing.py
   python train.py
   ```

4. **Benchmark Inference Latency:**
   ```bash
   python inference.py
   ```