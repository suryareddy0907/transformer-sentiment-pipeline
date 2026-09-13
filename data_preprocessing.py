import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_and_preprocess_data(file_path="data.csv"):
    df = pd.read_csv(file_path)
    df_filtered = df[df['Sentiment'] != 'neutral'].copy()
    df_filtered['label'] = df_filtered['Sentiment'].map({'positive': 1, 'negative': 0})
    
    texts = df_filtered['Sentence'].values
    labels = df_filtered['label'].values

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = FinancialSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialSentimentDataset(val_texts, val_labels, tokenizer)

    return train_dataset, val_dataset, tokenizer