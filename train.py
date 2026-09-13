import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from data_preprocessing import load_and_preprocess_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    train_dataset, val_dataset, _ = load_and_preprocess_data()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)

    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | Training Loss: {total_train_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "pytorch_model.bin")

if __name__ == "__main__":
    train()