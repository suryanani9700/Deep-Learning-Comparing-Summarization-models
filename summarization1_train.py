import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load dataset
file_path = '/home/016701321@SJSUAD/news_summary_more.csv'
df = pd.read_csv(file_path)

# Use a smaller dataset for testing
df = df.sample(1000)

# Dataset class
class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256, summary_len=64):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len
        self.summary_len = summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        summary = row['headlines']

        text_encoding = self.tokenizer(
            "summarize: " + text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': text_encoding.input_ids.squeeze(),
            'attention_mask': text_encoding.attention_mask.squeeze(),
            'labels': summary_encoding.input_ids.squeeze()
        }

# Tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# Set device to CPU
device = torch.device("cpu")
model.to(device)

# DataLoader
dataset = SummarizationDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 2
model.train()
for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    for batch_idx, batch in enumerate(loader, 1):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item()}")

    print(f"Finished Epoch {epoch+1}/{epochs}\n")

# Save model
model.save_pretrained('./my_t5_model')
tokenizer.save_pretrained('./my_t5_model')

print("Training complete and model saved to './my_t5_model'")
