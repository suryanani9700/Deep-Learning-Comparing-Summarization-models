import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.multiprocessing

torch.multiprocessing.set_start_method('spawn', force=True)

file_path = '/home/016701321@SJSUAD/news_summary_more.csv'
df = pd.read_csv(file_path)

df = df.sample(1000)

class SummarizationDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128, summary_len=32):
        self.data = []
        for idx, row in data.iterrows():
            text = row['text']
            summary = row['headlines']

            text_encoding = tokenizer(
                "summarize: " + text,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            summary_encoding = tokenizer(
                summary,
                max_length=summary_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.data.append({
                'input_ids': text_encoding.input_ids.squeeze(),
                'attention_mask': text_encoding.attention_mask.squeeze(),
                'labels': summary_encoding.input_ids.squeeze()
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

device = torch.device("cpu")
model.to(device)

dataset = SummarizationDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 2
accumulation_steps = 4 
model.train()

for epoch in range(epochs):
    print(f"Starting Epoch {epoch+1}/{epochs}")
    for batch_idx, batch in enumerate(loader, 1):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (batch_idx % accumulation_steps == 0) or (batch_idx == len(loader)):
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item()}")

    print(f"Finished Epoch {epoch+1}/{epochs}\n")

model.save_pretrained('./my_bart_model')
tokenizer.save_pretrained('./my_bart_model')

print("Training complete and model saved to './my_bart_model'")
