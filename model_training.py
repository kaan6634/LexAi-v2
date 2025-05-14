import os
# CUDA hatalarını hemen raporlamak için:
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedTokenizerFast
import json
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Model tanımı
class TransformerQA(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(TransformerQA, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(0.1)
        self.qa_outputs = nn.Linear(d_model, 2)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids) + self.positional_encoding[:, :input_ids.size(1), :]
        key_padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.dropout(x)
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


# Dataset sınıfı
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa = self.data[idx]
        context = qa["context"]
        question = qa["qas"][0]["question"]
        answer = qa["qas"][0]["answers"][0]

        answer_text = answer["text"]
        start_char = answer.get("answer_start", None)

        if start_char is None:
            return self.__getitem__((idx + 1) % len(self.data))

        end_char = start_char + len(answer_text)

        encoding = self.tokenizer(
            question,
            context,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        offset_mapping = encoding["offset_mapping"].squeeze()

        start_token, end_token = 0, 0
        for i, (start, end) in enumerate(offset_mapping.tolist()):
            if start is not None and end is not None:
                if start <= start_char < end:
                    start_token = i
                if start < end_char <= end:
                    end_token = i

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "start_positions": torch.tensor(start_token),
            "end_positions": torch.tensor(end_token)
        }


# Tokenizer'ı JSON dosyasından yükleme
tokenizer = PreTrainedTokenizerFast(tokenizer_file=r"trained_tokenizer.json")

# Padding token'ı yoksa ekle
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# Veri yükleme
with open("final_enriched_dataset_original_format.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Hyperparametreler
vocab_size = len(tokenizer)
d_model = 256
nhead = 8
num_layers = 4
batch_size = 4
epochs = 100
patience = 5
best_val_loss = float("inf")
patience_counter = 0

# Model oluşturma
model = TransformerQA(vocab_size, d_model, nhead, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()

# Dataset ve DataLoader
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]
train_dataset = QADataset(train_data, tokenizer)
val_dataset = QADataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# F1 skoru hesaplama
def compute_f1(pred_start, pred_end, true_start, true_end):
    f1s = []
    for ps, pe, ts, te in zip(pred_start, pred_end, true_start, true_end):
        pred_span = set(range(ps, pe + 1))
        true_span = set(range(ts, te + 1))
        if len(pred_span) == 0 or len(true_span) == 0:
            f1s.append(0.0)
            continue
        overlap = len(pred_span & true_span)
        precision = overlap / len(pred_span)
        recall = overlap / len(true_span)
        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1 = 2 * precision * recall / (precision + recall)
            f1s.append(f1)
    return sum(f1s) / len(f1s)


# Eğitim
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_pos = batch["start_positions"].to(device)
        end_pos = batch["end_positions"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(input_ids, attention_mask)

        loss_start = loss_fn(start_logits, start_pos)
        loss_end = loss_fn(end_logits, end_pos)
        loss = (loss_start + loss_end) / 2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"[Epoch {epoch+1}] Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    f1_scores = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)

            start_logits, end_logits = model(input_ids, attention_mask)

            loss_start = loss_fn(start_logits, start_pos)
            loss_end = loss_fn(end_logits, end_pos)
            loss = (loss_start + loss_end) / 2
            val_loss += loss.item()

            pred_start = torch.argmax(start_logits, dim=1).cpu().tolist()
            pred_end = torch.argmax(end_logits, dim=1).cpu().tolist()
            true_start = start_pos.cpu().tolist()
            true_end = end_pos.cpu().tolist()

            f1 = compute_f1(pred_start, pred_end, true_start, true_end)
            f1_scores.append(f1)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {avg_f1:.4f}")

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "transformer_qa_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break


# Grafik çizimi
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
