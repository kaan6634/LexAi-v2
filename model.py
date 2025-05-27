import os
# CUDA hatalarını hemen raporlamak için:
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import PreTrainedTokenizerFast
import json
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

          if attention_mask is not None:
              key_padding_mask = attention_mask == 0  # [batch, seq_len]
          else:
              key_padding_mask = None

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
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="trained_tokenizer.json",
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
    model_max_length=512
)

# Padding token'ını ekleyin, eğer yoksa
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# Hyperparametreler
vocab_size = len(tokenizer)  # Burada tekrar güncelle
d_model = 256
nhead = 8
num_layers = 4
batch_size = 4
epochs = 100
patience = 5  # Early stopping için sabır
best_val_loss = float("inf")
patience_counter = 0

# Modeli oluşturma

model = TransformerQA(vocab_size, d_model, nhead, num_layers).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.CrossEntropyLoss()



def predict_answer(model, tokenizer, question, context, device, max_length=512):
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            question,
            context,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_token_type_ids=True
        )

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        offset_mapping = encoding["offset_mapping"].squeeze(0)  # [seq_len, 2]
        token_type_ids = encoding["token_type_ids"].squeeze(0)  # [seq_len]

        # Model tahmini
        start_logits, end_logits = model(input_ids, attention_mask)
        start_logits = start_logits.squeeze(0)
        end_logits = end_logits.squeeze(0)

        # Sadece context'e (token_type_id == 1) ait tokenları bul
        context_indices = torch.where(token_type_ids == 1)[0]

        # Context tokenları içinden en yüksek olasılıklı start/end index'lerini al
        start_index = context_indices[torch.argmax(start_logits[context_indices])].item()
        end_index = context_indices[torch.argmax(end_logits[context_indices])].item()

        if start_index > end_index:
            return "connection error"

        start_char = offset_mapping[start_index][0].item()
        end_char = offset_mapping[end_index][1].item()

        answer = context[start_char:end_char]
        return answer.strip()




question = "Asliye Ceza Mahkemesi ifadesi, metindeki hukuki bağlamda ne anlama gelmektedir?"

# context.txt dosyasından içeriği al
with open("all_contexts.txt", "r", encoding="utf-8") as f:
    context = f.read()

answer = predict_answer(model, tokenizer, question, context, device)
print("Soru:", question)
print("\n\nCevap:", answer)


