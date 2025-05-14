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
from groq import Groq
from deep_translator import GoogleTranslator



client = Groq(api_key="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")


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

        start_logits, end_logits = model(input_ids, attention_mask)
        start_logits = start_logits.squeeze(0)
        end_logits = end_logits.squeeze(0)

        context_indices = torch.where(token_type_ids == 1)[0]

        start_index = context_indices[torch.argmax(start_logits[context_indices])].item()
        end_index = context_indices[torch.argmax(end_logits[context_indices])].item()

        # Swap if start > end
        if start_index > end_index:
            start_index, end_index = end_index, start_index

        # Get character positions
        start_char = offset_mapping[start_index][0].item()
        end_char = offset_mapping[end_index][1].item()

        # Bazı token'lar boş (örneğin padding) olabilir
        if start_char is None or end_char is None or start_char >= end_char:
            return "Cevap bulunamadı."

        # Cevabı al
        answer = context[start_char:end_char].strip()

        # Cümle tamamlanmamışsa genişletmeyi dene (nokta, virgül vs.)
        while end_char < len(context) and context[end_char] not in ".!?":
            end_char += 1
        if end_char < len(context):
            answer = context[start_char:end_char + 1].strip()

        return answer





question = "Çalışanın sözleşmesinin feshedilmesinin ana nedeni hangi gerekçelere dayandırılmaktadır"

context = """Davacı İsteminin Özeti:Davacı vekili, davalıya ait işyerinde Toplu İş Sözleşmesi kapsamında kabin memuru olarak çalışan ve sözleşmedeki hükümler nedeni ile iş güvencesinden yararlanan davacı işçinin iş sözleşmesinin geçerli neden olmadan feshedildiğini belirterek 4857 sayılı İş Kanunu’nun 18 ve devamı maddeleri uyarınca feshin geçersizliğine ve işe iadesine karar verilmesini talep etmiştir.Davalı Cevabının Özeti:Davalı işveren vekili, uçuş personeli olan davacının iş akdinin sık sık rapor alması nedeni ile uçuş operasyonunun aksamasına, iş gücü planlamasının olumsuz etkilenmesine, iş arkadaşlarının uçuş programlarında değişiklik yapılmasına neden olduğunu, bunun işyerinde olumsuzluklara yol açtığını, iş sözleşmesinin İş Kanunun 17. 18. ve 19. Maddeleri gereğince işçinin davranışlarından ve yeterliliğinden kaynaklanan geçerli sebeple kıdem ve ihbar tazminat ödenmek suretiyle feshedildiğini, davanın reddi gerektiğini savunmuştur. Yerel Mahkeme Kararının Özeti:Mahkemece yapılan yargılama sonunda, davacının almış olduğu raporlar resmi sağlık kurumlarınca verilmiş raporlar olup davacı rahatsızlığı dolayısıyla aldığı rapor sonucu işe gelmediği, 4857 Sayılı İş Kanununun 18 maddesinde fesih için geçerli nedenin bulunmadığı haller arasında sayılan hastalık ve kaza nedeniyle işçinin ihbar öneline 6 hafta eklenmesiyle bulunan süreyi aşmayan istirahat süresi içerisinde aldığı raporların devamsızlık sayılmadığı, davacının 2011 yılında aldığı rapor süresinin, davacının işyerindeki kıdemi dikkate alındığında ihbar öneline ilaveten 6 haftayı aşmadığı, geçerli feshin kanıtlanamadığı gerekçesi ile davanın kabulüne karar verilmiştir. Temyiz:Karar davalı vekili tarafından temyiz edilmiştir. Gerekçe:4857 Sayılı İş Kanunu’nun 18.maddesinde iş sözleşmesinin işveren tarafından işçinin yeterliliğinden veya davranışlarından kaynaklanan geçerli bir sebebe dayanılarak feshedilebileceği düzenlenmiştir. Söz konusu geçerli sebepler İş Kanunu’nun 25.maddesinde belirtilen derhal fesih için öngörülen nedenler yanında, bu nitelikte olmamakla birlikte, işçinin ve işyerinin normal yürüyüşünü olumsuz etkileyen hallerdir.İşçinin yeterliliğinden veya davranışlarından kaynaklanan sebepler ancak işyerinde olumsuzluklara yol açması halinde fesih için geçerli sebep olabilirler. İş ilişkisinin sürdürülmesinin işveren açısından önemli ve makul ölçüler içinde beklenemeyeceği durumlarda, feshin geçerli sebeplere dayandığı kabul edilmelidir.İş Kanunu’nun gerekçesinde hangi hallerin işçinin yetersizliği nedeniyle geçerli fesih hakkı bahşedeceği örnek kabilinden sayılmış olup bunlardan biri de sık sık hastalanarak rapor almadır. Sık sık rapor alma halinde, işveren aralıklı da olsa işçinin iş görme ediminden faydalanamayacaktır. Sık sık hastalanan ve rapor alan işçinin, bu nedenle devamsızlığının işyerinde olumsuzluklara yol açacağı açık bir olgudur. İş Kanunu’nun gerekçesinde sık sık hastalanmanın yeterlilikten kaynaklanan neden olarak örnek kabilinden sayılması, işyerinde olumsuzluklara yol açtığının kabul edilmesindendir.İşveren 4857 sayılı İş Kanunu’nun 18/3. f maddesi uyarınca aynı kanunun 25/I.b maddesi uyarınca önele ilaveten altı haftalık bekleme süresi içinde işçinin iş sözleşmesini feshedemez. Ancak işçinin aralıklı olmak üzere sık sık rapor alması bu kapsama girmez. Sık sık rapor alması durumunda toplam raporlu olduğu süre, bekleme süresi içinde kalsa bile, sık sık rapor alması işyerinde olumsuzluklara yol açmış ise, işçinin iş sözleşmesi bildirimli veya süreli olarak feshedilebilir. Bu durumda fesih geçerli nedene dayanmaktadır.Dosya içeriğine göre uçucu personel olan ve Toplu İş Sözleşme hükümleri uyarında 4857 sayılı İş Kanunu’nun iş güvencesi hükümlerinden yararlanan davacı işçinin her yıl rapor aldığı ve 2011 yılında da bir çok kez rahatsızlığı nedeni ile aralıklı da olsa rapor aldığı anlaşılmaktadır. Sık sık rapor almada bekleme süresinin aranmasına gerek yoktur. 4857 sayılı İş Kanunu’nun 18/3.f bendindeki geçersizlik bir kez rapor alma halinde bekleme süresi içinde fesihte sözkonusudur. Bu nedenle mahkemenin bu gerekçesi yerinde değildir. Davacının sık sık rapor alması nedeni ile önceden hazırlanan uçuş programlarının aksayacağı, iş gücü planlamasının değişeceği açıktır. Davacının sık sık rapor alması işyerinde olumsuzluklara yol açmış ve iş ilişkisinin işveren açısından çekilmez hale getirmiştir. İşverenin feshi, davacının yeterliliğinden kaynaklanan geçerli nedene dayandığından, davanın reddi gerekir. Yazılı gerekçe ile davanın kabulü hatalıdır.4857 sayılı İş Yasasının 20/3 maddesi uyarınca Dairemizce aşağıdaki şekilde karar verilmiştir. Sonuç:Yukarda açıklanan gerekçe ile;1. Mahkemenin kararının BOZULARAK ORTADAN KALDIRILMASINA,2. Davanın REDDİNE,3. Harç peşin alındığından yeniden alınmasına yer olmadığına,4. Davacının yaptığı yargılama giderinin üzerinde bırakılmasına, davalının yaptığı yargılama gideri 70.00 TL'nin davacıdan tahsili ile davalıya ödenmesine,5. Karar tarihinde yürürlükte bulunan tarifeye göre 1.320 TL ücreti vekaletin davacıdan alınarak davalıya verilmesine,6. Peşin alınan temyiz harcının isteği halinde davalıya iadesine,Kesin olarak 14.01.2013 tarihinde oybirliği ile karar verildi."""  # Tüm bağlamı buraya ekle



answer = predict_answer(model, tokenizer, question, context, device)



if answer == "Cevap bulunamadı (Başlangıç bitişten büyük).":
    print("Model cevap veremedi.")
    quit(1)

print("Soru:", question)
print("\n\nCevap:", answer)


# Örneğin kendi QA modelinden gelen cevap:
answer = predict_answer(model, tokenizer, question, context, device)
print("Soru:", question)
print("\n\nCevap:", answer)

# Groq API istemcisi (güvenli kullanım için .env'den al!)


# Groq modelinden cevabı düzenlemesini iste
response = client.chat.completions.create(
    model="llama3-8b-8192",
    messages=[
        {"role": "system",   "content": "Rephrase the provided text to make it clearer and easier to understand without shortening or summarizing it."},
        {"role": "user", "content": f"Cevap: {answer}"}
    ],
)

# Düzenlenmiş cevabı al
cleaned_output = response.choices[0].message.content.strip()

print("\nGroq ile Düzenlenmiş Cevap:\n", cleaned_output)

# İngilizce cevabı Türkçeye çevir
translated_output = GoogleTranslator(source='en', target='tr').translate(cleaned_output)

# Temizleme (bazı bozuklukları düzeltmek için)
translated_output = (
    translated_output
    .replace(" .", ".")
    .replace(" ,", ",")
    .strip()
)

print("\nTürkçeye Çevrilmiş Cevap:\n", translated_output)