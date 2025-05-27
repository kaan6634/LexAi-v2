import json
import fitz  # PyMuPDF
from groq import Groq
from deep_translator import GoogleTranslator


# GROQ API key
client = Groq(
    api_key="api_key_here"  # Replace with your actual API key
)

def pdf_to_context(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for i, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {i+1} ---\n{text}"
    doc.close()
    return full_text

def generate_qa_from_context(content):
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that generates questions and answers based on the provided content."
                },
                {
                    "role": "user",
                    "content": f"Please generate relevant questions and answers based on the following content:\n\n{content}"
                }
            ],
            model="llama3-8b-8192",
            temperature=0.3,
            max_tokens=800
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error during API call: {e}")
        return None

def parse_qa_output(text_response, start_id=300000):
    lines = text_response.splitlines()
    qa_pairs = []
    question = ""
    answer = ""
    current_id = start_id

    for line in lines:
        if line.strip().lower().startswith("q"):
            if question and answer:
                q_tr = GoogleTranslator(source='en', target='tr').translate(question.strip())
                a_tr = GoogleTranslator(source='en', target='tr').translate(answer.strip())

                qa_pairs.append({
                    "id": str(current_id),
                    "is_impossible": False,
                    "question": q_tr,
                    "answers": [
                        {
                            "text": a_tr,
                            "answer_start": None,
                            "answer_end": None
                        }
                    ]
                })
                current_id += 1
                answer = ""
            question = line.split(":", 1)[1] if ":" in line else line
        elif line.strip().lower().startswith("a"):
            answer = line.split(":", 1)[1] if ":" in line else line
        else:
            if answer != "":
                answer += " " + line.strip()

    if question and answer:
        q_tr = GoogleTranslator(source='en', target='tr').translate(question.strip())
        a_tr = GoogleTranslator(source='en', target='tr').translate(answer.strip())
        qa_pairs.append({
            "id": str(current_id),
            "is_impossible": False,
            "question": q_tr,
            "answers": [
                {
                    "text": a_tr,
                    "answer_start": None,
                    "answer_end": None
                }
            ]
        })

    return qa_pairs

if __name__ == "__main__":
    # PDF oku
    pdf_path = "LexAI_Sunum.pdf"
    context = pdf_to_context(pdf_path)

    print("Generating Q&A from PDF content...")
    qa_output = generate_qa_from_context(context)

    if qa_output:

        qa_pairs = parse_qa_output(qa_output)
        print(qa_pairs  )

        # 1. Mevcut dataset'i yükle
        with open("turkish_QA_law_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)

        # 2. Yeni context + qas yapısını ekle
        new_entry = {
            "context": context,
            "qas": qa_pairs
        }
        dataset.append(new_entry)

        # 3. Dosyaya tekrar yaz
        with open("turkish_QA_law_dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        print("✅ Yeni QA'ler 'turkish_QA_law_dataset.json' dosyasına eklendi.")
    else:
        print("❌ QA üretilemedi.")

