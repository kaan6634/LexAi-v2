from groq import Groq
from deep_translator import GoogleTranslator

# Groq API istemcisi
client = Groq(api_key="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")

def simplify_and_translate(text):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Bu cevabı daha açık ve sade bir şekilde yeniden yaz."},
            {"role": "user", "content": text}
        ]
    )
    cleaned_output = response.choices[0].message.content.strip()
    translated_output = GoogleTranslator(source='en', target='tr').translate(cleaned_output)
    return translated_output.replace(" .", ".").replace(" ,", ",").strip()

if __name__ == "__main__":
    input_text = input("Sadeleştirilecek metni girin: ").strip()
    result = simplify_and_translate(input_text)
    print("\n--- Çevirisi ve Sadeleştirilmiş Metin ---")
    print(result)
