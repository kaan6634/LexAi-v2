import fitz  # PyMuPDF

def pdf_to_context(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for i, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {i+1} ---\n{text}"

    doc.close()
    return full_text

# Örnek kullanım
pdf_path = "LexAI_Sunum.pdf"
context = pdf_to_context(pdf_path)
print(context)
