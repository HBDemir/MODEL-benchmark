# !pip install streamlit openai pypdf pandas

import streamlit as st
import pandas as pd
import openai
import PyPDF2
import os
import shutil

st.set_page_config(page_title="MOF Toxicity MCQ Creator", layout="wide")
st.title("MOF Toxicity MCQ Benchmark Creator (PDF to MCQ)")

openai_api_key = st.text_input("OpenAI API Key", type="password")

# PDF yükleme ve klasöre kaydetme
pdf_folder = "uploaded_pdfs"
os.makedirs(pdf_folder, exist_ok=True)

uploaded_files = st.file_uploader("PDF makaleleri yükleyin (klasöre kaydedilecek)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Yüklenen PDF sayısı: {len(uploaded_files)}")
    for pdf_file in uploaded_files:
        file_path = os.path.join(pdf_folder, pdf_file.name)
        with open(file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
    st.success(f"Tüm PDF'ler '{pdf_folder}' klasörüne kaydedildi.")

# PDF'den metin çıkarma

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def extract_mof_toxicity_sentences(text):
    mof_keywords = ['MOF', 'metal-organic framework', 'nMOF', 'ZIF-8', 'UiO-66', 'MIL-101']
    tox_keywords = ['toxic', 'toxicity', 'toxicological', 'cytotoxicity', 'adverse effects', 'hazard', 'safe', 'safety']
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant = []
    for s in sentences:
        if any(mk.lower() in s.lower() for mk in mof_keywords) and any(tk.lower() in s.lower() for tk in tox_keywords):
            relevant.append(s.strip())
    return relevant

def generate_mcq(sentence, api_key):
    prompt = f"""
You are creating a scientific benchmark for MOF toxicity. Based only on the following sentence, generate a multiple-choice question (MCQ) that asks whether the specific MOF mentioned by name in the sentence (e.g., 'HKUST-1', 'UiO-66', etc.) exhibits toxic properties. The question must include the MOF's actual name as it appears in the sentence (e.g., 'Does the MOF HKUST-1 exhibit toxic properties?'). Do not use generic phrases like 'the MOF described in the sentence'. Do not ask about general concepts, implications, or hypothetical scenarios. Provide 4 options (A, B, C, D), only one of which is correct, and indicate the correct answer.

Sentence: \"{sentence}\"

Output format:
Question: ...
A) ...
B) ...
C) ...
D) ...
Correct answer: ...
"""
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.85
    )
    return response.choices[0].message.content
# MCQ üret ve göster
btn = st.button("PDF'lerden MCQ Benchmark Oluştur")
if btn:
    if not openai_api_key:
        st.warning("Lütfen OpenAI API anahtarınızı girin.")
    else:
        all_results = []
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        for pdf_path in pdf_files:
            st.write(f"İşleniyor: {os.path.basename(pdf_path)}")
            text = extract_text_from_pdf(pdf_path)
            sentences = extract_mof_toxicity_sentences(text)
            for sent in sentences:
                try:
                    mcq = generate_mcq(sent, openai_api_key)
                    all_results.append({
                        "pdf": os.path.basename(pdf_path),
                        "sentence": sent,
                        "mcq": mcq
                    })
                except Exception as e:
                    st.warning(f"Hata: {e}")
        if all_results:
            df = pd.DataFrame(all_results)
            st.success(f"{len(df)} MCQ üretildi.")
            for i, row in df.iterrows():
                st.markdown(f"### Soru {i+1} ({row['pdf']})")
                st.markdown(f"**Cümle:** {row['sentence']}")
                st.code(row['mcq'], language='markdown')
                st.markdown("---")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Sonuçları CSV olarak indir", csv, "mof_toxicity_mcq_benchmark.csv", "text/csv")
        else:
            st.info("Hiç uygun cümle bulunamadı.") 