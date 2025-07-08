# !pip install streamlit openai pypdf pandas numpy matplotlib seaborn scikit-learn plotly kaleido sentence-transformers

import streamlit as st
import pandas as pd
import openai
import PyPDF2
import os
import shutil
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Görselleştirme modülünü import et
from benchmark_app.visualization import DirectEmbeddingVisualizer

st.set_page_config(page_title="MOF Toxicity MCQ Creator", layout="wide")
st.title("🔬 MOF Toxicity MCQ Benchmark Creator with Embedding Analysis")

# Sidebar ayarları
st.sidebar.header("⚙️ Ayarlar")
enable_embeddings = st.sidebar.checkbox("Embedding Analizini Etkinleştir", value=True)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.1)
show_individual_visualizations = st.sidebar.checkbox("Her Chunk için Ayrı Görselleştirme", value=True)
save_visualizations = st.sidebar.checkbox("Görselleştirmeleri Kaydet", value=False)

# API key
openai_api_key = st.text_input("OpenAI API Key", type="password")

# Embedding modeli cache
@st.cache_resource
def load_embedding_model():
    """Embedding modelini yükle"""
    return SentenceTransformer('all-MiniLM-L6-v2')

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

# Query for embedding analysis
if enable_embeddings:
    query_text = st.text_input("Embedding analizi için sorgu metni", 
                              value="MOF toxicity and biocompatibility assessment")

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
    mof_keywords = ['MOF', 'metal-organic framework', 'nMOF', 'ZIF-8', 'UiO-66', 'MIL-101', 'HKUST-1', 'ZIF-67', 'MOF-5']
    tox_keywords = ['toxic', 'toxicity', 'toxicological', 'cytotoxicity', 'adverse effects', 'hazard', 'safe', 'safety', 'biocompatibility']
    nontox_keywords = ['safe', 'safety', 'biocompatibility', 'non-toxic', 'harmless', 'benign']

    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    relevant = []
    for s in sentences:
        s_lower = s.lower()
        if any(mk.lower() in s_lower for mk in mof_keywords):
            # Toksik veya non-toksik kelimelerden en az biri varsa al
            if any(tk in s_lower for tk in tox_keywords) or any(ntk in s_lower for ntk in nontox_keywords):
                if len(s.strip()) > 20:
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

def process_chunk_with_embeddings(chunk_text, query_text, embedding_model, chunk_idx):
    """Chunk'ı embedding analizi ile işle"""
    
    # Embeddings oluştur
    query_embedding = embedding_model.encode(query_text)
    chunk_embedding = embedding_model.encode(chunk_text)
    
    # Cosine similarity hesapla
    similarity_score = cosine_similarity(
        query_embedding.reshape(1, -1), 
        chunk_embedding.reshape(1, -1)
    )[0][0]
    
    return {
        'query_embedding': query_embedding,
        'chunk_embedding': chunk_embedding,
        'similarity_score': similarity_score,
        'chunk_text': chunk_text
    }

# MCQ üret ve göster
btn = st.button("🚀 PDF'lerden MCQ Benchmark Oluştur")

if btn:
    if not openai_api_key:
        st.warning("Lütfen OpenAI API anahtarınızı girin.")
    else:
        # Embedding modeli ve visualizer'ı hazırla
        if enable_embeddings:
            with st.spinner("Embedding modeli yükleniyor..."):
                embedding_model = load_embedding_model()
                visualizer = DirectEmbeddingVisualizer()
                
                # Görselleştirme klasörü oluştur
                if save_visualizations:
                    viz_output_dir = "embedding_visualizations"
                    os.makedirs(viz_output_dir, exist_ok=True)
        
        all_results = []
        embedding_results = []
        all_embeddings_data = []  # Toplu görselleştirme için
        
        pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for pdf_idx, pdf_path in enumerate(pdf_files):
            pdf_name = os.path.basename(pdf_path)
            status_text.text(f"İşleniyor: {pdf_name}")
            
            # PDF'den metin çıkar
            text = extract_text_from_pdf(pdf_path)
            sentences = extract_mof_toxicity_sentences(text)
            
            st.subheader(f"📄 {pdf_name}")
            st.write(f"Bulunan ilgili cümle sayısı: {len(sentences)}")
            
            # Her cümle (chunk) için işlem
            for sent_idx, sent in enumerate(sentences):
                try:
                    # MCQ oluştur
                    mcq = generate_mcq(sent, openai_api_key)
                    
                    # Embedding analizi
                    embedding_result = None
                    if enable_embeddings:
                        embedding_result = process_chunk_with_embeddings(
                            sent, query_text, embedding_model, sent_idx + 1
                        )
                        
                        # Similarity threshold kontrolü
                        if embedding_result['similarity_score'] >= similarity_threshold:
                            
                            # Bireysel görselleştirme
                            if show_individual_visualizations:
                                st.subheader(f"🔍 Embedding Analizi - {pdf_name} - Cümle {sent_idx + 1}")
                                
                                # Similarity score göster
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Similarity Score", f"{embedding_result['similarity_score']:.4f}")
                                with col2:
                                    if embedding_result['similarity_score'] >= 0.8:
                                        st.success("🟢 Yüksek Benzerlik")
                                    elif embedding_result['similarity_score'] >= 0.6:
                                        st.warning("🟡 Orta Benzerlik")
                                    else:
                                        st.error("🔴 Düşük Benzerlik")
                                with col3:
                                    st.info(f"Threshold: {similarity_threshold}")
                                
                                # Tek chunk görselleştirme
                                save_path = f"{viz_output_dir}/{pdf_name}_chunk_{sent_idx + 1}.html" if save_visualizations else None
                                fig = visualizer.visualize_single_comparison(
                                    embedding_result['query_embedding'],
                                    embedding_result['chunk_embedding'],
                                    sent,
                                    embedding_result['similarity_score'],
                                    sent_idx + 1,
                                    save_path=save_path
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Matplotlib görselleştirme
                                save_path_matplotlib = f"{viz_output_dir}/{pdf_name}_chunk_{sent_idx + 1}_matplotlib.png" if save_visualizations else None
                                matplotlib_fig = visualizer.visualize_matplotlib_comparison(
                                    embedding_result['query_embedding'],
                                    embedding_result['chunk_embedding'],
                                    sent,
                                    embedding_result['similarity_score'],
                                    sent_idx + 1,
                                    save_path=save_path_matplotlib
                                )
                                st.pyplot(matplotlib_fig)
                                
                                # Chunk metnini göster
                                with st.expander("📝 Chunk Metni"):
                                    st.write(sent)
                                
                                st.markdown("---")
                            
                            # Toplu analiz için veri topla
                            all_embeddings_data.append({
                                'query_embedding': embedding_result['query_embedding'],
                                'chunk_embedding': embedding_result['chunk_embedding'],
                                'chunk_text': sent,
                                'similarity_score': embedding_result['similarity_score'],
                                'pdf_name': pdf_name,
                                'chunk_idx': sent_idx + 1
                            })
                            
                            embedding_results.append({
                                'pdf': pdf_name,
                                'chunk_index': sent_idx + 1,
                                'similarity_score': embedding_result['similarity_score'],
                                'chunk_text': sent
                            })
                    
                    # MCQ sonuçlarını kaydet
                    all_results.append({
                        "pdf": pdf_name,
                        "sentence": sent,
                        "mcq": mcq,
                        "similarity_score": embedding_result['similarity_score'] if embedding_result else None
                    })
                    
                except Exception as e:
                    st.warning(f"Hata: {e}")
            
            # Progress bar güncelle
            progress_bar.progress((pdf_idx + 1) / len(pdf_files))
        
        status_text.text("✅ İşlem tamamlandı!")
        
        # Sonuçları göster
        if all_results:
            st.success(f"🎉 Toplam {len(all_results)} MCQ üretildi!")
            
            # Embedding özeti ve toplu görselleştirme
            if enable_embeddings and all_embeddings_data:
                st.subheader("📊 Embedding Analizi Özeti")
                
                # Toplu görselleştirme oluştur
                if len(all_embeddings_data) > 1:
                    st.subheader("🔄 Toplu Embedding Analizi")
                    
                    # Tüm karşılaştırmalar için HTML raporu oluştur
                    if save_visualizations:
                        query_embedding = all_embeddings_data[0]['query_embedding']
                        chunk_embeddings = [item['chunk_embedding'] for item in all_embeddings_data]
                        chunk_texts = [item['chunk_text'] for item in all_embeddings_data]
                        similarity_scores = [item['similarity_score'] for item in all_embeddings_data]
                        
                        html_report = visualizer.visualize_all_comparisons(
                            query_embedding, 
                            chunk_embeddings, 
                            chunk_texts, 
                            similarity_scores,
                            query_text=query_text,
                            output_dir=viz_output_dir
                        )
                        st.success(f"📁 Detaylı HTML raporu '{viz_output_dir}' klasöründe oluşturuldu!")
                
                # İstatistikler
                embedding_df = pd.DataFrame(embedding_results)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ortalama Similarity", f"{embedding_df['similarity_score'].mean():.4f}")
                with col2:
                    st.metric("Maksimum Similarity", f"{embedding_df['similarity_score'].max():.4f}")
                with col3:
                    st.metric("Minimum Similarity", f"{embedding_df['similarity_score'].min():.4f}")
                with col4:
                    st.metric("Threshold Üstü", f"{len(embedding_df[embedding_df['similarity_score'] >= similarity_threshold])}")
                
                # Embedding sonuçları tablosu
                st.subheader("📋 Embedding Sonuçları")
                st.dataframe(embedding_df)
            
            # MCQ sonuçlarını göster
            st.subheader("📝 Üretilen MCQ'lar")
            
            # Similarity skoruna göre sırala (eğer varsa)
            df = pd.DataFrame(all_results)
            if enable_embeddings:
                df = df.sort_values('similarity_score', ascending=False, na_position='last')
            
            for i, row in df.iterrows():
                st.markdown(f"### Soru {i+1} ({row['pdf']})")
                if enable_embeddings and row['similarity_score'] is not None:
                    st.markdown(f"**Similarity Score:** {row['similarity_score']:.4f}")
                st.markdown(f"**Cümle:** {row['sentence']}")
                st.code(row['mcq'], language='markdown')
                st.markdown("---")
            
            # CSV indirme
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Sonuçları CSV olarak indir", 
                csv, 
                "mof_toxicity_mcq_benchmark_with_embeddings.csv", 
                "text/csv"
            )
            
            # Embedding sonuçlarını CSV olarak indir
            if enable_embeddings and embedding_results:
                embedding_csv = pd.DataFrame(embedding_results).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Embedding Sonuçlarını CSV olarak indir", 
                    embedding_csv, 
                    "embedding_analysis_results.csv", 
                    "text/csv"
                )
        else:
            st.info("Hiç uygun cümle bulunamadı.")