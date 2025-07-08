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
from visualization import DirectEmbeddingVisualizer

st.set_page_config(page_title="MOF Toxicity MCQ Creator", layout="wide")
st.title("🔬 MOF Toxicity MCQ Benchmark Creator with Embedding Analysis")

# Sidebar ayarları
st.sidebar.header("⚙️ Ayarlar")
enable_embeddings = st.sidebar.checkbox("Embedding Analizini Etkinleştir", value=True)
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.5, 0.1)
show_individual_visualizations = st.sidebar.checkbox("Her Chunk için Ayrı Görselleştirme", value=True)
save_visualizations = st.sidebar.checkbox("Görselleştirmeleri Kaydet", value=False)

# QUERY ÇEŞİTLENDİRME AYARLARI
st.sidebar.header("🔍 Query Çeşitlendirme")
enable_query_diversification = st.sidebar.checkbox("Query Çeşitlendirmeyi Etkinleştir", value=False)
num_diversified_queries = st.sidebar.slider("Çeşitlendirme Sayısı", 1, 10, 3)
diversification_method = st.sidebar.selectbox(
    "Çeşitlendirme Yöntemi", 
    ["Automatic (GPT)", "Manual", "Hybrid"]
)

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

# QUERY ÇEŞİTLENDİRME FONKSİYONLARI
def generate_diversified_queries(base_query, num_queries, api_key):
    """GPT kullanarak query çeşitlendirme"""
    prompt = f"""
    Given the base query: "{base_query}"
    
    Generate {num_queries} diverse variations of this query that maintain the same semantic meaning but use different:
    - Terminology (synonyms, technical terms, alternative phrases)
    - Sentence structures
    - Perspectives (clinical, research, regulatory, etc.)
    - Specificity levels (more specific or more general)
    
    Each variation should be relevant for embedding-based similarity search in scientific literature.
    
    Return only the queries, one per line, numbered:
    """
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        queries = []
        lines = response.choices[0].message.content.strip().split('\n')
        for line in lines:
            if line.strip() and any(char.isalpha() for char in line):
                # Numaralandırmayı temizle
                clean_query = line.strip()
                if clean_query[0].isdigit():
                    clean_query = clean_query.split('.', 1)[-1].strip()
                    clean_query = clean_query.split(')', 1)[-1].strip()
                queries.append(clean_query)
        
        return queries[:num_queries]
        
    except Exception as e:
        st.error(f"Query çeşitlendirme hatası: {e}")
        return [base_query]

def get_predefined_query_variations():
    """Önceden tanımlanmış query çeşitlendirmeleri"""
    return [
        "MOF toxicity and biocompatibility assessment",
        "Metal-organic framework cytotoxicity evaluation",
        "Nanostructured MOF safety and hazard analysis",
        "Biocompatible MOF materials toxicological screening",
        "MOF-based drug delivery safety profile",
        "Toxicity mechanisms of metal-organic frameworks",
        "In vitro and in vivo MOF biocompatibility studies",
        "MOF nanoparticle adverse effects assessment",
        "Safe MOF design for biomedical applications",
        "MOF material safety and regulatory considerations"
    ]

def calculate_multi_query_similarity(chunk_text, queries, embedding_model):
    """Birden fazla query ile benzerlik hesapla"""
    chunk_embedding = embedding_model.encode(chunk_text)
    similarities = []
    
    for query in queries:
        query_embedding = embedding_model.encode(query)
        similarity = cosine_similarity(
            query_embedding.reshape(1, -1), 
            chunk_embedding.reshape(1, -1)
        )[0][0]
        similarities.append(similarity)
    
    return {
        'similarities': similarities,
        'max_similarity': max(similarities),
        'avg_similarity': np.mean(similarities),
        'best_query_idx': np.argmax(similarities)
    }

# Query for embedding analysis
if enable_embeddings:
    base_query = st.text_input("Embedding analizi için temel sorgu metni", 
                              value="MOF toxicity and biocompatibility assessment")
    
    # Query çeşitlendirme seçenekleri
    if enable_query_diversification:
        st.subheader("🔍 Query Çeşitlendirme")
        
        if diversification_method == "Automatic (GPT)":
            if st.button("🤖 Otomatik Query Çeşitlendirme"):
                if openai_api_key:
                    with st.spinner("Queries çeşitlendiriliyor..."):
                        diversified_queries = generate_diversified_queries(
                            base_query, num_diversified_queries, openai_api_key
                        )
                        st.session_state.diversified_queries = diversified_queries
                else:
                    st.warning("OpenAI API anahtarı gerekli!")
        
        elif diversification_method == "Manual":
            st.info("Manuel query'leri aşağıya ekleyin:")
            manual_queries = []
            for i in range(num_diversified_queries):
                query = st.text_input(f"Query {i+1}", key=f"manual_query_{i}")
                if query:
                    manual_queries.append(query)
            st.session_state.diversified_queries = manual_queries
        
        elif diversification_method == "Hybrid":
            predefined = get_predefined_query_variations()
            selected_queries = st.multiselect(
                "Önceden tanımlanmış queries'den seçin:",
                predefined,
                default=predefined[:num_diversified_queries]
            )
            st.session_state.diversified_queries = selected_queries
        
        # Çeşitlendirilen queries'leri göster
        if 'diversified_queries' in st.session_state:
            st.subheader("📋 Çeşitlendirilen Queries")
            for i, query in enumerate(st.session_state.diversified_queries):
                st.write(f"{i+1}. {query}")

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

def process_chunk_with_embeddings(chunk_text, queries, embedding_model, chunk_idx):
    """Chunk'ı embedding analizi ile işle (çoklu query desteği)"""
    
    if isinstance(queries, str):
        queries = [queries]
    
    # Multi-query similarity hesapla
    multi_query_result = calculate_multi_query_similarity(chunk_text, queries, embedding_model)
    
    # En iyi query'nin embedding'ini al
    best_query = queries[multi_query_result['best_query_idx']]
    query_embedding = embedding_model.encode(best_query)
    chunk_embedding = embedding_model.encode(chunk_text)
    
    return {
        'query_embedding': query_embedding,
        'chunk_embedding': chunk_embedding,
        'similarity_score': multi_query_result['max_similarity'],
        'avg_similarity': multi_query_result['avg_similarity'],
        'best_query': best_query,
        'all_similarities': multi_query_result['similarities'],
        'queries': queries,
        'chunk_text': chunk_text
    }

# MCQ üret ve göster
btn = st.button("🚀 PDF'lerden MCQ Benchmark Oluştur")

if btn:
    if not openai_api_key:
        st.warning("Lütfen OpenAI API anahtarınızı girin.")
    else:
        # Query'leri hazırla
        if enable_embeddings:
            if enable_query_diversification and 'diversified_queries' in st.session_state:
                queries_to_use = st.session_state.diversified_queries
                if base_query not in queries_to_use:
                    queries_to_use = [base_query] + queries_to_use
            else:
                queries_to_use = [base_query]
            
            st.info(f"Kullanılacak query sayısı: {len(queries_to_use)}")
            
            with st.spinner("Embedding modeli yükleniyor..."):
                embedding_model = load_embedding_model()
                visualizer = DirectEmbeddingVisualizer()
                
                # Görselleştirme klasörü oluştur
                if save_visualizations:
                    viz_output_dir = "embedding_visualizations"
                    os.makedirs(viz_output_dir, exist_ok=True)
        
        all_results = []
        embedding_results = []
        all_embeddings_data = []
        
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
                            sent, queries_to_use, embedding_model, sent_idx + 1
                        )
                        
                        # Similarity threshold kontrolü
                        if embedding_result['similarity_score'] >= similarity_threshold:
                            
                            # Bireysel görselleştirme
                            if show_individual_visualizations:
                                st.subheader(f"🔍 Embedding Analizi - {pdf_name} - Cümle {sent_idx + 1}")
                                
                                # Çoklu query sonuçlarını göster
                                if enable_query_diversification:
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Max Similarity", f"{embedding_result['similarity_score']:.4f}")
                                    with col2:
                                        st.metric("Avg Similarity", f"{embedding_result['avg_similarity']:.4f}")
                                    with col3:
                                        st.metric("Best Query", f"#{embedding_result['best_query'].split()[:3]}")
                                    with col4:
                                        if embedding_result['similarity_score'] >= 0.8:
                                            st.success("🟢 Yüksek")
                                        elif embedding_result['similarity_score'] >= 0.6:
                                            st.warning("🟡 Orta")
                                        else:
                                            st.error("🔴 Düşük")
                                    
                                    # Tüm query'lerin skorlarını göster
                                    query_scores_df = pd.DataFrame({
                                        'Query': queries_to_use,
                                        'Similarity': embedding_result['all_similarities']
                                    }).sort_values('Similarity', ascending=False)
                                    st.dataframe(query_scores_df)
                                    
                                    # En iyi query'yi vurgula
                                    st.info(f"**En iyi query:** {embedding_result['best_query']}")
                                
                                else:
                                    # Tek query durumu
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
                                
                                # Görselleştirme
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
                                'avg_similarity': embedding_result['avg_similarity'],
                                'best_query': embedding_result['best_query'],
                                'pdf_name': pdf_name,
                                'chunk_idx': sent_idx + 1
                            })
                            
                            embedding_results.append({
                                'pdf': pdf_name,
                                'chunk_index': sent_idx + 1,
                                'max_similarity_score': embedding_result['similarity_score'],
                                'avg_similarity_score': embedding_result['avg_similarity'],
                                'best_query': embedding_result['best_query'],
                                'chunk_text': sent
                            })
                    
                    # MCQ sonuçlarını kaydet
                    result_dict = {
                        "pdf": pdf_name,
                        "sentence": sent,
                        "mcq": mcq,
                        "max_similarity_score": embedding_result['similarity_score'] if embedding_result else None,
                        "avg_similarity_score": embedding_result['avg_similarity'] if embedding_result else None,
                        "best_query": embedding_result['best_query'] if embedding_result else None
                    }
                    all_results.append(result_dict)
                    
                except Exception as e:
                    st.warning(f"Hata: {e}")
            
            # Progress bar güncelle
            progress_bar.progress((pdf_idx + 1) / len(pdf_files))
        
        status_text.text("✅ İşlem tamamlandı!")
        
        # Sonuçları göster
        if all_results:
            st.success(f"🎉 Toplam {len(all_results)} MCQ üretildi!")
            
            # Embedding özeti
            if enable_embeddings and all_embeddings_data:
                st.subheader("📊 Embedding Analizi Özeti")
                
                # İstatistikler
                embedding_df = pd.DataFrame(embedding_results)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Ortalama Max Similarity", f"{embedding_df['max_similarity_score'].mean():.4f}")
                with col2:
                    st.metric("Ortalama Avg Similarity", f"{embedding_df['avg_similarity_score'].mean():.4f}")
                with col3:
                    st.metric("Maksimum Similarity", f"{embedding_df['max_similarity_score'].max():.4f}")
                with col4:
                    st.metric("Threshold Üstü", f"{len(embedding_df[embedding_df['max_similarity_score'] >= similarity_threshold])}")
                
                # Embedding sonuçları tablosu
                st.subheader("📋 Embedding Sonuçları")
                st.dataframe(embedding_df)
            
            # MCQ sonuçlarını göster
            st.subheader("📝 Üretilen MCQ'lar")
            
            # Similarity skoruna göre sırala
            df = pd.DataFrame(all_results)
            if enable_embeddings:
                df = df.sort_values('max_similarity_score', ascending=False, na_position='last')
            
            for i, row in df.iterrows():
                st.markdown(f"### Soru {i+1} ({row['pdf']})")
                if enable_embeddings and row['max_similarity_score'] is not None:
                    st.markdown(f"**Max Similarity:** {row['max_similarity_score']:.4f}")
                    if 'avg_similarity_score' in row and row['avg_similarity_score'] is not None:
                        st.markdown(f"**Avg Similarity:** {row['avg_similarity_score']:.4f}")
                    if 'best_query' in row and row['best_query'] is not None:
                        st.markdown(f"**Best Query:** {row['best_query']}")
                st.markdown(f"**Cümle:** {row['sentence']}")
                st.code(row['mcq'], language='markdown')
                st.markdown("---")
            
            # CSV indirme
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Sonuçları CSV olarak indir", 
                csv, 
                "mof_toxicity_mcq_benchmark_with_diversified_queries.csv", 
                "text/csv"
            )
            
            # Embedding sonuçlarını CSV olarak indir
            if enable_embeddings and embedding_results:
                embedding_csv = pd.DataFrame(embedding_results).to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Embedding Sonuçlarını CSV olarak indir", 
                    embedding_csv, 
                    "embedding_analysis_results_diversified.csv", 
                    "text/csv"
                )
        else:
            st.info("Hiç uygun cümle bulunamadı.")