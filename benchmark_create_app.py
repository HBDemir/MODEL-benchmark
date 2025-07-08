import streamlit as st
import pandas as pd

st.set_page_config(page_title="MOF Toxicity MCQ Benchmark Viewer", layout="wide")
st.title("MOF Toxicity MCQ Benchmark Viewer")

st.write("Yüklediğiniz benchmark CSV dosyasındaki çoktan seçmeli soruları ve metinleri interaktif olarak görüntüleyin, arayın ve filtreleyin.")

uploaded_file = st.file_uploader("Bir benchmark CSV dosyası yükleyin", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Yüklendi: {uploaded_file.name} ({len(df)} soru)")

    # Arama kutusu
    search = st.text_input("Soru veya metin içinde ara:")
    if search:
        mask = df.apply(lambda row: search.lower() in str(row.get('mcq','')).lower() or search.lower() in str(row.get('context','')).lower(), axis=1)
        filtered = df[mask]
    else:
        filtered = df

    st.write(f"Toplam {len(filtered)} sonuç gösteriliyor.")

    for i, row in filtered.iterrows():
        st.markdown(f"### Soru {i+1}")
        if 'context' in row:
            st.markdown(f"**Metin:** {row['context']}")
        if 'mcq' in row:
            st.code(row['mcq'], language='markdown')
        if 'source_file' in row:
            st.caption(f"Kaynak: {row['source_file']}")
        st.markdown("---")
else:
    st.info("Lütfen bir CSV dosyası yükleyin.") 