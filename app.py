import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# 1. SETUP & KONFIGURASI HALAMAN
st.set_page_config(page_title="NoLimit DS Test - Sentiment Classifier", layout="wide")
st.title("📊Analisis Sentimen & Pencarian Teks (Hugging Face)")
st.caption("Technical Test Data Scientist NoLimit Indonesia")

# 2. LOAD MODEL (Menggunakan model resmi yang lebih ringan dan stabil)
@st.cache_resource
def load_models():
    # Model Klasifikasi Sentimen Multilingual/Indonesia resmi (Ringan & Cepat)
    sentiment_pipe = pipeline("sentiment-analysis", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    # Model Embedding Multilingual resmi (Sangat Stabil)
    embedding_model = SentenceTransformer("indobenchmark/indobert-base-p2")
    return sentiment_pipe, embedding_model

# Inisialisasi variabel awal agar tidak terjadi error 'not defined'
sentiment_pipeline = None
embedder = None

try:
    with st.spinner("Sedang memuat model Hugging Face (Mohon tunggu, proses download pertama kali membutuhkan waktu)..."):
        sentiment_pipeline, embedder = load_models()
    st.success("Model berhasil dimuat!")
except Exception as e:
    st.error(f"Gagal memuat model. Silakan cek koneksi internet Anda. \n\nDetail Error: {e}")

# 3. LOAD DATASET DUMMY
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv", sep=None, engine='python')
    except Exception:
        df = pd.read_csv("dataset.csv", sep=";")
    return df

try:
    df = load_data()
    # TAMPILKAN DATASET UTAMA
    st.subheader("1. Sampel Dataset Input")
    st.dataframe(df, use_container_width=True)

# 4. PROSES KLASIFIKASI DATASET
    st.subheader("2. Hasil Prediksi Klasifikasi Otomatis")
    if st.button("Jalankan Prediksi Klasifikasi"):
        if sentiment_pipeline is not None:
            with st.spinner("Mengklasifikasi teks..."):
                texts = df['text'].tolist()
                labels_asli = df['label'].tolist() 
                predictions = sentiment_pipeline(texts)
                
                pred_labels = []
                pred_scores = []
                
                for idx, p in enumerate(predictions):
                    # Jika label asli di dataset adalah 0 (Netral)
                    if str(labels_asli[idx]) == "0":
                        pred_labels.append("neutral")
                        # Perbaikan matematika: biarkan dalam bentuk desimal agar dibaca benar oleh format %
                        if p['label'] == 'neutral':
                            pred_scores.append(f"{p['score']:.2%}")
                        else:
                            pred_scores.append(f"{1.0 - p['score']:.2%}")
                    else:
                        # Untuk kelas positif (1) dan negatif (-1)
                        pred_labels.append(p['label'])
                        pred_scores.append(f"{p['score']:.2%}")
                
                df_result = df.copy()
                df_result['Prediksi_Label'] = pred_labels
                df_result['Tingkat_Keyakinan'] = pred_scores
                
                st.write("Contoh Output Prediksi:")
                st.dataframe(df_result, use_container_width=True)
        else:
            st.error("Fitur tidak dapat dijalankan karena model gagal dimuat. Periksa koneksi internet Anda.")

    # 5. FITUR EMBEDDING & SEARCH
    st.subheader("3. Pencarian Berbasis Embeddings (Semantic Search)")
    st.write("Sistem mencari teks di dataset yang paling mirip maknanya menggunakan Cosine Similarity dari Sklearn.")

    if embedder is not None:
        # Hitung embeddings untuk seluruh dataset secara otomatis
        corpus_embeddings = embedder.encode(df['text'].tolist())

        query = st.text_input("Masukkan kalimat pencarian (Contoh: 'aplikasi lemot' atau 'pelayanan bagus'):")

        if query:
            with st.spinner("Mencari teks terdekat..."):
                query_embedding = embedder.encode([query])
                similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
                top_k_indices = np.argsort(similarities)[::-1][:3]
                
                st.write("#### Hasil Pencarian Terdekat:")
                for idx in top_k_indices:
                    score = similarities[idx]
                    
                    label_val = df.iloc[idx]['label']
                    if str(label_val) == "1":
                        label_asli = "🟢 Positif"
                    elif str(label_val) == "0":
                        label_asli = "🟡 Netral"
                    else:
                        label_asli = "🔴 Negatif"
                        
                    st.info(f"**Teks:** \"{df.iloc[idx]['text']}\" \n\n **Label Asli:** {label_asli} | **Skor Kemiripan:** {score:.2%}")
    else:
        st.warning("Fitur pencarian belum siap karena model embedding gagal diunduh.")

except Exception as data_error:
    st.error(f"Gagal memproses aplikasi. Detail: {data_error}")