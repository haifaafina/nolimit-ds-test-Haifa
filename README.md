# NoLimit Data Scientist Technical Test - Sentiment Classifier & Semantic Search

Repositori ini berisi solusi pengerjaan *Technical Test* untuk posisi **Data Scientist** di **NoLimit Indonesia**. Proyek ini mengimplementasikan **Pilihan A** yaitu klasifikasi sentimen berbasis model transformer dan pencarian kemiripan teks berbasis *embeddings* menggunakan bahasa pemrograman Python dan antarmuka web interaktif Streamlit.

---

## 📊 Fitur Aplikasi

Aplikasi dasbor interaktif ini mencakup tiga fitur utama yang bekerja secara *real-time*:
1. **Eksplorasi Sampel Dataset**: Membaca dan menampilkan representasi data teks masukan beserta label aslinya secara langsung dari file lokal (`dataset.csv`).
2. **Klasifikasi Sentimen Otomatis (Hugging Face)**: Melakukan prediksi sentimen (*positive*, *neutral*, *negative*) menggunakan model transformer Hugging Face yang dikombinasikan dengan visualisasi tingkat keyakinan (*confidence score*) probabilitas.
3. **Pencarian Berbasis Embeddings (Semantic Search)**: Memungkinkan pengguna memasukkan kata kunci acak untuk mencari 3 teks terdekat yang paling mirip maknanya menggunakan representasi vektor padat (*dense vector representation*) dan perhitungan kedekatan *Cosine Similarity*.

---

## 🛠️ Model Machine Learning & Arsitektur Teknis

Untuk memenuhi standar kebutuhan pemrosesan bahasa alami (NLP) Bahasa Indonesia yang akurat, proyek ini memadukan kombinasi model berikut:

* **Model Klasifikasi Sentimen**: `lxyuan/distilbert-base-multilingual-cased-sentiments-student`  
  Sebuah model *knowledge-distillation* berbasis DistilBERT multibahasa yang sangat ringan, cepat, dan dioptimalkan untuk memprediksi emosi/sentimen teks secara efisien.
* **Model Text Embedding**: `indobenchmark/indobert-base-p2`  
  Model bahasa BERT spesifik Bahasa Indonesia dari IndoBenchmark yang dilatih menggunakan korpus masif bahasa lokal. Model ini memberikan representasi konteks semantik yang sangat kuat dan peka terhadap frasa atau kosakata khas Indonesia.
* **Metrik Kedekatan (Similarity Metric)**: `Cosine Similarity` (via Scikit-Learn)  
  Digunakan untuk mengukur nilai kosinus sudut antara vektor pencarian (*query*) dengan matriks embedding dataset untuk mengurutkan tingkat kemiripan makna.

---

## 📂 Struktur Repositori

```text
nolimit-ds-test-Haifa/
│
├── data/
│   ├── app.py            # Kode sumber utama aplikasi Streamlit
│   ├── dataset.csv       # File data tiruan lokal (Teks & Label)
│   └── requirements.txt  # Daftar pustaka & dependensi eksternal
│
└── README.md             # Dokumentasi proyek
