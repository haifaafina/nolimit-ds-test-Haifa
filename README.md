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
└── README.md             # Dokumentasi proyek (File ini)

graph TD
    Start([INITIALIZATION: Streamlit Engine Dijalankan]) --> LoadModel[MODEL LOADING: Memuat Arsitektur DistilBERT dan IndoBERT]
    DataIngest[DATA INGESTION: Membaca Berkas dataset.csv]
    DataExplo[DATA EXPLORATION: Merender Sampel Dataset pada Dasbor]

    LoadModel --> DataIngest
    DataIngest --> DataExplo

    DataExplo --> PathA{BRANCH A: ANALISIS SENTIMEN}
    DataExplo --> PathB{BRANCH B: SEMANTIC SEARCH}

    PathA --> UserA[Pengguna Memilih Data Teks dan Mengeksekusi Prediksi]
    UserA --> ProcA[Pipeline Transformer Memproses Ekstraksi Label]
    ProcA --> CheckNetral{EVALUASI LOGIKA: Apakah Label Asli sama dengan 0}
    
    CheckNetral -->|Ya| OverrideNetral[Override ke neutral dan Hitung Ulang Score]
    CheckNetral -->|Tidak| RetainAsli[Retain Hasil Asli dari Model]
    
    OverrideNetral --> RenderA[RENDERING: Tampilkan Tabel Hasil Klasifikasi]
    RetainAsli --> RenderA

    PathB --> UserB[Pengguna Menginput Kueri Pencarian Bebas]
    UserB --> ProcB[IndoBERT Mengonversi Kueri Menjadi Vektor Embeddings]
    ProcB --> MathB[KOMPUTASI MATEMATIKA: Menghitung Jarak Cosine Similarity]
    MathB --> RankB[Mengekstrak Top 3 Hasil Paling Relevan]
    RankB --> RenderB[RENDERING: Tampilkan Hasil Konseptual dan Kemiripan]

    RenderA --> End([END])
    RenderB --> End

    style Start fill:#4CAF50,stroke:#388E3C,stroke-width:2px,color:#FFF
    style End fill:#F44336,stroke:#D32F2F,stroke-width:2px,color:#FFF
    style PathA fill:#0284C7,stroke:#0369A1,stroke-width:2px,color:#FFF
    style PathB fill:#0284C7,stroke:#0369A1,stroke-width:2px,color:#FFF
    style CheckNetral fill:#F59E0B,stroke:#D97706,stroke-width:2px,color:#FFF