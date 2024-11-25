import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Memuat model yang sudah disimpan
best_model_filename = 'LinearSVC.pkl'
best_model = pickle.load(open(best_model_filename, 'rb'))

# Memuat TfidfVectorizer yang digunakan saat pelatihan model
vectorizer_filename = 'tfidf_vectorizer.pkl'  # Misalnya, jika Anda juga menyimpan vectorizer
vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

# Menambahkan antarmuka pengguna di Streamlit
st.title("Deteksi Bullying dalam Komentar")

# Kolom untuk memasukkan komentar
comment = st.text_area("Masukkan komentar di sini:")

# Ketika tombol prediksi ditekan
if st.button("Prediksi"):
    if comment.strip() == "":
        st.warning("Harap masukkan komentar terlebih dahulu!")
    else:
        # Menggunakan vectorizer untuk mengubah teks input menjadi fitur yang sesuai untuk model
        comment_vectorized = vectorizer.transform([comment])

        # Menggunakan model untuk memprediksi apakah komentar adalah bullying atau tidak
        prediction = best_model.predict(comment_vectorized)

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.write("Komentar ini **mengandung bullying**.")
        else:
            st.write("Komentar ini **tidak mengandung bullying**.")
