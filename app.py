import streamlit as st
import pickle

# Memuat model yang sudah disimpan
with open("LinearSCV.pkl", "rb") as f:
    model = pickle.load(f)

# Antarmuka pengguna di Streamlit untuk input
st.title("Prediksi dengan Model LinearSVC")

# Misalnya, kita ingin pengguna memasukkan beberapa fitur untuk prediksi
feature_1 = st.number_input("Fitur 1")
feature_2 = st.number_input("Fitur 2")
# Tambahkan fitur lainnya sesuai kebutuhan

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    # Membuat prediksi menggunakan model
    prediction = model.predict([[feature_1, feature_2]])  # Sesuaikan dengan jumlah dan jenis fitur
    st.write(f"Prediksi: {prediction[0]}")
