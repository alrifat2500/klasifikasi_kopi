import streamlit as st
import pandas as pd 
import joblib


model = joblib.load("model_klasifikasi_kopi.joblib")

st.title("Klasifikasi Kualitas Kopi")
st.markdown("Klasifikasi Kualitas kopi berdasarkan fitur Kadar Kafein, Tingkat Keasaman, dan Jenis Proses")

kadar_kafein = st.slider("Kadar Kafein", 30.0, 200.0, 110.0)
tingkat_keasaman = st.slider("Tingkat Keasaman", 0.0, 7.0, 5.0, 0.1)
jenis_proses = st.radio("Jenis Proses", ["Natural", "Honey", "Washed"], index=0)

if st.button("Prediksi", type="primary"):
    data = pd.DataFrame(
        [[kadar_kafein, tingkat_keasaman, jenis_proses]],
        columns=["Kadar Kafein", "Tingkat Keasaman", "Jenis Proses"]
    )
    prediksi = model.predict(data)[0]
    persentase = max(model.predict_proba(data)[0])
    
    st.success(f"Prediksi: {prediksi} dengan keyakinan {persentase*100:.2f}%")
    st.balloons()

st.divider()
st.caption("Dibuat dengan = oleh **Alrifat**")