import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Fungsi untuk memuat model dan melakukan prediksi
def load_model():
    # Ganti path_model dengan path menuju model Anda
    path_model = r"C:\Model H5\best_model.h5"
    model = tf.keras.models.load_model(path_model)
    return model

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_disease(model, img_array):
    img_array = tf.image.resize(img_array, (150, 150))
    img_array = tf.image.rgb_to_grayscale(img_array)
    img_array = tf.image.grayscale_to_rgb(img_array)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    prediction = model.predict(img_array)
    return prediction

# Fungsi untuk memberikan rekomendasi pengobatan berdasarkan penyakit
def recommend_treatment(disease):
    if disease == "Coccidiosis":
        return """
        - Vaksinasi coccidiosis
        - Perhatikan kondisi dan kebersihan kandang
        - Gunakan Koksidiostat, Ssulfadimetoksin, dan antibiotik (tetrasiklin, eritromisin, spektinomisin, dan tilosin)
        - Lakukan terapi antioksidan
        """
    elif disease == "New Castle Disease":
        return """
        - Ayam yang tertular harus dengan cepat dikarantina
        - Jika terlalu parah, ayam harus dimusnahkan untuk menghindari penularan
        - Vaksinasi melalui tetes mata
        """
    elif disease == "Salmonella":
        return """
        - Desinfeksi kandang dari salmonela
        - Rutin membersihkan tempat makan ayam
        - Berikan Antibiotik seperti amoksisilin dan kolistin
        """
    else:
        return "Rekomendasi pengobatan tidak tersedia."

# Fungsi utama untuk tampilan Streamlit
def main():
    # Menambahkan judul dan latar belakang dengan gambar dari Pexels
    st.title("Chicken Disease Classification")
    bg_image_url = "https://www.pexels.com/id-id/foto/empat-ayam-jantan-aneka-warna-1769279/"

    # Menampilkan gambar dari URL
    st.markdown(
        f"""
        <style>
            body {{
                background-image: url('{bg_image_url}');
                background-size: cover;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Menambahkan sidebar
    st.sidebar.title("Upload Chicken Feces Image")

    # Upload gambar dari user
    uploaded_file = st.sidebar.file_uploader("Choose a chicken feces image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Tampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Chicken Feces Image.", use_column_width=True)

        # Memuat model
        model = load_model()

        if model is not None:
            # Konversi gambar ke array
            img_array = np.array(image)

            # Lakukan prediksi
            prediction = predict_disease(model, img_array)

            # Mendapatkan nama penyakit berdasarkan prediksi
            disease_mapping = {0: "Coccidiosis", 1: "New Castle Disease", 2: "Salmonella"}
            predicted_disease = disease_mapping[np.argmax(prediction)]

            # Tampilkan hasil prediksi
            st.write("### Prediction:")
            st.write(f"The predicted disease class is: {predicted_disease}")

            # Berikan rekomendasi pengobatan
            st.write("### Treatment Recommendation:")
            treatment_recommendation = recommend_treatment(predicted_disease)
            st.write(treatment_recommendation)