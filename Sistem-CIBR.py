import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Konfigurasi folder untuk menyimpan gambar upload dan dataset
UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Membuat folder jika belum ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Memuat model VGG16 untuk ekstraksi fitur
base_model = VGG16(weights='imagenet', include_top=False)

# Memeriksa apakah file memiliki ekstensi yang diperbolehkan
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi untuk ekstraksi fitur
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.vgg16.preprocess_input(expanded_img_array)
    features = base_model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

# Fungsi untuk membangun database fitur
def build_feature_database():
    feature_database = {}
    for img_name in os.listdir(DATASET_FOLDER):
        if allowed_file(img_name):
            img_path = os.path.join(DATASET_FOLDER, img_name)
            feature_database[img_name] = extract_features(img_path)
    return feature_database

# Fungsi untuk menemukan gambar serupa
def find_similar_images(query_features, feature_database, top_n=5):
    similarities = {}
    for img_name, features in feature_database.items():
        similarity = cosine_similarity(query_features.reshape(1, -1), features.reshape(1, -1))[0][0]
        similarities[img_name] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return sorted_similarities[:top_n]

# Memuat database fitur
feature_database = build_feature_database()

# Antarmuka pengguna menggunakan Streamlit
st.title("Pencarian gambar Truk")
st.write("Unggah gambar untuk mencari gambar serupa dari dataset.")

# Mengunggah file gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=list(ALLOWED_EXTENSIONS))

if uploaded_file is not None:
    try:
        # Simpan gambar yang diunggah
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Ekstrak fitur dari gambar yang diunggah
        query_features = extract_features(file_path)
        
        # Temukan gambar serupa
        similar_images = find_similar_images(query_features, feature_database)

        # Tampilkan gambar query
        st.image(file_path, caption="Gambar Query", use_column_width=True)

        # Tampilkan gambar serupa
        st.subheader("Gambar Serupa:")
        for image_name, similarity in similar_images:
            image_path = os.path.join(DATASET_FOLDER, image_name)
            st.image(image_path, caption=f"{image_name} - Kesamaan: {similarity:.4f}", use_column_width=True)
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Menambahkan footer
st.markdown("---")  # Garis horizontal
st.markdown("<footer style='text-align: center;'>Dibuat dengan &hearts; Oleh Kelompok 6</footer>", unsafe_allow_html=True)
