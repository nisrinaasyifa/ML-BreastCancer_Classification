# Analisis Klasifikasi Kanker Payudara Menggunakan K-Nearest Neighbors (KNN)

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk menganalisis klasifikasi tumor kanker payudara malignant (ganas) dan benign (jinak) dengan menggunakan algoritma Machine Learning, yaitu **K-Nearest Neighbors (KNN)**. Dataset yang digunakan adalah [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer) dari Library `scikit-learn`: 

## 📂 Struktur Proyek
- `DataScience_BreastCancerClassification.ipynb`: Script utama untuk preprocessing, training, evaluasi, dan visualisasi model.
- `DataScience_BreastCancerClassification.py` : Script Python tanpa output.
- `Portofolio_BreastCancer_Classification.pdf` : Penjelasan tentang proyek ini dalam bentuk power point.
- `breast_cancer_dataset.csv`: Dataset yang telah disimpan dalam format CSV sehingga dapat digunakan kembali untuk eksperimen selanjutnya.
- `README.md`: Dokumentasi proyek ini.

## 🛠️ Teknologi yang Digunakan
- **Python**: bahasa pemrograman tingkat tinggi yang sederhana, mudah dibaca, dan fleksibel. Dikenal dengan sintaks yang jelas, Python banyak digunakan dalam pengembangan web, data science, kecerdasan buatan, dan otomatisasi.
- **Visual Studio Code**: editor kode yang mendukung berbagai bahasa pemrograman, memiliki fitur seperti debugging, IntelliSense, terminal terintegrasi.
- **scikit-learn**: library machine learning yang menyediakan alat untuk data mining dan analisis data.
- **matplotlib**: library plotting untuk membuat visualisasi statistil, animasi, dan interaktif dalam Python.
- **Seaborn**: library visualisasi data Python berbasis Matplotlib yang terintegrasi dengan Pandas, dirancang untuk membuat - visualisasi statistik yang informatif dan menarik dengan kode minimal.
- **NumPy**: Sebuah package untuk komputasi ilmiah dengan Python, menyediakan dukungan untuk array dan matriks.
- **pandas**: Sebuah library manipulasi dan analisis data yang menyediakan struktur data seperti DataFrames.

## 🔍 Tahapan Analisis Data
1. **Load Data**: Memuat dataset kanker payudara dari `sklearn.datasets`.
2. **Menyimpan Dataset** ke dalam file CSV untuk referensi di masa depan.
3. **Preprocessing Data**:
   - Mengecek missing value dan duplikasi data.
   - Normalisasi fitur menggunakan `StandardScaler`.
   - Pembagian dataset menjadi data training (80%) dan data testing (20%).
   - Visualisasi distribusi data.
4. **Pemodelan Machine Learning**:
   - **KNN**: Menentukan jumlah tetangga optimal dengan GridSearchCV.
5. **Evaluasi Performa Model**:
   - **Classification report** menunjukkan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score).
   - **Model accuracy**, yaitu persentase prediksi yang benar dibandingkan dengan total data uji.
6. **Visualisasi Data** 
   - **Confusion Matrix KNN** untuk menunjukkan jumlah prediksi yang benar dan salah untuk setiap kelas.

## 📊 Hasil Visualisasi
Proyek ini mencakup beberapa visualisasi seperti:
- Distribusi kelas dalam dataset.
- Confusion matrix untuk KNN.

## 📌 Kesimpulan
**Akurasi Tinggi**
   - Model KNN mencapai akurasi 96%, yang menunjukkan kinerja yang sangat baik dalam mengklasifikasikan data kanker.

**Performa pada Kelas Malignant dan Benign**
   - Presisi & recall yang tinggi untuk kedua kelas.
   - Recall untuk Malignant (0.93) lebih rendah daripada Benign (0.99), yang berarti beberapa kasus kanker salah diklasifikasikan.

**Confusion Matrix Insights**
   - 3 Kasus ganas salah diklasifikasikan sebagai jinak, yang dapat berisiko dalam diagnosis medis.
   - 1 Kasus jinak salah diklasifikasikan sebagai ganas, yang berpotensi menyebabkan kecemasan yang tidak perlu bagi pasien.

💡 *Terima kasih semoga bermanfaat!*
