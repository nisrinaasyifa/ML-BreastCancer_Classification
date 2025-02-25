# Import Library
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# 1. Load dataset Breast Cancer dari scikit-learn
cancer = datasets.load_breast_cancer()
X = cancer.data    # inputan untuk machine learning
y = cancer.target  # output yang dinginkan dari machine learning

# Mengubah dataset menjadi DataFrame.
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Simpan dataset ke dalam file CSV
df.to_csv("breast_cancer_dataset.csv", index=False)
print("Dataset telah disimpan sebagai 'breast_cancer_dataset.csv'")

# Tampilkan 5 baris teratas
df.head(5)

# Menampilkan kumpulan data
df.info()

# Memisahkan fitur (X) dan target (y)
X = df.drop(columns=['target'])
y = df['target']
# Hitung label target
y.value_counts()

# 2. Preprocessing Data
# Pengecekan missing value
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)

# Pengecekan duplikat
duplicates = df.duplicated().sum()
print("Jumlah duplikat:", duplicates)

# Menghapus duplikasi jika ada
df.drop_duplicates(inplace=True)

# Normalisasi fitur untuk meningkatkan performa KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Membagi dataset menjadi training dan testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                    test_size=0.2, random_state=42, stratify=y)

# 4. Visualisasi distribusi target
plt.figure(figsize=(6, 4))
sns.countplot(x=y, palette='husl')
plt.title('Distribusi dalam Dataset Breast Cancer')
plt.xlabel('Kelas')
plt.ylabel('Jumlah Sampel')
plt.xticks(ticks=[0, 1], labels=['Malignant', 'Benign'])
plt.show()

# 5. Processing Data - Hyperparameter Tuning
param_grid_knn = {'n_neighbors': range(1, 20)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
grid_knn.fit(X_train, y_train)
print("Best parameters for KNN:", grid_knn.best_params_)

# 6. Training model dengan KNN
knn_model = KNeighborsClassifier(n_neighbors=grid_knn.best_params_['n_neighbors'])
knn_model.fit(X_train, y_train)

# Prediksi pada data uji
y_pred_knn = knn_model.predict(X_test)

# 7. Hasil evaluasi model KNN 
# Menampilkan laporan klasifikasi
print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=['Malignant', 'Benign']))

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred_knn)
print(f'Akurasi Model KNN: {accuracy:.2f}')

# 8.Visualisasi Confusion Matrix untuk KNN
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues', xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()

# Analisis hasil
print("\nAnalisis Hasil:")
print("1. Dataset Breast Cancer memiliki dua kelas: malignant (ganas) dan benign (jinak).")
print("2. Normalisasi fitur membantu meningkatkan akurasi dengan menghindari dominasi fitur dengan skala besar.")
print("4. Nilai k optimal untuk dataset ini adalah k=7")
print("3. Model KNN dengan k=7 memberikan akurasi yang cukup tinggi, menunjukkan bahwa KNN cocok untuk dataset ini.")

# Penjelasan hasil visualisasi
print("\nPenjelasan Visualisasi:")
print("1. Confusion Matrix menunjukkan jumlah prediksi yang benar dan salah. Mayoritas prediksi benar, menunjukkan model yang cukup akurat.")
print("2. Distribusi kelas menunjukkan bahwa jumlah sampel benign lebih banyak daripada malignant, yang dapat mempengaruhi keseimbangan model.")
