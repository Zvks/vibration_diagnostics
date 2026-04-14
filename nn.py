from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# Путь к корневой папке с данными
base_dir = 'Fourier\\coord\\X'

# Список для хранения данных и меток
data_list = []
labels = []
N_TOP = 750  # количество сильнейших пиков

for class_name in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(class_path, filename)
            
            # Читаем CSV
            df = pd.read_csv(file_path)
            
            # Убираем заголовок, если он есть
            if df.iloc[0, 0] == 'Frequency_Hz' and df.iloc[0, 1] == 'Amplitude':
                df = df.iloc[1:].reset_index(drop=True)
            
            # Преобразуем в числовой формат
            df = df.astype(float)
            
            # Сортируем по амплитуде по убыванию
            df_sorted = df.sort_values(by='Amplitude', ascending=False)
            
            # Берём топ-N пиков
            top_n = df_sorted.head(N_TOP)
            
            # Если меньше N — дополняем нулями
            if len(top_n) < N_TOP:
                padding = pd.DataFrame(
                    np.zeros((N_TOP - len(top_n), 2)),
                    columns=['Frequency_Hz', 'Amplitude']
                )
                top_n = pd.concat([top_n, padding], ignore_index=True)
            
            # Преобразуем в массив: сначала частоты, потом амплитуды → или чередуем?
            # Вариант 1: [f1, f2, ..., f750, a1, a2, ..., a750] → 1500 признаков
            features = np.concatenate([top_n['Frequency_Hz'].values, top_n['Amplitude'].values])
            
            # Альтернатива (Вариант 2): только амплитуды → 750 признаков
            # features = top_n['Amplitude'].values

            data_list.append(features.astype(np.float32))
            labels.append(class_name)

X = np.array(data_list)  # shape: (num_samples, 4)
y = np.array(labels)

# Кодируем метки классов
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Разделяем на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Классы:", label_encoder.classes_)
print("Размер X_train:", X_train.shape)
print("Размер X_test:", X_test.shape)
print("Уникальные метки:", np.unique(y_train))

# === Подготовка меток и нормализация (опционально, но рекомендуется) ===
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Построение модели ===
num_classes = len(np.unique(y_encoded))

model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')  # ← количество классов, а не len(labels)!
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # ← ИСПРАВЛЕНО!
    metrics=['accuracy']
)

# === Обучение с validation_split ===
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# === Графики ===
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# === Оценка на тесте ===
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nТестовая точность: {test_acc:.4f}")

# === Сохранение модели ===
model.save("nn_fft.h5")
print("Модель сохранена как 'nn_fft.h5'")