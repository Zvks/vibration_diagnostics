import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

def organising_single_dataset(df, n):
    """
    Применяет скользящее окно к датафрейму
    
    Parameters:
    df : pandas.DataFrame - исходный датафрейм
    n : int - размер окна (количество отсчетов)
    
    Returns:
    numpy.array - матрица сегментов размера (количество_сегментов, n)
    """
    # Извлечение второго столбца (акселерометр)
    second_column = df.iloc[:, 1] 
    # Конвертация в numpy array
    second_column_numpy = second_column.to_numpy()
    # Получение длины массива
    length_of_second_column = len(second_column_numpy)
    
    # Проверка, что длина данных достаточна
    if length_of_second_column < n:
        raise ValueError(f"Длина данных ({length_of_second_column}) меньше размера окна ({n})")
    
    # Инициализация пустого массива для результатов
    result_array = np.empty((length_of_second_column - n + 1, n))
    
    # Создание скользящих окон
    for i in range(length_of_second_column - n + 1):
        result_array[i] = second_column_numpy[i:i+n]
    
    return result_array

# Загрузка и подготовка данных
path_lst = [
    "Data_prepared\\Cracking\\Cracking_Z",
    "Data_prepared\\Ideal\\Ideal_Z",
    "Data_prepared\\Offset_Pulley\\Offset_Pulley_Z",
    "Data_prepared\\Wear\\Wear_Z"
]

X = []
Y = ['Cracking', 'Ideal', 'Offset_Pulley', 'Wear']

for path in path_lst:
    tmp = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            tmp.append(df)
    X.append(tmp)

X_extend = []
y_extend = []

for i, clss in enumerate(X):
    tmp = []
    for exmpl in clss:
        tmp.append(organising_single_dataset(exmpl, 226)) # Частота дискретизации: 679 Гц (226/679 ≈ 0.33с) 226 отсчетов ≈ 0.33 секунды
    class_data = np.vstack(tmp)
    X_extend.append(class_data)
    # Создаем метки для текущего класса
    y_extend.extend([i] * len(class_data))

# Объединяем все данные
X_final = np.vstack(X_extend)
y_final = np.array(y_extend)

print(f"Общий размер данных: X {X_final.shape}, y {y_final.shape}")
print(f"Распределение классов: {np.unique(y_final, return_counts=True)}")

# Кодируем метки в one-hot format
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_final)
y_categorical = to_categorical(y_encoded)

print(f"Метки после кодирования: {y_categorical.shape}")

# Разделяем данные на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

# Разделяем тренировочные на train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=np.argmax(y_train, axis=1)
)

print(f"Тренировочные данные: {X_train.shape}")
print(f"Валидационные данные: {X_val.shape}")
print(f"Тестовые данные: {X_test.shape}")

# Изменяем форму данных для 1D CNN (добавляем dimension для каналов)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Форма данных после reshape:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"X_test: {X_test.shape}")

# Создаем модель 1D CNN
def create_1d_cnn_model(input_shape, num_classes):
    model = Sequential([
        # Первый сверточный блок
        Conv1D(filters=64, kernel_size=7, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Второй сверточный блок
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Третий сверточный блок
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Четвертый сверточный блок
        Conv1D(filters=512, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Полносвязные слои
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

# Создаем и компилируем модель
input_shape = (X_train.shape[1], 1)  # (226, 1)
num_classes = len(np.unique(y_encoded))

model = create_1d_cnn_model(input_shape, num_classes)

# Компилируем модель
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=0.00001,
    verbose=1
)

# Обучение модели
print("Начало обучения...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=2,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Оценка модели
print("\nОценка на тестовых данных...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Тестовая точность: {test_accuracy:.4f}")
print(f"Тестовая loss: {test_loss:.4f}")

# Визуализация обучения
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # График точности
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # График потерь
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Предсказания на тестовых данных
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=Y))

# Матрица ошибок
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=Y, yticklabels=Y)
plt.title('Матрица ошибок')
plt.xlabel('Предсказа')
plt.ylabel('True')
plt.show()

# Сохранение модели
#model.save('1d_cnn_vibration_classifier_Z_2.h5')
print("Модель сохранена как '1d_cnn_vibration_classifier.h5'")

# Функция для предсказания на новых данных
def predict_vibration(signal, model, label_encoder):
    """
    Предсказание для нового сигнала вибрации
    """
    # Преобразование сигнала в нужный формат
    if len(signal.shape) == 1:
        signal = signal.reshape(1, -1, 1)
    elif len(signal.shape) == 2:
        signal = signal.reshape(signal.shape[0], signal.shape[1], 1)
    
    # Предсказание
    predictions = model.predict(signal)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    return predicted_label, predictions

# Пример использования на одном тестовом примере
sample_idx = 0
sample = X_test[sample_idx:sample_idx+1]
true_label = y_true_classes[sample_idx]

predicted_label, probabilities = predict_vibration(sample, model, label_encoder)

print(f"\nПример предсказания:")
print(f"Истинный класс: {Y[true_label]}")
print(f"Предсказанный класс: {Y[predicted_label[0]]}")
print(f"Вероятности: {dict(zip(Y, probabilities[0]))}")

