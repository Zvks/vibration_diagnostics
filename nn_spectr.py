import os
# Для работы с файлами
from PIL import Image, ImageEnhance
# Отрисовка изображений
from tensorflow import keras
import random
# Генерация случайных чисел
import matplotlib.pyplot as plt
# Отрисовка графиков
import numpy as np
# Библиотека работы с массивами
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import math
# Математические функции
# from keras.models import Model # Не используется
# from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D # Не используется

IMAGE_PATH = 'Spectrogram\\X\\'

# Определение списка имен классов
CLASS_LIST = sorted(os.listdir(IMAGE_PATH))
# Определение количества классов
CLASS_COUNT = len(CLASS_LIST)
# Проверка результата
print(f'Количество классов: {CLASS_COUNT}, метки классов: {CLASS_LIST}')

data_files = []                           # Cписок путей к файлам картинок
data_labels = []                          # Список меток классов, соответствующих файлам
data_images = []                          # Пустой список для данных изображений

for class_label in range(CLASS_COUNT):    # Для всех классов по порядку номеров (их меток)
    class_name = CLASS_LIST[class_label]  # Выборка имени класса из списка имен
    class_path = IMAGE_PATH + class_name  # Формирование полного пути к папке с изображениями класса
    class_files = os.listdir(class_path)  # Получение списка имен файлов с изображениями текущего класса
    print(f'Размер класса {class_name} составляет {len(class_files)} фотографий')

    # Добавление к общему списку всех файлов класса с добавлением родительского пути
    data_files += [f'{class_path}/{file_name}' for file_name in class_files]

    # Добавление к общему списку меток текущего класса - их ровно столько, сколько файлов в классе
    data_labels += [class_label] * len(class_files)

print('Общий размер базы для обучения:', len(data_labels))

# Задание единых размеров изображений
IMG_WIDTH = 200                         # Ширина изображения
IMG_HEIGHT = 200                        # Высота изображения

for file_name in data_files:
    # Открытие и смена размера изображения
    img = Image.open(file_name).resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img.convert('RGB'))
    img_np = np.array(img)                # Перевод в numpy-массив
    data_images.append(img_np)            # Добавление изображения в виде numpy-массива к общему списку

x_data = np.array(data_images)            # Перевод общего списка изображений в numpy-массив
y_data = np.array(data_labels)            # Перевод общего списка меток класса в numpy-массив

print(f'В массив собрано {len(data_images)} фотографий следующей формы: {img_np.shape}')
print(f'Общий массив данных изображений следующей формы: {x_data.shape}')
print(f'Общий массив меток классов следующей формы: {y_data.shape}')

# Нормированние массива изображений
x_data = x_data / 255.0

# Создание модели последовательной архитектуры
model = Sequential()

# Блок 1
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
model.add(BatchNormalization()) # Добавляем BatchNorm после Conv
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25)) # Добавляем Dropout после пулинга

# Блок 2
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Блок 3 (Опционально, можно добавить больше)
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Плоский слой перед Dense
model.add(Flatten())

# Полносвязные слои
model.add(Dense(64, activation='relu')) # Увеличим количество нейронов
model.add(BatchNormalization())
model.add(Dropout(0.3)) # Больше Dropout перед выходом
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(CLASS_COUNT, activation='softmax'))

model.summary()

# Компиляция модели
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy']) # Попробуем немного повысить lr

# Обучение модели сверточной нейронной сети подготовленных данных
store_learning = model.fit(x_data,  # x_train
                           y_data,  # y_train
                           validation_split=0.2,
                           shuffle=True,
                           batch_size=32,
                           epochs=45,
                           verbose=1)

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
# Создание полотна для рисунка
plt.figure(1, figsize=(18, 5))

# Задание первой (левой) области для построения графиков
plt.subplot(1, 2, 1)
# Отрисовка графиков 'loss' и 'val_loss' из значений словаря store_learning.history
plt.plot(store_learning.history['loss'],
         label='Значение ошибки на обучающем наборе')
plt.plot(store_learning.history['val_loss'],
         label='Значение ошибки на проверочном наборе')
# Задание подписей осей
plt.xlabel('Эпоха обучения')
plt.ylabel('Значение ошибки')
plt.legend()

# Задание второй (правой) области для построения графиков
plt.subplot(1, 2, 2)
# Отрисовка графиков 'accuracy' и 'val_accuracy' из значений словаря store_learning.history
plt.plot(store_learning.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(store_learning.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
# Задание подписей осей
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()

# Фиксация графиков и рисование всей картинки
plt.show()

model.save("nn_spectr.h5") # Переименуем модель

print("end")