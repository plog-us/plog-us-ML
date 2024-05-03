import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import zipfile
import os
import time
import keras
import matplotlib.pyplot as plt
import random
autotune = tf.data.experimental.AUTOTUNE


# # ZIP 파일 경로
# zip_file_path = '/content/drive/MyDrive/recycle6GB.zip'

# # 압축 해제할 폴더 경로
# dataset_path = '/content/drive/MyDrive/recycle_data'

# # 압축 해제
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(dataset_path)

# 압축 해제된 파일 목록 확인
dataset_path = r"/home/ivpl-d29/dataset/recycle_data/"
extracted_files = os.listdir(dataset_path)
print("압축 해제된 파일 목록:", extracted_files)

# 데이터 경로 설정
dataset_path = r"/home/ivpl-d29/dataset/recycle_data/"

# 클래스 목록 얻기
classes = os.listdir(dataset_path)
classes.sort()

# 각 클래스별로 이미지 파일 목록 가져오기
file_lists = []
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    file_list = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))]
    file_lists.append(file_list)

# 이미지 파일과 레이블 생성
images = []
labels = []
for label, file_list in enumerate(file_lists):
    images += file_list
    labels += [label] * len(file_list)

# 데이터를 훈련 및 검증 세트로 나누기
train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                      test_size=0.2, random_state=42, stratify=labels)

# Convert labels to string
train_labels = list(map(str, train_labels))
val_labels = list(map(str, val_labels))

# Get the number of classes
num_classes = len(set(train_labels))

# 훈련 데이터에 대한 데이터 증강 및 전처리 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
)

# 검증 데이터에 대한 전처리 설정
validation_datagen = ImageDataGenerator(rescale=1./255)

# 훈련 데이터 제너레이터 설정
batch_size = 32
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": train_images, "class": train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

# Convert the generator to a tf.data.Dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: zip(train_generator.filepaths, train_generator.labels),
    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(num_classes,), dtype=tf.float32))
)

# Cache the entire dataset
train_dataset = train_dataset.cache()

# Validation data generator without cache
validation_generator = validation_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": val_images, "class": val_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

# Convert the generator to a tf.data.Dataset
validation_dataset = tf.data.Dataset.from_generator(
    lambda: zip(validation_generator.filepaths, validation_generator.labels),
    output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(num_classes,), dtype=tf.float32))
)
# Cache the entire dataset
validation_dataset = validation_dataset.cache()

# 클래스 목록과 해당 클래스의 색상 지정
class_colors = {'can': 'red', 'glass': 'blue', 'paper': 'green', 'plastic': 'purple', 'trash':'pink', 'metal':'orange', 'cardboard':'black', 'battery':'yellow', 'biological':'green', 'clothes':'pink'}

# 클래스별로 4개의 이미지 출력
plt.figure(figsize=(16, 16))
for class_name in extracted_files:
    class_path = os.path.join(dataset_path, class_name)
    class_images = [file for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))]
    selected_images = random.sample(class_images, 4)

    for i, image_name in enumerate(selected_images):
        image_path = os.path.join(class_path, image_name)
        image = plt.imread(image_path)

        plt.subplot(len(extracted_files), 7, extracted_files.index(class_name) * 7 + i + 1)
        plt.imshow(image)
        plt.title(class_name, color=class_colors[class_name])
        plt.axis('off')

plt.tight_layout()
plt.show()




# MobileNet 모델 불러오기
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 모델 구축
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# MobileNet의 가중치를 동결
base_model.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
epochs = 30
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# 훈련 및 검증 손실
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 훈련 및 검증 정확도
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# 에폭 수
epochs_range = range(1, epochs+1)

# 손실 그래프
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')

# 정확도 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# ----------------------------------------------------------------

# 테스트 데이터셋을 로드하고 전처리
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
     # 테스트 데이터를 위한 분할 설정
)

# 모델 평가
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

import numpy as np
import matplotlib.pyplot as plt

# 테스트 데이터셋에서 이미지 가져오기
num_images_to_display = 8

# 테스트 데이터셋 제너레이터 재설정
test_generator.reset()

# 테스트 데이터셋에서 일부 이미지 및 라벨 가져오기
images, labels = next(test_generator)

# 모델 예측
predictions = model.predict(images)

# 클래스 인덱스를 클래스 레이블로 변환
class_labels = list(test_generator.class_indices.keys())

# 이미지 및 예측 결과를 출력
plt.figure(figsize=(12, 8))
for i in range(num_images_to_display):
    plt.subplot(4, 4, i+1)

    # 이미지를 [0, 255] 범위로 정규화 및 형 변환
    normalized_image = (images[i] * 255).astype(np.uint8)

    # 이미지 출력
    plt.imshow(normalized_image)
    plt.axis('off')

    # 실제 라벨 및 예측 결과 출력
    true_label = class_labels[np.argmax(labels[i])]
    pred_label = class_labels[np.argmax(predictions[i])]

    plt.title(f'True: {true_label}\nPredicted: {pred_label}', color='green' if true_label == pred_label else 'red')

plt.show()

# 테스트 데이터에 대한 예측 수행
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = tf.argmax(y_pred_prob, axis=1)

# 정확도 출력
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# 모델 저장
saved_model_path = '/home/ivpl-d29/myProject/recycle_py/mymodel/'
model.save(saved_model_path)

# 모델 로드
model = tf.keras.models.load_model(r"/home/ivpl-d29/myProject/recycle_py/mymodel/saved_model.pb")

# TFLite로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 변환된 모델 저장
with open('/home/ivpl-d29/myProject/recycle_py/mymodel/', 'wb') as f:
    f.write(tflite_model)