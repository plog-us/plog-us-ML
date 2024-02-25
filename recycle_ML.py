import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import accuracy_score
from tensorflow.python.client import device_lib
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# unzip
# zip_file_path = '/content/drive/MyDrive/recycle6GB.zip'
# dataset_path = '/content/drive/MyDrive/recycle_data'
# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(dataset_path)

# tensorboard callback
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M%S")
log_dir = f'/home/ivpl/SeinChoi/logs/fit/{timestamp}'
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='500,520')

# GPU checking
print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Check the list of unzipped files
dataset_path = r"/home/ivpl/Dataset/recycle_data/"
extracted_files = os.listdir(dataset_path)
print("압축 해제된 파일 목록:", extracted_files)

# get class list
classes = os.listdir(dataset_path)
classes.sort()

# Check the list of unzipped files
file_lists = []
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    file_list = [os.path.join(class_path, file) for file in os.listdir(class_path) if file.endswith(('jpg', 'jpeg', 'png'))]
    file_lists.append(file_list)

# Create image files and labels
images = []
labels = []
for label, file_list in enumerate(file_lists):
    images += file_list
    labels += [label] * len(file_list)

# Devide Train, Validation set
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# Convert labels to string
train_labels = list(map(str, train_labels))
val_labels = list(map(str, val_labels))

# Get the number of classes
num_classes = len(set(train_labels))

# train augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# val augmentation
validation_datagen = ImageDataGenerator(rescale=1./255)

# train generator
batch_size = 8
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": train_images, "class": train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

# val generator
validation_generator = validation_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": val_images, "class": val_labels}),
    x_col="filename",
    y_col="class",
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

import matplotlib.pyplot as plt
import random

# class list
class_colors = {'can': 'red', 'glass': 'blue', 'paper': 'green', 'plastic': 'purple', 'trash':'pink', 'metal':'orange', 'cardboard':'black'}

# print images
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


# MobileNet
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# model build
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# parameter
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
import math
def lr_scheduler(epoch, lr):
    max_epochs = 50
    base_lr = 0.001

    lr = 0.45 * base_lr * (1 + math.cos(math.pi * epoch / max_epochs))

    return lr
from tensorflow.keras.callbacks import LearningRateScheduler
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# training
epochs =150
history = model.fit(train_generator, epochs=epochs,
                    validation_data=validation_generator,
                    callbacks=[lr_scheduler_callback, tboard_callback])

import matplotlib.pyplot as plt

# history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs_range = range(1, epochs+1)

# loss graph
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
# accuracy graph
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.show()

# 테스트 데이터셋을 로드하고 전처리
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
     # 테스트 데이터를 위한 분할 설정
)

# 모델 평가
loss, accuracy = model.evaluate(test_generator)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

import numpy as np
import matplotlib.pyplot as plt

# test data prediction
num_images_to_display = 8
test_generator.reset()
images, labels = next(test_generator)
predictions = model.predict(images)
class_labels = list(test_generator.class_indices.keys())

# print prediction image
plt.figure(figsize=(12, 8))
for i in range(num_images_to_display):
    plt.subplot(4, 4, i+1)

    normalized_image = (images[i] * 255).astype(np.uint8)

    plt.imshow(normalized_image)
    plt.axis('off')

    true_label = class_labels[np.argmax(labels[i])]
    pred_label = class_labels[np.argmax(predictions[i])]

    plt.title(f'True: {true_label}\nPredicted: {pred_label}', color='green' if true_label == pred_label else 'red')
plt.show()

# prediction test data
y_true = test_generator.classes
y_pred_prob = model.predict(test_generator)
y_pred = tf.argmax(y_pred_prob, axis=1)

# print accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# model save
saved_model_path = "/home/ivpl/SeinChoi"
model.save(saved_model_path)

# model load
loaded_model = tf.keras.models.load_model(saved_model_path)

# translate TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
tflite_model = converter.convert()

# TFLite model save
tflite_model_path = "/home/ivpl/SeinChoi/model.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
