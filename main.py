import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib
import classifiers.cnn_1
import classifiers.cnn_2
import classifiers.cnn_3
import classifiers.cnn_data_augmentation_1


#dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)

data_dir = 'C:\\Users\\Medion\\.keras\\datasets\\flower_photos'
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 360
img_width = 360
num_classes = 5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

cfs = [
    classifiers.cnn_1.Classifier(img_width, img_height, num_classes),
    classifiers.cnn_2.Classifier(img_width, img_height, num_classes),
    classifiers.cnn_3.Classifier(img_width, img_height, num_classes),
    classifiers.cnn_data_augmentation_1.Classifier(img_width, img_height, num_classes),
]

for classifier in cfs:
    classifier.train(train_ds, val_ds, epochs=20)

results = {}
for classifier in cfs:
    results[classifier.name()] = classifier.validation(val_ds)

print('results:')
for k, v in results.items():
    print(f"{k}: {v * 100:.5}%")
