import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.losses import BinaryCrossentropy
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from numpy.random import default_rng
import numpy as np
import os
import cv2
import sys

# int: 4486 (84%)
# not: 861 (16%)
# total: 5347

DATA_DIRECTORY = "D:\\.School\\WSDOT ML Research\\animal-crossing-loader\\tmpdata"
CLASS = ["int", "not"]
IMG_SIZE = 100
img_array = []

# get images
for c in CLASS:
    path = os.path.join(DATA_DIRECTORY, c)
    label = CLASS.index(c)
    temp_img_array = os.listdir(path)

    for i in range(800):
        img = temp_img_array[i]

        try:
            curr_img = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            curr_img = cv2.resize(curr_img, (IMG_SIZE, IMG_SIZE)) / 255
            img_array.append([curr_img, label])
        except Exception as e:
            pass

# create training and testing sets
rng = default_rng()
rng.shuffle(img_array)
training_ds = img_array[0:1200]
testing_ds = img_array[1200:1600]
x_train = []
y_train = []
x_test = []
y_test = []

for img, label in training_ds:
    x_train.append(img)
    y_train.append(label)

x_train = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_train = np.array(x_train)
y_train = np.array(y_train)

for img, label in testing_ds:
    x_test.append(img)
    y_test.append(label)

x_test = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
x_test = np.array(x_test)
y_test = np.array(y_test)

# create model
model = Sequential(
    [
        Input(shape=(100, 100, 1)),
        Conv2D(32, 3, 1, padding='same', activation='relu'),
        Conv2D(64, 3, 1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, 3, 1, padding='same', activation='relu'),
        Conv2D(256, 3, 1, padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1)
    ]
)

model.compile(
    loss=BinaryCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()
history = model.fit(x_train, y_train, batch_size=40, epochs=5)
print("\nEnd Training, Start Evaluating\n")
[loss, accuracy] = model.evaluate(x_test, y_test, batch_size=40)
print("\nEvaluation on Test Data: Loss = {}, accuracy = {}".format(round(loss, 5), round(accuracy, 5)))

plt.plot(history.history['accuracy'], label='accuracy')
plt.title("Model Accuracy vs Epochs")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.show()
