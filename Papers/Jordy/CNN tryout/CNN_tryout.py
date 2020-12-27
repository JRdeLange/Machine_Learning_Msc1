import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

# Read in the data
from tensorflow.python.data import AUTOTUNE

dataFile = open("digits.txt", "r")
dataLines = dataFile.readlines()
dataArrayRaw = np.zeros([len(dataLines), 15 * 16])
labelArray = np.zeros([len(dataLines)])

# Make the data into a numpy array and also construct the labels
for i in range(0, len(dataLines)):
    index = 0
    labelArray[i] = int(i / 200)
    for j in range(0, len(dataLines[i])):
        if dataLines[i][j].isnumeric():
            dataArrayRaw[i][index] = float(dataLines[i][j])
            index += 1

# Rearrange the data to 15x16 pictures
dataArray = np.zeros([2000, 16, 15, 1])
for x in range(0, 2000):
    for i in range(0, 16):
        for j in range(0, 15):
            dataArray[x][i][j][0] = dataArrayRaw[x][i * 15 + j]

# The test and training arrays
trainingDataArray = np.zeros([1000, 16, 15, 1])
testDataArray = np.zeros([1000, 16, 15, 1])
trainingLabelArray = np.zeros([1000])
testLabelArray = np.zeros([1000])

# Split into a training and a test set
for i in range(0, 10):
    for j in range(0, 100):
        trainingDataArray[i * 100 + j] = dataArray[i * 200 + j]
        testDataArray[i * 100 + j] = dataArray[i * 200 + j + 100]
        trainingLabelArray[i * 100 + j] = labelArray[i * 200 + j]
        testLabelArray[i * 100 + j] = labelArray[i * 200 + j + 100]

# Make the values range from 0 to 1
trainingDataArray = trainingDataArray.astype("float32") / 6.
testDataArray = testDataArray.astype("float32") / 6.0

# Make the label arrays one-hot-encoding vectors
trainingLabelArray = np_utils.to_categorical(trainingLabelArray, 10)
testLabelArray = np_utils.to_categorical(testLabelArray, 10)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.2, 0.2)),
    # tf.keras.layers.experimental.preprocessing.RandomZoom(width_factor=(-0.2, 0.1), height_factor=(-0.2, 0.1),
    #                                                      fill_mode="constant"),
    # tf.keras.layers.experimental.preprocessing.RandomZoom(width_factor=(-0.1, 0.1), height_factor=(-0.1, 0.1),
    #                                                      fill_mode="constant")
])

for j in range(0, 10):
    thing = np.zeros([1000, 16, 15, 1])
    thing2 = np.zeros([1000, 10])
    for i in range(0, 1000):
        thing[i] = data_augmentation(tf.expand_dims(trainingDataArray[i], 0))
        thing2[i] = trainingLabelArray[i]

    trainingDataArray = np.append(trainingDataArray, thing, axis=0)
    trainingLabelArray = np.append(trainingLabelArray, thing2, axis=0)


# Make datasets from the raw data
train_dataset = tf.data.Dataset.from_tensor_slices((trainingDataArray, trainingLabelArray))
test_dataset = tf.data.Dataset.from_tensor_slices((testDataArray, testLabelArray))

# Shuffle the data
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# Make the model!
model = Sequential()

# Convolve
model.add(Conv2D(32, (2, 2), input_shape=(16, 15, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, (2, 2), input_shape=(16, 15, 1)))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten
model.add(Flatten())

# The output layer

model.add(Dense(10))

model.add(Activation('softmax'))


# Compile it
model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Look at it
model.summary()

# Train it
model.fit(train_dataset, epochs=10, validation_data=test_dataset)


# https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/


'''
# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0, 0.2), width_factor=(0, 0.2))
])

print(len(train_dataset))

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

print(len(train_dataset))
print("----------------------------")
'''