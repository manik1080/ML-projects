## IMPORT STATEMENTS
import matplotlib.pyplot as plt
import numpy as np

# cnn model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

# Augmentation
from tensorflow.keras.preprocessing.image import random_rotation
from tensorflow.image import random_flip_left_right, random_crop
from sklearn.preprocessing import OneHotEncoder

# Get Dataset
from keras.datasets import cifar10
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

x_train = x_train[0:10000]
y_train = y_train[0:10000]
x_test = x_test[0:5000]
y_test = y_test[0:5000]

# Check shape
d = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test':y_test}
for i in d:
    print(f"{i}: {d[i].shape}")



## AUGMENT and add to training data
random_indices = np.random.choice(x_train.shape[0], size=5000, replace=False)
x_train_subset = x_train[random_indices]
y_train_subset = y_train[random_indices]

def augment_image(image):
    image = random_flip_left_right(image)
    image = random_rotation(image, 0.2)
    image = random_crop(image, [32, 32, 3])

    return image

augmented_images = []
for image in x_train_subset:
  augmented_image = augment_image(image)
  augmented_images.append(augmented_image)

augmented_images = np.array(augmented_images)
augmented_labels = y_train_subset

# combine
x_train_augment = np.concatenate((x_train, augmented_images))
y_train_augment = np.concatenate((y_train, y_train_subset))




# One hot Encode
ohe = OneHotEncoder()
ohe.fit(y_train.reshape(-1, 1))
y_train_onehot = ohe.transform(y_train_augment.reshape(-1, 1)).toarray()
y_test_onehot = ohe.transform(y_test.reshape(-1, 1)).toarray()

# Define model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization

cnn_model = Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, (5, 5), activation='relu'),
    # Conv2D(32, (5, 5), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Training
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(x_train_augment, y_train_onehot, epochs=10, batch_size=32, validation_data=(x_test, y_test_onehot))
