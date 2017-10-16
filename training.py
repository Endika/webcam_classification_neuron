import glob
import os
import sys

import cv2
import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.utils import np_utils


def get_im(path):
    """Prepare img."""
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 128))
    # resized = resized.transpose()
    return resized


def _save_to_csv(filename, value_dict):
    with open('{}'.format(filename), 'w') as csv_file:
        for key, value in value_dict.items():
            csv_file.write('{};{}\n'.format(key, value))


def _load_from_csv(filename):
    value_dict = {}
    with open('{}'.format(filename), 'r') as csv_file:
        for row in csv_file.readlines():
            data = row.split(';')
            value_dict[data[0]] = int(data[1].replace('\n', ''))
    return value_dict


def create_model():
    """Create keras model."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3),
                     padding='valid',
                     input_shape=(128, 128, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories) + 1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=1e-6, momentum=0.9)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                  metrics=['accuracy'])
    return model


try:
    epochs = int(sys.argv[1])
except Exception:
    epochs = 1
categories = set()
data_files = []
print("scan files...")
for path, subdirs, files in os.walk('dataset'):
    for name in files:
        cat = path.split('/')[-1]
        categories.add(cat)
        data_files.append((os.path.join(path, name), cat))
print('total files: {}'.format(len(data_files)))
print('total categories: {}'.format(len(categories)))

tokenizer_categories = {}
if glob.glob('tokenizer_categories.csv'):
    tokenizer_categories = _load_from_csv('tokenizer_categories.csv')
else:
    for i, value in enumerate(categories):
        tokenizer_categories[value] = i + 1
    _save_to_csv('tokenizer_categories.csv', tokenizer_categories)
print(tokenizer_categories)
X_train = []
y_train = []
for data in data_files:
    filepath = data[0]
    category = data[1]
    X_train.append(get_im(filepath))
    y_train.append(tokenizer_categories[category])

X_train = np.array(X_train, dtype=np.uint8)
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)
y_train = np.array(y_train, dtype=np.uint8)
y_train = np_utils.to_categorical(y_train, len(categories) + 1)

print(X_train.shape)
print(y_train.shape)
if glob.glob('neuron_webcam.h5'):
    model = load_model('neuron_webcam.h5')
else:
    model = create_model()
# model.summary()
model.fit(X_train, y_train, batch_size=32, epochs=epochs)
          # validation_data=(X_train, y_train))
model.save('neuron_webcam.h5', overwrite=True)
