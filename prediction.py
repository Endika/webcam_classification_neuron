import os

import cv2
import numpy as np
from keras.models import load_model


def get_im(path):
    """Prepare img."""
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (128, 128))
    # resized = resized.transpose()
    return resized


def _load_from_csv(filename):
    value_dict = {}
    with open('{}'.format(filename), 'r') as csv_file:
        for row in csv_file.readlines():
            print(row)
            data = row.split(';')
            value_dict[data[0]] = int(data[1].replace('\n', ''))
    return value_dict


def alert_box(tokenizer_categories, predictions):
    """Show alert box."""
    alert = ['hand_ups']
    index = np.argmax(predictions)
    for word, value in tokenizer_categories.items():
        print('{}: {}'.format(word, predictions[value]))
        if value == index:
            if word in alert and predictions[value] >= 0.9:
                os.system(
                    'zenity --error --text="{} detected" '
                    '--title="Warning!"'.format(word))


tokenizer_categories = _load_from_csv('tokenizer_categories.csv')
os.system('streamer -f jpeg -s 1024 -o test.jpeg')
X_train = [get_im('test.jpeg')]

X_train = np.array(X_train, dtype=np.uint8)
X_train = X_train.reshape(X_train.shape[0], 128, 128, 1)

print(X_train.shape)
model = load_model('neuron_webcam.h5')

predictions = model.predict(X_train, batch_size=128, verbose=1)
alert_box(tokenizer_categories, predictions[0])
os.system('rm test.jpeg')
