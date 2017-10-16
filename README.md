Simple image classification
===========================

It's a very simple example. Capture images from your webcam to classify them.


How to install
==============

This script tested in ubuntu linux.

Linux use:

- ```zenity``` for popup message.

- ```streamer``` for get image from webcam

Clone the repository:


```
git clone https://github.com/Endika/webcam_classification_neuron
cd webcam_clarrification_neuron
pip install -r requirements.txt
```

How to work
===========

Generate your dataset:

```
python generate_data.py [number_photos] [tag]

python generate_data.py 30 normal  # take 30 photos in normal form.
python generate_data.py 30 hand_ups  # take 30 photos in hand ups.
```


Before of them training de model

```
python training.py 60  # training model with dataset 60 epochs more or less. Epoch depend for your dataset.
```


And last go to predict

```
python prediction.py
```

This script take a new photo and predict the result. And hand_ups detect
show de alert popup.


Custom the script to detect your's categories. When more categories training more epochs needs.
