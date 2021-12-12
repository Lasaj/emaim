"""
Author: Rick Wainwright
Date: 27/11/2021

Driver for the DenseNet121 model for the ChestX-ray14 data set for "Explain My AI Model"
"""

import tensorflow as tf
from preprocessing import get_image_filenames, get_labels, plot_example, plot_performance
from generator import XraySequence
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
import saliency.core as saliency
import os
from datetime import datetime


"""
Preprocessing as described by Ho and Gwak paper:
downscaling: 224x224 for KD, 229x229 for InceptionV3
normalise: [-1, 1] (using mean and std deviation)
split: 30,805 patients into 70% train, 10% val, 20% test
augmentation: random horizontal flip
"""

# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
LABEL_SRC = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
CHECKPOINT_PATH = "./ChestX-ray14/checkpoints/cp-{epoch:04d}.ckpt"

# Set training variables
OUTPUT_SIZE = (299, 299)
BATCH_SIZE = 1
EPOCHS = 50

# Get data and create generators
train_imgs = get_image_filenames(IMG_DIR, TRAIN_FILE_LIST)
val_imgs = get_image_filenames(IMG_DIR, VAL_FILE_LIST)
test_imgs = get_image_filenames(IMG_DIR, TEST_FILE_LIST)

train_labels = get_labels(LABEL_SRC, TRAIN_FILE_LIST)
val_labels = get_labels(LABEL_SRC, VAL_FILE_LIST)
test_labels = get_labels(LABEL_SRC, TEST_FILE_LIST)

train_ds = XraySequence(train_imgs, train_labels, output_size=OUTPUT_SIZE, batch_size=BATCH_SIZE)
val_ds = XraySequence(val_imgs, val_labels, output_size=OUTPUT_SIZE, batch_size=BATCH_SIZE)
test_ds = XraySequence(test_imgs, test_labels, output_size=OUTPUT_SIZE, batch_size=BATCH_SIZE)

# Set up callback to save checkpoints
checkpoint_dir = os.path.dirname(CHECKPOINT_PATH)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH, save_weights_only=True, verbose=1)

# Check sanity
plot_example(train_ds, img_size=OUTPUT_SIZE, finding_list=train_ds.FINDING_LABEL)

# Prepare model
# TODO: find pretrained model for weights, classes=14?
# model = DenseNet121(include_top=True, weights=None, input_tensor=None, input_shape=(299, 299, 1),
#                     pooling=None, classes=14, classifier_activation='softmax')

model = InceptionV3(include_top=True, weights=None, input_tensor=None, input_shape=(299, 299, 1),
                    pooling=None, classes=14, classifier_activation='softmax')

# TODO: weight decay of 5e-4
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
curves = model.fit(train_ds, validation_data=val_ds, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cp_callback])
plot_performance(curves)

results = model.evaluate(test_ds, batch_size=BATCH_SIZE)
predictions = model.predict(test_ds)

with open("results.txt", 'a') as results_file:
    results_file.write(f"\n{datetime.now()}")
    for result in results:
        results_file.write(f"{result}\n")

with open("predictions", 'a') as predictions_file:
    predictions_file.write(f"\n{datetime.now()}")
    for p in predictions:
        predictions_file.write(f"{p}\n")

