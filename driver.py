"""
Author: Rick Wainwright
Date: 27/11/2021

Driver for the InceptionV3 model for the ChestX-ray14 data set for "Explain My AI Model"
"""
import pandas as pd
import numpy as np
from preprocessing import get_image_filenames, plot_example, plot_performance, get_model, get_callbacks, get_scores
from generator import XraySequence, get_idg, get_generator_from_df
from datetime import datetime
from itertools import chain


"""
Preprocessing as described by Ho and Gwak paper:
downscaling: 224x224 for KD, 229x229 for InceptionV3
normalise: [-1, 1] (using mean and std deviation)
split: 30,805 patients into 70% train, 10% val, 20% test
augmentation: random horizontal flip
"""

# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
CHECKPOINT_PATH = "./ChestX-ray14/checkpoints/cp-{epoch:04d}.ckpt"

# Set training variables
AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
OUTPUT_SIZE = 299  # height is the same as width
BATCH_SIZE = 16
EPOCHS = 500

if CURRENT_MODEL not in AVAILABLE_MODELS:
    print('Invalid model')
    exit()

# Get train/val/test split by patient IDs
train_imgs = get_image_filenames(TRAIN_FILE_LIST)
val_imgs = get_image_filenames(VAL_FILE_LIST)
test_imgs = get_image_filenames(TEST_FILE_LIST)

# Update NIH data
df = pd.read_csv(DATA)

df['path'] = df.apply(lambda row: f"{IMG_DIR}{row['Image Index']}", axis=1)  # add image paths

# Get all possible finding labels
labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
labels = [x for x in labels if x != 'No Finding']
labels = sorted(labels)

# One hot encode labels
for label in labels:  # May not be required for training but could be useful
    if len(label) > 1:
        df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0.0)

# Get dataframes for each split of the data
train_df = df[(df["Image Index"].isin(train_imgs))]
valid_df = df[(df["Image Index"].isin(val_imgs))]
test_df = df[(df["Image Index"].isin(test_imgs))]

# Change strings of findings to lists
train_df['labels'] = train_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
valid_df['labels'] = valid_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
test_df['labels'] = test_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

# Create generators for each split
core_idg = get_idg()
train_gen = get_generator_from_df(core_idg, train_df, BATCH_SIZE, labels, OUTPUT_SIZE)
valid_gen = get_generator_from_df(core_idg, valid_df, BATCH_SIZE, labels, OUTPUT_SIZE)
test_X, test_Y = next(get_generator_from_df(core_idg, test_df, BATCH_SIZE, labels, OUTPUT_SIZE))

# Get model
model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = get_callbacks(CURRENT_MODEL)

# Train model
curves = model.fit(train_gen, validation_data=(test_X, test_Y), epochs=EPOCHS, callbacks=callbacks)

# Evaluate model
y_pred = model.predict(test_X)

results = model.evaluate(test_X, test_Y, batch_size=BATCH_SIZE)
get_scores(labels, y_pred, test_Y, datetime.now(), CURRENT_MODEL)
plot_performance(curves, model=CURRENT_MODEL)

model.save(f"{datetime.now()}_{CURRENT_MODEL}_CX14.h5")

with open(f"{datetime.now()}_{CURRENT_MODEL}_results.txt", 'a') as results_file:
    results_file.write(f"{datetime.now()}")
    for result in results:
        results_file.write(f"{result}\n")

with open(f"{datetime.now()}_{CURRENT_MODEL}_predictions", 'a') as predictions_file:
    predictions_file.write(f"{datetime.now()}")
    for p in y_pred:
        predictions_file.write(f"{p}\n")
