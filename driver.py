"""
Author: Rick Wainwright
Date: 27/11/2021

Driver for the InceptionV3 model for the ChestX-ray14 data set for "Explain My AI Model"
"""

import pandas as pd
import time
from preprocessing import get_image_filenames, plot_performance, get_scores, get_labels, prepare_df, plot_example, \
    get_class_weights
from generator import get_idg, get_generator_from_df, get_callbacks, get_model


# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
# CHECKPOINT_PATH = "./ChestX-ray14/checkpoints/cp-{epoch:04d}.ckpt"

# Set training variables
AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[1]  # Select from AVAILABLE_MODELS
ONLY_FINDINGS = False  # Only include samples with findings if True
NO_COMORBID = False  # Remove samples with more than one finding if True
OUTPUT_SIZE = 299  # height is the same as width
BATCH_SIZE = 32
EPOCHS = 50

if CURRENT_MODEL not in AVAILABLE_MODELS:
    print('Invalid model')
    exit()

# Get train/val/test split by patient IDs
train_imgs = get_image_filenames(TRAIN_FILE_LIST)
val_imgs = get_image_filenames(VAL_FILE_LIST)
test_imgs = get_image_filenames(TEST_FILE_LIST)

# Update NIH data
df = pd.read_csv(DATA)
labels = get_labels(df, only_findings=ONLY_FINDINGS)
df = prepare_df(df, labels, IMG_DIR, only_findings=ONLY_FINDINGS, no_comorbid=NO_COMORBID)

# Get dataframes for each split of the data
train_df = df[(df["Image Index"].isin(train_imgs))]
valid_df = df[(df["Image Index"].isin(val_imgs))]
test_df = df[(df["Image Index"].isin(test_imgs))]

# Create generators for each split
core_idg = get_idg()
train_gen = get_generator_from_df(core_idg, train_df, BATCH_SIZE, labels, OUTPUT_SIZE)
valid_gen = get_generator_from_df(core_idg, valid_df, BATCH_SIZE, labels, OUTPUT_SIZE)
test_X, test_Y = next(get_generator_from_df(core_idg, test_df, 1024, labels, OUTPUT_SIZE))

weights = get_class_weights(labels, train_df)

# Get model and optimiser
model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = get_callbacks(CURRENT_MODEL)

# Check sanity
# plot_example(train_gen, labels, CURRENT_MODEL)

# Train model
curves = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS, callbacks=callbacks, class_weight=weights)

# Evaluate model
y_pred = model.predict(test_X)

results = model.evaluate(test_X, test_Y, batch_size=BATCH_SIZE)
model.save(f"{time.strftime('%Y%m%d_%H%M')}_{CURRENT_MODEL}_CX14.h5")

get_scores(labels, y_pred, test_Y, time.strftime('%Y%m%d_%H%M'), CURRENT_MODEL)
plot_performance(curves, model=CURRENT_MODEL)

with open(f"{time.strftime('%Y%m%d_%H%M')}_{CURRENT_MODEL}_results.txt", 'a') as results_file:
    results_file.write(f"{time.strftime('%Y%m%d_%H%M')}")
    for result in results:
        results_file.write(f"{result}\n")

with open(f"{time.strftime('%Y%m%d_%H%M')}_{CURRENT_MODEL}_predictions", 'a') as predictions_file:
    predictions_file.write(f"{time.strftime('%Y%m%d_%H%M')}")
    for p in y_pred:
        predictions_file.write(f"{p}\n")
