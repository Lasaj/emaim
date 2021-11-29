"""
Author: Rick Wainwright
Date: 27/11/2021

Driver for the DenseNet121 model for the ChestX-ray14 data set for "Explain My AI Model"
"""

import tensorflow as tf
from preprocessing import get_image_filenames, get_labels, plot_example, one_hot_label
from generator import XraySequence
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.applications.inception_v3 import InceptionV3
import saliency.core as saliency


"""
Preprocessing as described by Ho and Gwak paper:
downscaling: 224x224 for KD, 229x229 for InceptionV3
normalise: [-1, 1] (using mean and std deviation)
split: 30,805 patients into 70% train, 10% val, 20 test
augmentation: random horizontal flip
"""


IMG_DIR = "./ChestX-ray14/images/"
LABEL_SRC = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
TRAIN_VAL_FILE_LIST = "./ChestX-ray14/train_val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"

OUTPUT_SIZE = (224, 224)
BATCH_SIZE = 1
EPOCHS = 50

train_val_imgs = get_image_filenames(IMG_DIR, TRAIN_VAL_FILE_LIST)
test_imgs = get_image_filenames(IMG_DIR, TEST_FILE_LIST)

# TODO: split labels into train/val/test
labels = get_labels(LABEL_SRC)

train_ds = XraySequence(train_val_imgs, labels, output_size=OUTPUT_SIZE, batch_size=BATCH_SIZE)

plot_example(train_ds, img_size=OUTPUT_SIZE)

# TODO: find pretrained model
# model = DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
#                     input_shape=None, pooling=None, classes=1000)

model = InceptionV3(include_top=True, weights='imagenet', input_tensor=None,
                    input_shape=None, pooling=None, classes=15, classifier_activation='softmax')

# TODO: weight decay of 5e-4
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), loss='categorical_crossentropy',
              metrics=['accuracy'])

# curves = model.fit(train_ds, batch_size=BATCH_SIZE, epochs=EPOCHS)
