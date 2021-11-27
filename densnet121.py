"""
Author: Rick Wainwright
Date: 27/11/2021

DensNet121 model for the ChestX-ray14 data set for "Explain My AI Model"
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
import glob
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.utils import Sequence
import saliency.core as saliency


"""
Preprocessing as described by Ho and Gwak paper:
downscaling: 224x224 for KD, 229x229 for InceptionV3
normalise: [-1, 1] (using mean and std deviation)
split: 30,805 patients into 70% train, 10% val, 20 test
augmentation: random horizontal flip
"""

# TODO: get train/test/val split from files, don't glob folders
TRAIN_SRC = "./ChestX-ray14/images/*.png"
# VAL_SRC = "./images/val/*.png"
# TEST_SRC = "./images/test/*.png"
LABEL_SRC = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
OUTPUT_SIZE = (224, 224)
BATCH_SIZE = 1


class XraySequence(Sequence):
    """
    Generator to provide (image, [findings]) pairs in batches.
    Options to resize images, augment by random horizontal flips and shuffle at the end of epoch.
    """
    def __init__(self, img_set, label_set, batch_size=16, output_size=(224, 224),
                 augment=None, shuffle=True, seed=1):
        self.x = img_set
        self.y = label_set
        self.batch_size = batch_size
        self.output_size = output_size
        self.augment = augment
        self.shuffle = shuffle
        self.seed = seed
        self.indices = list(range(len(self.x)))

    def __len__(self):
        return np.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):  # TODO: random horizontal flips
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([reshape((get_image(filename)), self.output_size, 1) for filename in batch_x]), \
               np.array([split_findings(findings) for findings in batch_y])

    def op_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def get_image_filenames(img_source_dir):
    return sorted(glob.glob(img_source_dir))


def get_labels(label_source_file):
    df = pd.read_csv(label_source_file)
    return df['Finding Labels'].tolist()


def get_image(filename):
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img)
    # TODO: do we need to normalise? Use preprocess_input.
    # Normalise
    # mean = tf.reduce_mean(img)
    # sd = np.std(img)
    # return (img - mean) / sd
    return img


def split_findings(findings):
    return findings.split('|')


def reshape(image, output_size, dimension):
    image = tf.image.resize(image, [output_size[0], output_size[1]], antialias=False)
    return tf.reshape(image, (output_size[0], output_size[1], dimension))


def plot_example(data, rows=3, cols=3):
    """
    Sanity check to print some images and findings. Only use with batch size = 1!
    If there are multiple findings this only lists the first one.
    :param data: dataset of (image, [findings])
    :param rows: number of rows
    :param cols: number of columns
    """
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(rows * cols):
        # TODO: refactor to cope with batch size > 1
        img = tf.reshape(data[i][0], OUTPUT_SIZE)
        label = data[i][1]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(label[0][0], size=12, color='black')
        plt.xticks(())
        plt.yticks(())
    plt.savefig('fig.png')
    plt.close(fig)


# Start script:
images = get_image_filenames(TRAIN_SRC)
labels = get_labels(LABEL_SRC)

train_ds = XraySequence(images, labels, output_size=OUTPUT_SIZE, batch_size=BATCH_SIZE)

plot_example(train_ds)

# TODO: find pretrained model
model = DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
                    input_shape=None, pooling=None, classes=1000)
