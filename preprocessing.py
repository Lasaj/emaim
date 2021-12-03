"""
Author: Rick Wainwright
Date: 27/11/2021

Preprocessing and helper functions for ChestX-ray14 dataset images and labels.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications.inception_v3 import preprocess_input


def get_image_filenames(img_dir: str, file: str) -> [str]:
    # TODO use path not str literal
    file_names = []
    with open(file) as img_list:
        for line in img_list:
            file_names.append(f"{img_dir}{line.strip()}")
    return file_names


def get_labels(label_source_file: str, img_list_file: str) -> [str]:
    df = pd.read_csv(label_source_file)
    df = df.set_index('Image Index')
    labels = []
    with open(img_list_file) as indices:
        for line in indices:
            labels.append(df.loc[line.strip()].loc['Finding Labels'])
    return labels


def one_hot_label(labels: [str], findings: [str]) -> [float]:
    num_findings = len(findings)
    encoding = (np.zeros(num_findings, dtype='int'))
    for label in labels.split('|'):
        if label in findings:
            encoding[findings.index(label)] = 1
    return encoding


def get_image(filename: str):
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img)
    # Normalise - replaced by model's preprocess in reshape function
    # mean = tf.reduce_mean(img)
    # sd = np.std(img)
    # return (img - mean) / sd
    return img


def prepare_image(image, output_size, dimension=1):
    """
    Resizes the image to output_size, adds the additional dimension and normalises image
    :param image: tensor of the image
    :param output_size: (height, width)
    :param dimension: additional axis for model, 1 in the case of X-ray images
    :return: reshaped tensor
    """
    # Some images have multiple layers, so flatten these
    _, _, depth = image.shape
    if depth > 1:
        image = image[:, :, :1]

    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [output_size[0], output_size[1]], antialias=False)
    image = preprocess_input(image)
    return tf.reshape(image, (output_size[0], output_size[1], dimension))


def plot_example(data, finding_list, rows=3, cols=3, img_size=(1024, 1024)):
    """
    Sanity check to print some images and findings. Only use with batch size = 1!
    If there are multiple findings this only lists the first one.
    :param finding_list: array of findings
    :param data: dataset of (image, [findings])
    :param rows: number of rows
    :param cols: number of columns
    :param img_size:
    """
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(rows * cols):
        # TODO: refactor to cope with batch size > 1
        img = tf.reshape(data[i][0], img_size)
        label = finding_list[np.argmax(data[i][1])]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(label, size=12, color='black')
        plt.xticks(())
        plt.yticks(())
    plt.savefig('fig.png')
    plt.close(fig)


def plot_performance(curves):
    # plot accuracy
    fig2, (gax1, gax2) = plt.subplots(1, 2)
    gax1.plot(curves.history['accuracy'])
    gax1.plot(curves.history['val_accuracy'])
    gax1.legend(['train', 'test'], loc='upper left')
    gax1.title.set_text("Accuracy")
    # plot loss
    gax2.plot(curves.history['loss'])
    gax2.plot(curves.history['val_loss'])
    gax2.legend(['train', 'test'], loc='upper left')
    gax2.title.set_text("Loss")
    fig2.savefig('./acc_loss.png')
