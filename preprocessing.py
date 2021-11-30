"""
Author: Rick Wainwright
Date: 27/11/2021

Preprocessing and helper functions for ChestX-ray14 dataset images and labels.
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd


def get_image_filenames(img_dir: str, file: str) -> [str]:
    # TODO use path not str literal
    file_names = []
    with open(file) as img_list:
        for line in img_list:
            file_names.append(f"{img_dir}{line.strip()}")
    return file_names


def get_labels(label_source_file: str) -> [str]:
    # TODO: find out how to handle multiple categories
    df = pd.read_csv(label_source_file)
    return df['Finding Labels'].tolist()


def one_hot_label(labels: [str], findings: [str]) -> [float]:
    num_findings = len(findings)
    encoding = (np.zeros(num_findings, dtype='int'))
    for label in labels.split('|'):
        encoding[findings.index(label)] = 1
    return encoding


def get_image(filename: str):
    img = tf.io.read_file(filename)
    img = tf.io.decode_png(img)
    # TODO: do we need to normalise? Use preprocess_input.
    # Normalise
    # mean = tf.reduce_mean(img)
    # sd = np.std(img)
    # return (img - mean) / sd
    return img


def reshape(image, output_size, dimension):
    """
    Resizes the image to output_size and add the additional dimension
    :param image: tensor of the image
    :param output_size: (height, width)
    :param dimension: additional axis for model, 1 in the case of X-ray images
    :return: reshaped tensor
    """
    _, _, depth = image.shape
    if depth > 1:
        image = image[:, :, :1]
    image = tf.image.resize(image, [output_size[0], output_size[1]], antialias=False)
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
