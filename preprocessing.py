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
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score


def get_image_filenames(file: str) -> [str]:
    with open(file) as img_list:
        file_names = [f"{line.strip()}" for line in img_list]
    return file_names


def get_scores(labels, y_pred, test_Y, now, model):
    for label, p_count, t_count in zip(labels,
                                       100 * np.mean(y_pred, 0),
                                       100 * np.mean(test_Y, 0)):
        print('%s: actual: %2.2f%%, predicted: %2.2f%%' % (label, t_count, p_count))

    fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
    for (idx, c_label) in enumerate(labels):
        fpr, tpr, thresholds = roc_curve(test_Y[:, idx].astype(int), y_pred[:, idx])
        c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (c_label, auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    fig.savefig(f'{now}_{model}_trained_net.png')

    print('{} ROC auc score: {:.3f}'.format(model, roc_auc_score(test_Y.astype(int), y_pred)))


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


def prepare_image(image, output_size, dimension=3, aug=True):
    """
    Resizes the image to output_size, adds the additional dimension and normalises image
    :param aug: True if augmentation is to be performed
    :param image: tensor of the image
    :param output_size: (height, width)
    :param dimension: additional axis for model, 1 in the case of X-ray images
    :return: reshaped tensor
    """
    # Some images have multiple layers, so flatten these
    _, _, depth = image.shape
    if depth > 1:
        image = image[:, :, :1]

    image = tf.image.grayscale_to_rgb(image)
    if aug:
        image = tf.image.random_flip_left_right(image)
    image = tf.image.resize(image, [output_size[0], output_size[1]], antialias=False)
    image = preprocess_input(image)
    return tf.reshape(image, (output_size[0], output_size[1], dimension))


def plot_example(data, finding_labels, model, rows=5, cols=5, img_size=(1024, 1024)):
    """
    Sanity check to print some images and findings. Only use with batch size = 1!
    If there are multiple findings this only lists the first one.
    :param data: dataset of (image, [findings])
    :param finding_labels: array of findings
    :param model: name of model to save file
    :param rows: number of rows
    :param cols: number of columns
    :param img_size:
    """
    fig = plt.figure(figsize=(4 * cols, 4 * rows))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    for i in range(rows * cols):
        # TODO: refactor to cope with batch size > 1
        img = tf.reshape(data[i][0], (299, 299, 3))
        img = tf.image.resize(img, img_size)
        print(np.amin(img), np.amax(img))
        label_indices = np.argwhere(data[i][1] == np.amax(data[i][1])).flatten().tolist()
        labels = [finding_labels[x] for x in label_indices if x > 0]
        label = '|'.join(labels)
        if len(label) <= 1:
            label = 'No Finding'
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.title(label, size=12, color='black')
        plt.xticks(())
        plt.yticks(())
    plt.savefig(f'{model}_fig.png')
    plt.close(fig)


def plot_performance(curves, model):
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
    fig2.savefig(f'./{model}_acc_loss.png')
