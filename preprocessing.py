"""
Author: Rick Wainwright
Date: 27/11/2021

Preprocessing and helper functions for ChestX-ray14 dataset images and labels.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, average_precision_score


def get_image_filenames(file: str) -> [str]:
    with open(file) as img_list:
        file_names = [f"{line.strip()}" for line in img_list]
    return file_names


def get_model(num_labels, image_size):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3))
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)
    return tf.keras.Model(base_model.input, output)


def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'model.{model_name}.h5',
        verbose=1,
        save_best_only=True)
    # erly = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks.append(checkpoint)
    # callbacks.append(erly)
    return callbacks


def get_scores(labels, y_pred, test_Y, now):
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
    fig.savefig(f'{now}_IV3_trained_net.png')

    print('InceptionV3 ROC auc score: {:.3f}'.format(roc_auc_score(test_Y.astype(int), y_pred)))


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


def plot_example(data, finding_labels, rows=3, cols=3, img_size=(1024, 1024)):
    """
    Sanity check to print some images and findings. Only use with batch size = 1!
    If there are multiple findings this only lists the first one.
    :param finding_labels: array of findings
    :param data: dataset of (image, [findings])
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
        label = finding_labels[np.argmax(data[i][1])]
        plt.subplot(rows, cols, i+1)
        plt.imshow(img)
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
