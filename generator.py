"""
Author: Rick Wainwright
Date: 28/11/2021

Generator to feed batches of (image, findings) to model.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
from preprocessing import prepare_image, get_image, one_hot_label
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.optimizers import SGD, RMSprop


def get_idg():
    return ImageDataGenerator(rescale=(1./255),
                              samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip=False,
                              vertical_flip=False,
                              # height_shift_range=0.05,
                              # width_shift_range=0.1,
                              rotation_range=5,
                              # shear_range=0.1,
                              fill_mode='nearest',
                              zoom_range=0.15,
                              )


def get_generator_from_df(idg, df, batch_size, labels, image_size=299, shuffle=True):
    return idg.flow_from_dataframe(dataframe=df,
                                   directory=None,
                                   x_col='path',
                                   y_col='labels',
                                   class_mode='categorical',
                                   batch_size=batch_size,
                                   classes=labels,
                                   target_size=(image_size, image_size),
                                   shuffle=shuffle,
                                   validate_filenames=False
                                   )


def get_model(current_model, num_labels, image_size, weights='imagenet'):
    models = {
        'InceptionV3': InceptionV3(include_top=False, weights='imagenet', input_shape=(image_size, image_size, 3)),
        'InceptionResNetV2': InceptionResNetV2(include_top=False, weights=None, input_shape=(image_size,
                                                                                             image_size, 3))
    }
    optimizers = {
        'InceptionV3': SGD(learning_rate=0.001, momentum=0.9, decay=5e-4),
        'InceptionResNetV2': RMSprop()
    }

    base_model = models[current_model]
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_labels, activation="sigmoid")(x)
    model = tf.keras.Model(base_model.input, output)
    if weights != 'imagenet':
        model.load_weights(weights)
    return model, optimizers[current_model]


def get_callbacks(model_name):
    callbacks = []
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0)
    callbacks.append(tensor_board)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'model.{model_name}.h5',
        verbose=1,
        save_best_only=True)
    # early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    callbacks.append(checkpoint)
    # callbacks.append(early)
    return callbacks


class XraySequence(Sequence):
    """
    Generator to provide (image, [findings]) pairs in batches.
    Options to resize images, augment by random horizontal flips and shuffle at the end of epoch.
    """

    def __init__(self, img_set, label_set, batch_size=16, output_size=(299, 299),
                 augment=None, shuffle=True, finding='any'):
        self.x = img_set
        self.y = label_set
        self.batch_size = batch_size
        self.output_size = output_size
        self.augment = augment
        self.shuffle = shuffle
        self.finding = finding
        self.indices = list(range(len(self.x)))
        self.FINDING_LABELS = [
            'Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion',
            'Emphysema',
            'Fibrosis',
            'Hernia',
            'Infiltration',
            'Mass',
            # 'No Finding',
            'Nodule',
            'Pleural_Thickening',
            'Pneumonia',
            'Pneumothorax'
        ]

        # TODO: make finding selectable, default to any
        # if not finding == 'any':

    def __len__(self):
        return (np.ceil(len(self.x) / self.batch_size)).astype(int)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([prepare_image(get_image(filename), self.output_size) for filename in batch_x]), \
               np.array([one_hot_label(labels, self.FINDING_LABELS) for labels in batch_y])

    def op_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
