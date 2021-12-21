"""
Author: Rick Wainwright
Date: 28/11/2021

Generator to feed batches of (image, findings) to model.
"""

import numpy as np
from tensorflow.keras.utils import Sequence
from preprocessing import prepare_image, get_image, one_hot_label
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_idg():
    return ImageDataGenerator(rescale=1 / 255,
                              samplewise_center=True,
                              samplewise_std_normalization=True,
                              horizontal_flip=True,
                              vertical_flip=False,
                              height_shift_range=0.05,
                              width_shift_range=0.1,
                              rotation_range=5,
                              shear_range=0.1,
                              fill_mode='reflect',
                              zoom_range=0.15)


def get_generator_from_df(idg, df, batch_size, labels, image_size=299):
    return idg.flow_from_dataframe(dataframe=df,
                                   directory=None,
                                   x_col='path',
                                   y_col='labels',
                                   class_mode='categorical',
                                   batch_size=batch_size,
                                   classes=labels,
                                   target_size=(image_size, image_size))


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
