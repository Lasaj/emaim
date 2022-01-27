"""
Author: Rick Wainwright
Date: 13/01/2022

Estimating the uncertainty in medical imaging classification predictions
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import seaborn as sns
from preprocessing import get_labels, prepare_df, get_image_filenames
from generator import get_idg, get_generator_from_df, get_model

# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
WEIGHTS = "./complete/664055_IRNV2/model.InceptionResNetV2.h5"
# TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
# VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
ONLY_FINDINGS = False

AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
OUTPUT_SIZE = 299  # height is the same as width
BATCH_SIZE = 400
NUM_PREDS = 100


def predict_prob(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    print("preds")
    print(preds[0])
    return np.stack(preds).mean(axis=0)


def predict_class(X, model, num_samples):
    proba_preds = predict_prob(X, model, num_samples)
    print("proba preds")
    print(proba_preds[0])
    return np.argmax(proba_preds, axis=1)


## SETUP ##
# Get train/val/test split by patient IDs
# train_imgs = get_image_filenames(TRAIN_FILE_LIST)
# val_imgs = get_image_filenames(VAL_FILE_LIST)
test_imgs = get_image_filenames(TEST_FILE_LIST)

# Load NIH data
df = pd.read_csv(DATA)

# Get all possible finding labels
labels = get_labels(df, only_findings=ONLY_FINDINGS)
print(labels)

# Update data
df = prepare_df(df, labels, IMG_DIR)
df.to_csv('df.csv')
print("DataFrame complete")

# Get dataframes for each split of the data
# train_df = df[(df["Image Index"].isin(train_imgs))]
# valid_df = df[(df["Image Index"].isin(val_imgs))]
test_df = df[(df["Image Index"].isin(test_imgs))]

individual_dfs = {'All': test_df}
for label in labels:
    individual_dfs[label] = test_df[test_df[label] > 0]


accuracies = {}
# Create generators for each split
core_idg = get_idg()
# X_train, y_train = next(get_generator_from_df(core_idg, train_df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False))

for label in individual_dfs.keys():
    test_df = individual_dfs[label]
    X_test, y_test = next(get_generator_from_df(core_idg, test_df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False))

    # X_test = [X_test[8:9]]
    # y_test = [y_test[8]]

    model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE, weights=WEIGHTS)

    # print(X_train[0])
    # print(X_train[0].shape)

    ## OUTPUT ##
    # 100 predictions
    y_pred = predict_class(X_test, model, 100)
    print("y_pred")
    print(y_pred[0])

    print("y_test")
    print(y_test)
    classes = np.argmax(y_test, axis=1)

    print("classes")
    print(classes)

    """
    B. Ghoshal and A. Tucker
    """

    p_hat = []  # 2
    for t in range(len(X_test)):  # 50 repetitions of MC  # 3
        p_hat.append(model.predict(X_test, verbose=0))  # 4
    MC_samples = np.array(p_hat)  # 5
    y_pred_bayesian = np.mean(MC_samples, axis=0)  # average out 50 predictions into 1  # 6
    bayesPredictions = np.argmax(y_pred_bayesian, axis=1)  # return highest likelihood

    print("MC_samples")
    print(MC_samples)
    print("bayesPredictions")
    print(bayesPredictions)
    acc = np.mean(bayesPredictions == y_test)

    total = len(bayesPredictions)
    correct = 0
    for i in range(total):
        if classes[i] == bayesPredictions[i]:
            print(i)
            correct += 1

    print("Correct:", correct)
    accuracy = correct / total
    print("Accuracy:", accuracy)
    accuracies[label] = accuracy

    def predictive_entropy(MC_samples):
        eps = 1e-5
        prob = np.mean(MC_samples, axis=0)
        return -1 * np.sum(np.log(prob + eps) * prob, axis=1)

    H = predictive_entropy(MC_samples)
    print("H")
    print(H)
    print("H min", H.min(), "H max", H.max())
    H_norm = (H - H.min()) / (H.max() - H.min())

    MC_samples = np.array(p_hat)
    mean_prob = np.mean(MC_samples, axis=0)
    y_pred = np.argmax(mean_prob, axis=1)

    sns.set()
    sns.kdeplot(H_norm[classes == y_pred], shade=True, color='forestgreen')
    sns.kdeplot(H_norm[classes != y_pred], shade=True, color='tomato')
    plt.title(f"{label}, Accuracy:{accuracy}")
    plt.savefig(f"{time.strftime('%Y%m%d_%H%M')}_{label}.png")
    plt.close()

print(accuracies)
for l in accuracies.keys():
    print(f"{l}: {accuracies[l]}")