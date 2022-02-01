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
WEIGHTS = "./complete/665412_softmax_no_NF/model.InceptionV3.h5"
# TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
# VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
ONLY_FINDINGS = True

AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
OUTPUT_SIZE = 299  # height is the same as width
NUM_PREDS = 400  # number of predictions to use
BATCH_SIZE = NUM_PREDS


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


def get_all_true(lst):
    results = []
    for findings in lst:
        all_true = []
        for i, v in enumerate(findings):
            if v > 0:
                all_true.append(i)
        results.append(all_true)
    return results


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

for index, label in enumerate(individual_dfs.keys()):
    print("###", index, label, "###")
    test_df = individual_dfs[label]
    X_test, y_test = next(get_generator_from_df(core_idg, test_df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False))

    # X_test = [X_test[8:9]]
    # y_test = [y_test[8]]

    model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE, weights=WEIGHTS)

    # print(X_train[0])
    # print(X_train[0].shape)

    ## OUTPUT ##
    # 100 predictions
    y_pred = predict_class(X_test, model, NUM_PREDS)
    print("y_pred")
    print(y_pred[0])

    print("y_test")
    print(y_test)
    if index == 0:
        classes = np.array(get_all_true(y_test))
    else:
        classes = [index - 1] * NUM_PREDS

    print("classes")
    print(classes)

    """
    B. Ghoshal and A. Tucker
    """

    p_hat = []  # 2
    for t in range(NUM_PREDS):  # 50 repetitions of MC  # 3
        p_hat.append(model.predict(X_test, verbose=0))  # 4
    MC_samples = np.array(p_hat)  # 5
    y_pred_bayesian = np.mean(MC_samples, axis=0)  # average out 50 predictions into 1  # 6
    bayesPredictions = np.argmax(y_pred_bayesian, axis=1)  # return highest likelihood

    print("MC_samples")
    print(MC_samples)
    print("bayesPredictions")
    print(bayesPredictions)
    # acc = np.mean(bayesPredictions == y_test)

    correct_preds = []
    correct = 0
    for i in range(NUM_PREDS):
        if index == 0:
            if bayesPredictions[i] in classes[i]:
                print(i)
                correct_preds.append(i)
                correct += 1
        else:
            if classes[i] == bayesPredictions[i]:
                print(i)
                correct_preds.append(i)
                correct += 1

    print("Correct:", correct)
    accuracy = correct / NUM_PREDS
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
    print("H norm")
    print(H_norm)

    # MC_samples = np.array(p_hat)
    mean_prob = np.mean(MC_samples, axis=0)
    print("Mean prob")
    print(mean_prob)
    y_pred = np.argmax(mean_prob, axis=1)
    print("y_pred")
    print(y_pred)

    H_norm_correct = []
    for i in correct_preds:
        H_norm_correct.append(H_norm[i])

    H_norm_incorrect = []
    for i in range(len(H_norm)):
        if i not in correct_preds:
            H_norm_incorrect.append(H_norm[i])

    sns.set()
    sns.kdeplot(H_norm_correct, shade=True, color='forestgreen')
    sns.kdeplot(H_norm_incorrect, shade=True, color='tomato')

    plt.title(f"{label}, Accuracy:{accuracy}")
    plt.savefig(f"{time.strftime('%Y%m%d_%H%M')}_{label}.png")
    plt.close()

print(accuracies)
for l in accuracies.keys():
    print(f"{l}: {accuracies[l]}")