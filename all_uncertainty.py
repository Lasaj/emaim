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
from sklearn.metrics import confusion_matrix, accuracy_score

# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
WEIGHTS = "./complete/665412_softmax_no_NF/model.InceptionV3.h5"
# TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
# VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
ONLY_FINDINGS = True
SPLIT_FINDINGS = False

AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
OUTPUT_SIZE = 299  # height is the same as width
NUM_PREDS = 50  # number of predictions to use
BATCH_SIZE = 1


def predict_prob(X, model, num_samples):
    preds = [model(X, training=True) for _ in range(num_samples)]
    return np.stack(preds).mean(axis=0)


def predict_class(X, model, num_samples):
    proba_preds = predict_prob(X, model, num_samples)
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


def predictive_entropy(MC_samples):
    eps = 1e-5
    prob = np.mean(MC_samples, axis=0)
    return -1 * np.sum(np.log(prob + eps) * prob, axis=1)


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
if SPLIT_FINDINGS:  # create adf for each finding
    for label in labels:
        individual_dfs[label] = test_df[test_df[label] > 0]

accuracies = {}
# Create generators for each split
core_idg = get_idg()
# X_train, y_train = next(get_generator_from_df(core_idg, train_df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False))
img_gen = get_generator_from_df(core_idg, test_df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False)

model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE, weights=WEIGHTS)

top_row = ["Image Index", "Correct", "Prediction", "True Findings", "Predictive Entropy (PH)", "Normalised PH"]
uncertainty_df = pd.DataFrame(columns=top_row)

for i in range(len(test_df) - 1):
    X_test, y_test = img_gen[i]
    X_test = np.array([X_test[0]])
    file_name = img_gen.filenames[i].split('/')[3]  # requires shuffle = False

    y_pred = predict_class(X_test, model, NUM_PREDS)  # standard prediction

    classes = np.array(get_all_true(y_test))  # all true findings

    p_hat = []  # 2
    for t in range(NUM_PREDS):  # 50 repetitions of Monte Carlo dropout
        p_hat.append(model.predict(X_test, verbose=0))
    MC_samples = np.array(p_hat)
    y_pred_bayesian = np.mean(MC_samples, axis=0)
    bayesPredictions = np.argmax(y_pred_bayesian, axis=1)  # return highest likelihood

    # Show what is happening
    print("\n", file_name)
    print("Bayes prediction", bayesPredictions[0])
    print("True findings", classes[0])

    correct = "False"
    if bayesPredictions[0] in classes[0]:  # check if prediction is true
        correct = "True"

    H = predictive_entropy(MC_samples)
    print(H)

    # Record findings in dataframe
    new_row = []
    new_row.append(file_name)
    new_row.append(correct)
    new_row.append(bayesPredictions[0])
    new_row.append(classes[0])
    new_row.append(H[0])
    new_row.append(0)
    uncertainty_df.loc[len(uncertainty_df)] = new_row

Hs = uncertainty_df["Predictive Entropy (PH)"]
max_value = Hs.max()
min_value = Hs.min()

uncertainty_df["Normalised PH"] = (uncertainty_df["Predictive Entropy (PH)"] - min_value) / (max_value - min_value)

uncertainty_df.set_index("Image Index", inplace=True)
uncertainty_df.to_csv(f"{time.strftime('%Y%m%d_%H%M')}_uncertainty_{CURRENT_MODEL}.csv")
