"""
Author: Rick Wainwright
Date: 13/01/2022

Estimating the uncertainty in medical imaging classification predictions
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
# import seaborn as sns
from preprocessing import get_labels, prepare_df, get_image_filenames, get_class_weights
from generator import get_idg, get_generator_from_df, get_model, get_callbacks
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, multilabel_confusion_matrix

# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
WEIGHTS = "./complete/665412_softmax_no_NF/model.InceptionV3.h5"
TRAIN_FILE_LIST = "./ChestX-ray14/train_list.txt"
# VAL_FILE_LIST = "./ChestX-ray14/val_list.txt"
TEST_FILE_LIST = "./ChestX-ray14/test_list.txt"
ONLY_FINDINGS = True

AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
OUTPUT_SIZE = 299  # height is the same as width
NUM_PREDS = 50  # number of predictions to use
BATCH_SIZE = 8


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
    all_true = []
    for i, v in enumerate(lst):
        if v > 0:
            all_true.append(i)
    return all_true


def multi_label_accuracy(test_y, pred_y):
    n_correct = 0
    len_y = test_y.shape[0]

    for i in range(len_y - 1):
        if np.argmax(pred_y[i]) in get_all_true(list(test_y[i])):
            n_correct += 1

    return n_correct, len_y


## SETUP ##
# Get train/val/test split by patient IDs
train_imgs = get_image_filenames(TRAIN_FILE_LIST)
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
train_df = df[(df["Image Index"].isin(train_imgs))]
# valid_df = df[(df["Image Index"].isin(val_imgs))]
test_df = df[(df["Image Index"].isin(test_imgs))]

core_idg = get_idg()
test_X, test_Y = next(get_generator_from_df(core_idg, test_df, 1024, labels, OUTPUT_SIZE))

weights = get_class_weights(labels, train_df)

# Get model and optimiser
model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

callbacks = get_callbacks(CURRENT_MODEL)

# Evaluate model
y_pred = model.predict(test_X)

y_pred_standard = model.predict(test_X)
print(test_Y)
print(y_pred_standard)

print(test_Y.shape)
print(y_pred_standard.shape)

std_correct, std_total = multi_label_accuracy(test_Y, y_pred_standard)
print("Standard correct:", std_correct)
print("Total standard:", std_total)
print("Standard accuracy:", std_correct / std_total)

p_hat = []
for t in range(NUM_PREDS):
    p_hat.append(model.predict(test_X,verbose=0))
MC_samples = np.array(p_hat)

y_pred_bayesian = np.mean(MC_samples, axis=0)
bayes_correct, bayes_total = multi_label_accuracy(test_Y, y_pred_bayesian)
print("\nBayesian correct:", bayes_correct)
print("Total standard:", bayes_total)
print("Bayesian accuracy mean:", bayes_correct / bayes_total)

y_pred_bayesian = mode(MC_samples, axis=0)
bayes_correct, bayes_total = multi_label_accuracy(test_Y, y_pred_bayesian[0][0])
# print("Bayesian correct:", bayes_correct)
# print("Total standard:", bayes_total)
print("Bayesian accuracy mode:", bayes_correct / bayes_total)


"""
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 10))
sns.heatmap(cm_df, annot=True, cmap="GnBu", linewidths=.3, fmt="d")
plt.title('Standard CNN (MCDW: ' + '0.2) \nAccuracy:{0:.4f}'.format(acc_std))
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig(f"{time.strftime('%Y%m%d_%H%M')}_standard_prediction.png")
plt.close()


p_hat = []
for t in range(NUM_PREDS):
    p_hat.append(model.predict(test_X,verbose=0))
MC_samples = np.array(p_hat)

y_pred_bayesian = np.mean(MC_samples, axis=0)
matrix = confusion_matrix(test_Y.argmax(axis=1), y_pred_bayesian.argmax(axis=1))
# matrix = multilabel_confusion_matrix(test_Y, np.argmax(y_pred_bayesian, axis=1), labels=labels)
cm_norm = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
print(matrix)
class_acc = np.array(cm_norm.diagonal())
print(class_acc)

cm = np.array(matrix)
acc_bayes = accuracy_score(test_Y.argmax(axis=1), y_pred_bayesian.argmax(axis=1))
sns.set()

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(10, 10))
sns.heatmap(cm_df, annot=True,cmap="GnBu",linewidths=.3,fmt="d")
plt.title('Bayesian CNN (MCDW: ' + '0.2) \nAccuracy:{0:.4f}'.format(acc_bayes))
plt.xlabel('Prediction')
plt.ylabel('Actual')
plt.savefig(f"{time.strftime('%Y%m%d_%H%M')}_bayesian_prediction.png")
plt.close()
"""
