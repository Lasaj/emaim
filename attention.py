"""
Author: Rick Wainwright
Date: 27/11/2021

Produce bulk Vanilla and Smoothgrad Integrated Gradient maps from ChestX-ray14 data set
for "Explain My AI Model"
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import PIL.Image
from preprocessing import get_labels, decode_labels, prepare_df
from generator import get_idg, get_generator_from_df, get_model
import saliency.core as saliency


# Set locations of files
IMG_DIR = "./ChestX-ray14/images/"
DATA = "./ChestX-ray14/Data_Entry_2017_v2020.csv"
UNCERTAINTY = "./20220207_1307_uncertainty_InceptionV3.csv"
WEIGHTS = "./complete/665412_softmax_no_NF/model.InceptionV3.h5"

# Set training variables
to_figure = False
to_images = True
AVAILABLE_MODELS = ["InceptionV3", "InceptionResNetV2"]
CURRENT_MODEL = AVAILABLE_MODELS[0]  # Select from AVAILABLE_MODELS
ONLY_FINDINGS = True
NO_COMORBID = False

OUTPUT_SIZE = 299  # height is the same as width
BATCH_SIZE = 1
class_idx_str = 'class_idx_str'

if CURRENT_MODEL not in AVAILABLE_MODELS:
    print('Invalid model')
    exit()

start = time.time()


def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys == [saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model(images)
            output_layer = output_layer[:, target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}


def show_image(im, title='', ax=None):
    if ax is None:
        plt.figure()
    plt.axis('off')
    plt.imshow(im, cmap=plt.cm.inferno, vmin=0, vmax=1)
    plt.title(title, fontsize=22)


def save_image(im, img_name):
    plt.imsave(img_name, im, cmap=plt.inferno(), vmin=0, vmax=1)


def get_original(file_path):
    im = PIL.Image.open(file_path)
    im = im.resize((299,299))
    im = np.asarray(im)
    if im.ndim > 2:
        im = im[:,:,:3]
    else:
        im = np.stack((im,) * 3, axis=-1)
    return im


# Load NIH data
data_df = pd.read_csv(DATA)
uncertainty_df = pd.read_csv(UNCERTAINTY)

# Get all possible finding labels
labels = get_labels(data_df, only_findings=ONLY_FINDINGS)
print(labels)

# Update data
data_df = prepare_df(data_df, labels, IMG_DIR, only_findings=ONLY_FINDINGS, no_comorbid=NO_COMORBID)
data_df.set_index("Image Index", inplace=True)
uncertainty_df.set_index("Image Index", inplace=True)
print(len(data_df))
print(len(uncertainty_df))

df = data_df.join([uncertainty_df], how="inner")

df = df.sort_values(by="Normalised PH")
df.to_csv("20220215_All_Uncertainties.csv")
print(len(df))

# Create generators for each split
core_idg = get_idg()
img_gen = get_generator_from_df(core_idg, df, BATCH_SIZE, labels, OUTPUT_SIZE, shuffle=False)
print("DataFrame complete")

# Get model and optimiser
model, optimizer = get_model(CURRENT_MODEL, len(labels), OUTPUT_SIZE, weights=WEIGHTS)

top_row = ["Image Index"] + labels + ["Prediction"]
predictions_df = pd.DataFrame(columns=top_row)

for i in range(len(df) - 1):
    im, Y_true = img_gen[i]
    im = im[0]
    file_name = img_gen.filenames[i]
    im_orig = get_original(file_name)
    print(file_name)
    img_name = file_name.split('/')[3]
    predictions = model(np.array([im]))

    # update dataframe
    new_row = predictions.numpy().tolist()[0]
    new_row.append(decode_labels(predictions[0], labels))
    new_row.insert(0, img_name)
    predictions_df.loc[len(predictions_df)] = new_row

    prediction_class = np.argmax(predictions[0])
    call_model_args = {class_idx_str: prediction_class}

    # Compare the true findings and predictions
    true_findings = decode_labels(Y_true[0], labels)
    predicted_findings = decode_labels(predictions[0], labels)

    # Construct the saliency object. This alone doesn't do anything.
    integrated_gradients = saliency.IntegratedGradients()
    guided_ig = saliency.GuidedIG()

    # Baseline is a black image.
    baseline = np.zeros(im.shape)

    ### INTERGRATED GRADIENTS ###
    # Compute the vanilla mask and the smoothed mask.
    vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
    # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
    smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
    # Guided mask
    guided_ig_mask_3d = guided_ig.GetMask(
        im, call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

    # Call the visualization methods to convert the 3D tensors to 2D grayscale.
    vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
    smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)
    guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)

    if to_images:
        # save images
        save_image(vanilla_mask_grayscale, f"./maps/vanilla_IG/{img_name}")
        save_image(smoothgrad_mask_grayscale, f"./maps/smoothgrad_IG/{img_name}")
        save_image(guided_ig_mask_grayscale, f"./maps/guided_IG/{img_name}")

    if to_figure:
        # Set up matplot lib figures.
        ROWS = 1
        COLS = 4
        UPSCALE_FACTOR = 10

        title = f"{img_name} - True: {true_findings}; Predicted: {predicted_findings} ({np.amax(predictions[0]):.2f})"

        maps = plt.figure(figsize=(COLS * UPSCALE_FACTOR, (ROWS + 0.5) * UPSCALE_FACTOR))
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

        # Render the saliency masks.
        show_image(vanilla_mask_grayscale, title='Vanilla Integrated Gradients', ax=plt.subplot(ROWS, COLS, 2))
        show_image(smoothgrad_mask_grayscale, title='Smoothgrad Integrated Gradients', ax=plt.subplot(ROWS, COLS, 3))
        show_image(guided_ig_mask_grayscale, title='Guided Integrated Gradients', ax=plt.subplot(ROWS, COLS, 4))

        ax = plt.subplot(ROWS, COLS, 1)
        plt.axis('off')
        plt.title('Original Image', fontsize=22)
        plt.imshow(im_orig)

        plt.suptitle(title, fontsize=28)
        plt.savefig(f"./maps/{CURRENT_MODEL}_{img_name}")
        plt.close(maps)
        print(f"{CURRENT_MODEL}_{img_name}: {predicted_findings}")
        print(f"{predictions[0]}")

predictions_df.set_index("Image Index", inplace=True)
predictions_df.to_csv(f"{time.strftime('%Y%m%d_%H%M')}_attention_predictions.csv")

print(time.time() - start)
