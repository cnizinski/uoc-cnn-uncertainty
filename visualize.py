from keras import backend as K
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import cv2
import os
import time
import string
from scipy.signal import convolve2d
from .preprocessing import random_crop, pseudorandom_crop, test_gen, crop_generator
from .helpers import get_hfw, get_scalebar
from sklearn.metrics import confusion_matrix, classification_report


def get_gradcam_v1(cropped_img, targets, model, last_conv, scale, series):
    '''
    Produces Grad-CAM heatmaps from 224x224x3 image and trained model
    Adapted from: https://vincentblog.xyz/posts/class-activation-maps
    Inputs  : img (224x224x3 numpy array)
              label_idxs (output of train_gen.class_indices)
              model (trained and compiled keras model)
              last_conv (str, name of last convolutional layer)
              scale (boolean, normalize and resize cams)
    Outputs : gradcam_dict (dictionary of heatmaps for each class)
    '''
    # Preprocess input image
    x = cropped_img / 255
    x = np.expand_dims(x, axis=0)
    # Get model layer weights for final conv and fc
    final_conv = model.get_layer(last_conv)
    class_wts = model.layers[-1].get_weights()[0]
    #print(np.array(class_wts).shape)
    get_output = tf.keras.backend.function([model.layers[0].input], 
                 [final_conv.output, model.layers[-1].output])
    [conv_outputs, _predictions] = get_output(x)
    conv_outputs = conv_outputs[0, :, :, :]
    # Initialize and calculate CAM for each class
    gradcam_dict = {}
    for target_key, target_value in targets.items():
        cam = np.zeros(dtype=np.float32, shape=conv_outputs.shape[0:2])
        for index, weight in enumerate(class_wts[0:512, target_value]):
            temp = weight * conv_outputs[:, :, index]
            cam += temp
        gradcam_dict[target_key] = cam
    # Normalize and resize CAMs
    if scale is True:
        new_gradcam_dict = normalize_resize_gradcam(gradcam_dict, series, 0.0)
        return new_gradcam_dict
    else:
        return gradcam_dict


def normalize_resize_gradcam(cam_dict, pred_series, thresh):
    '''
    Normalizes class activation maps to class probabilities
    '''
    new_cam_dict = {}
    for key, value in cam_dict.items():
        class_prob = pred_series[key+"_prob"]
        value = np.abs(value)
        value = value / np.max(value) * class_prob
        new_hm = cv2.resize(value, (224,224))
        new_hm = np.where(new_hm < thresh, 0, new_hm)
        new_cam_dict[key] = new_hm
    return new_cam_dict


def single_image_plot(cropped_img, preds, scaled_dict, overlay):
    '''
    Creates figure with CAMs for single image
    Inputs  : cropped_img (224x224x3 np array cropped image)
              preds (dataframe series of results)
              scaled_cams (dict of scaled CAMs from get_gradcam)
              overlay (bool, overlay CAMs on cropped_img?)
    Outputs : fig
    '''
    # Setup plot parameters
    bbox1_props = dict(boxstyle="square,pad=0.12", fc="white", alpha=0.4, lw=0)
    bbox2_props = dict(boxstyle="square,pad=0.12", fc="yellow", alpha=1, lw=0)
    fig = plt.subplots(2,3, figsize=(7,4.75))
    ax = plt.subplot(231)
    plt.imshow(cropped_img)
    plt.axis("off")
    # Display classes/entropy
    temp_str = "True: "+preds['true_label']+\
        "\nEntropy="+str(np.round(preds['entropy'],3))
    plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
    # Display scale bar
    hfw = get_hfw(preds['image'])
    bar_px, bar_scale, bar_units = get_scalebar(hfw, 1024, 224)
    ax.add_patch(patches.Rectangle((5, 190), bar_px, 10, color='yellow'))
    if bar_units == "um":
        ann_str = str(bar_scale) + r" $\mu$m"
    else:
        ann_str = str(bar_scale) + " nm"
    plt.annotate(ann_str, (40,200), bbox=bbox2_props, fontsize=12)
    i = 1
    for key in scaled_dict:
        heatmap = scaled_dict[key]
        heatmap = cv2.resize(heatmap, (224,224))
        plt.subplot(231+i)
        if overlay is True:
            plt.imshow(cropped_img)
            #plt.imshow(heatmap, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.4)
            plt.imshow(heatmap, cmap='jet', alpha=0.4)
        else:
            #plt.imshow(heatmap, cmap='jet', vmin=0.0, vmax=1.0, alpha=1.0)
            plt.imshow(heatmap, cmap='jet', alpha=1.0)
        temp_str = key + "\n"+str(np.round(preds[key+'_prob'],3))+\
            r"$\pm$"+str(np.round(preds[key+'_unc'],3))
        plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
        plt.axis("off")
        i = i + 1
    plt.tight_layout()
    return fig
    

def single_class_plot(corr_df, targets, true_label, model, img_path, conf):
    '''
    Creates figure with CAMs for single class with minimum confidence score
    Inputs  : corr_df (dataframe of correct predictions)
              targets (dict of class indices)
              true_label (str, label of interest)
              model (trained keras model)
              img_path (str, path to image directory)
              conf (float 0.0 to 1.0)
    Outputs : fig
    '''
    # Setup plot parameters
    abc = string.ascii_letters
    bbox1_props = dict(boxstyle="square,pad=0.12", fc="white", alpha=0.4, lw=0)
    bbox2_props = dict(boxstyle="square,pad=0.12", fc="yellow", alpha=1, lw=0)
    fig = plt.figure(1, figsize=(6.5,7.0))
    spec = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    true_idx = targets[true_label]
    ii = 0
    while ii < 5:
        jj = 0
        while jj < 5:
            # Load single image belonging to correct class
            img_df = corr_df[corr_df['true_label']==true_label].sample(n=1, replace=True)
            series = img_df.iloc[0]
            full_path = os.path.join(img_path, series['image'])
            img = cv2.imread(full_path)
            # Get pseudorandom crop, predict image
            seed = np.random.randint(10000+jj+ii)
            cimg = pseudorandom_crop(img, 224, seed)
            img_batches = test_gen(img_df, img_path, 5, 1)
            img_crops = crop_generator(img_batches,224,'pseudorandom',seed,None,None)
            preds = model.predict_generator(img_crops,steps=1,verbose=0,workers=1)[0]
            # Check if crop's softmax score for true_label > conf
            true_prob = preds[true_idx]
            if true_prob >= conf:
                # Plot image crop and GradCAM for class
                res = get_gradcam_v1(cimg, targets, model, "add_16", True, series)
                hm = res[true_label]
                ax = fig.add_subplot(spec[ii,jj])
                ax.imshow(cimg)
                ax.imshow(hm, cmap='jet', alpha=0.4)
                temp_str = "("+abc[ii*5+jj]+")"#+str(np.round(true_prob,3))
                plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
                plt.axis("off")
                # Add scale bar if first image
                if (ii == 0) and (jj == 0):
                    hfw = get_hfw(series['image'])
                    bar_px, bar_scale, bar_units = get_scalebar(hfw, 1024, 224)
                    ax.add_patch(patches.Rectangle((5, 185), bar_px, 15, color='yellow'))
                    if bar_units == "um":
                        ann_str = str(bar_scale) + r" $\mu$m"
                    else:
                        ann_str = str(bar_scale) + " nm"
                    plt.annotate(ann_str, (40,200), bbox=bbox2_props)
                #
                jj += 1
            else:
                # Try new image/crop
                jj += 0
        ii += 1
    plt.suptitle(true_label, fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # return figure
    return fig


def multi_class_plot(corr_df, targets, model, img_path, conf):
    '''
    Creates figure with CAMs for single class with minimum confidence score
    Inputs  : corr_df (dataframe of correct predictions)
              targets (dict of class indices)
              true_label (str, label of interest)
              model (trained keras model)
              img_path (str, path to image directory)
              conf (float 0.0 to 1.0)
    Outputs : fig
    '''
    # Setup plot parameters
    abc = string.ascii_letters
    bbox1_props = dict(boxstyle="square,pad=0.12", fc="white", alpha=0.4, lw=0)
    bbox2_props = dict(boxstyle="square,pad=0.12", fc="yellow", alpha=1, lw=0)
    fig = plt.figure(1, figsize=(6.5,7.8))
    spec = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ii = 0
    for key, val in targets.items():
        temp_df = corr_df[corr_df['true_label']==key].sample(n=5, replace=True)
        jj = 0
        while jj < 5:
            # Load single image belonging to correct class
            img_df = temp_df[temp_df['true_label']==key].sample(n=1, replace=True)
            series = img_df.iloc[0]
            full_path = os.path.join(img_path, series['image'])
            img = cv2.imread(full_path)
            # Get pseudorandom crop, predict image
            seed = np.random.randint(10000)
            cimg = pseudorandom_crop(img, 224, seed)
            img_batches = test_gen(img_df, img_path, 5, 1)
            img_crops = crop_generator(img_batches,224,'pseudorandom',seed,None,None)
            preds = model.predict_generator(img_crops,steps=1,verbose=0,workers=1)[0]
            # Check if crop's softmax score for true_label > conf
            true_prob = preds[val]
            if true_prob >= conf:
                # Plot image crop and GradCAM for class
                res = get_gradcam_v1(cimg, targets, model, "add_16", True, series)
                hm = res[key]
                ax = fig.add_subplot(spec[ii,jj])
                ax.imshow(cimg)
                ax.imshow(hm, cmap='jet', alpha=0.4)
                temp_str = "("+abc[ii*5+jj]+")"#+str(np.round(true_prob,3))
                plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
                plt.axis("off")
                # Add scale bar if first image
                if (ii == 0) and (jj == 0):
                    hfw = get_hfw(series['image'])
                    bar_px, bar_scale, bar_units = get_scalebar(hfw, 1024, 224)
                    ax.add_patch(patches.Rectangle((5, 185), bar_px, 15, color='yellow'))
                    if bar_units == "um":
                        ann_str = str(bar_scale) + r" $\mu$m"
                    else:
                        ann_str = str(bar_scale) + " nm"
                    plt.annotate(ann_str, (40,200), bbox=bbox2_props)
                #
                if jj == 2:
                    ax.set_title(key, fontsize=15)
                jj += 1
            else:
                # Try new image/crop
                jj += 0
        ii += 1
    plt.tight_layout(w_pad=0.0, h_pad=0.3)
    # return figure
    return fig


def triple_cm(df_5u, df_5f, df_16, title, report):
    '''
    Returns supplemental figure comparing 2x 5-way and 1x 16-way predictions
    Inputs  : preds5u (dataframe of 5-way U3O8 model predictions)
              preds5f (df of 5-way UO3/U3O8/UO2 predictions)
              preds16 (df of 16-way predictions)
              title (string for title or bool False for none)
              report (bool, print classification report?)
    Outputs : fig  (pyplot figure of confusion matrices)
    '''
    # Set 5-way and 16-way prediction labels
    labels_5way = ["ADU", "AUC", "MDU", "SDU", "UO4"]
    labels_16way = ['ADU-U3O8','ADU-UO2','ADU-UO3','AUC-U3O8','AUC-UO3',
                    'AUCd-UO2','AUCi-UO2','MDU-U3O8','MDU-UO2','MDU-UO3',
                    'SDU-U3O8','SDU-UO2','SDU-UO3','UO4-U3O8','UO4-UO2','UO4-UO3']
    # Make figure
    amap = "Blues"
    fig = plt.figure(figsize=(6.5, 8.0))
    gs = gridspec.GridSpec(nrows=2, ncols=2, height_ratios=[1, 2.75])
    # 5-way U3O8
    _ax0 = fig.add_subplot(gs[0, 0])
    cm0 = confusion_matrix(df_5u['true_label'],df_5u['pred_label'],\
        labels=labels_5way)
    cm0_df = pd.DataFrame(cm0, index=labels_5way, columns=labels_5way)
    sns.heatmap(cm0_df,square=True,annot=True,cbar=False,fmt="d",cmap=amap)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title(r"(a) 5-class U$_{3}$O$_{8}$", fontsize=12)
    # 5-way Full
    _ax1 = fig.add_subplot(gs[0, 1])
    cm1 = confusion_matrix(df_5f['true_label'],df_5f['pred_label'],\
        labels=labels_5way)
    cm1_df = pd.DataFrame(cm1, index=labels_5way, columns=labels_5way)
    sns.heatmap(cm1_df,square=True,annot=True,cbar=False,fmt="d",cmap=amap)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("(b) 5-class Full", fontsize=12)
    # 16-way
    _ax2 = fig.add_subplot(gs[1:, :])
    cm2 = confusion_matrix(df_16['true_label'],df_16['pred_label'],\
        labels=labels_16way)
    cm2_df = pd.DataFrame(cm2, index=labels_16way, columns=labels_16way)
    sns.heatmap(cm2_df,square=True,annot=True,cbar=False,fmt="d",cmap=amap)
    plt.xticks(fontsize=10, rotation=90)
    plt.yticks(fontsize=10, rotation=0)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.title("(c) 16-class", fontsize=12)
    #
    if title is False:
        plt.tight_layout()
    else:
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        plt.suptitle(title, fontsize=15)
    # 
    if report is True:
        print(" 5-way U3O8 Model")
        print(classification_report(df_5u['true_label'],df_5u['pred_label'],\
            labels=labels_5way,zero_division=0))
        print("\n 5-way Full Model")
        print(classification_report(df_5f['true_label'],df_5f['pred_label'],\
            labels=labels_5way,zero_division=0))
        print("\n 16-way Full Model")
        print(classification_report(df_16['true_label'],df_16['pred_label'],\
            labels=labels_16way,zero_division=0))
    #
    return fig
