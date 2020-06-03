from keras import backend as K
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import os
from uncertainty_pkg import random_crop


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
    x = preprocess_input(cropped_img)
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
        new_gradcam_dict = normalize_resize_heatmaps(gradcam_dict, series, 0.0)
        return new_gradcam_dict
    else:
        return gradcam_dict


def normalize_resize_heatmaps(cam_dict, pred_series, thresh):
    '''
    Normalizes class activation maps to class probabilities
    '''
    # Get max and min of each heatmap
    #max_list = []
    #for key, value in cam_dict.items():
    #    max_list.append(np.max(value))
    #_glob_max = np.max(max_list)
    # Filter out values < thresh, normalize to global max
    new_cam_dict = {}
    for key, value in cam_dict.items():
        class_prob = pred_series[key+"_prob"]
        value = np.abs(value)
        value = value / np.max(value) * class_prob
        new_hm = cv2.resize(value, (224,224))
        new_hm = np.where(new_hm < thresh, 0, new_hm)
        new_cam_dict[key] = new_hm
    return new_cam_dict


def single_gradcam(cropped_img, preds, scaled_cams, overlay):
    '''
    Creates figure with CAMs for single image
    Inputs  : cropped_img (224x224x3 np array cropped image)
              preds (dataframe series of results)
              scaled_cams (dict of scaled CAMs from get_gradcam)
              overlay (bool, overlay CAMs on cropped_img?)
    Outputs : fig
    '''
    bbox1_props = dict(boxstyle="square,pad=0.12", fc="white", alpha=0.4, lw=0)
    fig = plt.subplots(2,3, figsize=(7,4.75))
    plt.subplot(231)
    plt.imshow(cropped_img)
    plt.axis("off")
    temp_str = "True: "+preds['true_label']+"\nEntropy="+\
        str(np.round(preds['entropy'],3))
    plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
    i = 1
    for key in scaled_cams:
        heatmap = scaled_cams[key]
        plt.subplot(231+i)
        if overlay is True:
            plt.imshow(cropped_img)
            plt.imshow(heatmap, cmap='jet', vmin=0.0, vmax=1.0, alpha=0.5)
        else:
            plt.imshow(heatmap, cmap='jet', vmin=0.0, vmax=1.0, alpha=1.0)
        temp_str = key + " CAM\n"+str(np.round(preds[key+'_prob'],3))+\
            r"$\pm$"+str(np.round(preds[key+'_unc'],3))
        plt.annotate(temp_str, (10,45), bbox=bbox1_props, fontsize=12)
        plt.axis("off")
        i = i + 1
    plt.tight_layout()
    return fig
    

def label_gradcam(corr_df, targets, true_label, model, img_path):
    '''
    Creates figure with CAMs for single image
    Inputs  : cropped_img (224x224x3 np array cropped image)
              preds (dataframe series of results)
              scaled_cams (dict of scaled CAMs from get_gradcam)
              overlay (bool, overlay CAMs on cropped_img?)
    Outputs : fig
    '''
    fig = plt.figure(1, figsize=(6.5,7.0))
    spec = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    for ii in range(0,5):
        for jj in range(0,5):
            temp_df = corr_df[corr_df['true_label']==true_label].sample(n=1, replace=True)
            # Get CAM for sampled image
            series = temp_df.iloc[0]
            full_path = os.path.join(img_path, series['image'])
            img = cv2.imread(full_path)
            cimg = random_crop(img, 224)
            results = get_gradcam_v1(cimg, targets, model, "add_16", True, series)
            heatmap = results[true_label]
            # Plot cropped image and CAM
            ax = fig.add_subplot(spec[ii,jj])
            ax.imshow(cimg)
            ax.imshow(heatmap, cmap='jet', vmin=0.2, vmax=1.0, alpha=0.5)
            plt.axis("off")
    plt.suptitle(true_label + " CAMs", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # return figure
    return fig


def multiple_gradcam(corr_df, targets, model, img_path):
    '''
    Creates figure with CAMs for single image
    Inputs  : cropped_img (224x224x3 np array cropped image)
              preds (dataframe series of results)
              scaled_cams (dict of scaled CAMs from get_gradcam)
              overlay (bool, overlay CAMs on cropped_img?)
    Outputs : fig
    '''
    fig = plt.figure(1, figsize=(6.5,7.8))
    spec = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ii = 0
    for key in targets.keys():
        temp_df = corr_df[corr_df['true_label']==key].sample(n=5, replace=True)
        for jj in range(0,5):
            # Get CAM for sampled images
            series = temp_df.iloc[jj]
            full_path = os.path.join(img_path, series['image'])
            img = cv2.imread(full_path)
            cimg = random_crop(img, 224)
            results = get_gradcam_v1(cimg, targets, model, "add_16", True, series)
            heatmap = results[key]
            # Plot cropped image and CAM
            ax = fig.add_subplot(spec[ii,jj])
            ax.imshow(cimg)
            ax.imshow(heatmap, cmap='jet', vmin=0.2, vmax=1.0, alpha=0.5)
            if jj == 2:
                ax.set_title(key, fontsize=15)
            plt.axis("off")
        ii = ii + 1
    plt.tight_layout(w_pad=0.0, h_pad=0.3)
    # return figure
    return fig