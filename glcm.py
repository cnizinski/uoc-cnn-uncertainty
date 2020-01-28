import numpy as np 
import math
import pandas as pd 
import time


def quant_img(img, q):
    '''
    Quantizes image from 0 to nlevels-1
    Inputs  : img (grayscale image)
              q (number of quantization levels, int)
    Ouputs  : qimg(quantized image)
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Quanization
    qimg = np.uint8(np.double(img)/255 * (q-1))
    return qimg


def get_glcms(img, levels, dist):
    '''
    Make GLCM (0, 45, 90, 135 deg)
    Inputs  : img (grayscale image)
              levels (quantization level, int)
              dist (distance between values, int)
    Outputs : glcm_dict
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Quantize image, initialize matrices
    qimg = quant_img(img, levels)
    P0 = np.zeros((levels, levels), dtype=int)
    P45 = np.zeros((levels, levels), dtype=int)
    P90 = np.zeros((levels, levels), dtype=int)
    P135 = np.zeros((levels, levels), dtype=int)
    # Build 0 degree GLCM
    for i in range(0, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-dist-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i][j+dist]
            P0[gl0][gl1] += 1
    # Build 45 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-dist-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j+dist]
            P45[gl0][gl1] += 1
    # Build 90 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(0, qimg.shape[1]-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j]
            P90[gl0][gl1] += 1
    # Build 135 degree GLCM
    for i in range(dist, qimg.shape[0]-1):
        for j in range(dist, qimg.shape[1]-1):
            gl0 = qimg[i][j]
            gl1 = qimg[i-dist][j-dist]
            P135[gl0][gl1] += 1
    # Return glcms as dict
    glcm_dict = {'P0':P0, 'P45':P45, 'P90':P90, 'P135':P135}
    return glcm_dict
    
#
# Generating Haralick features from directional GLCMs
#

def glcm_asm(glcms_dict, ngl):
    '''
    Returns directionally-averaged angular second moment (asm)
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of asm_list
    '''
    # Initialize features list
    asm_list = []
    # Iterate across glcms i and j
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        asm = 0
        # Iterate across i and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                asm += (n_glcm[i][j]) * (n_glcm[i][j])
        asm_list.append(asm)
    # Return average ASM value
    return np.mean(asm_list)


def glcm_contrast(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick contrast
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of contrast_list
    '''
    # Initialize features list
    contrast_list = []
    # Iterate across glcms i and j
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        contrast = 0
        # Iterate across i, and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                contrast += (i-j)**2 * n_glcm[i][j]
        contrast_list.append(contrast)
    # Return average contrast value
    return np.mean(contrast_list)


def glcm_homogeneity(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick homogeneity
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of homog_list
    '''
    # Initialize features list
    homog_list = []
    # Iterate across glcms i and j
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        homog = 0
        # Iterate across i, and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                front = 1 / (1 + (i-j)**2)
                homog += front * n_glcm[i][j]
        homog_list.append(homog)
    # Return average homogeneity value
    return np.mean(homog_list)


def glcm_entropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of entropy_list
    '''
    # Initialize features list
    entropy_list = []
    # Iterate across glcms i and j
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        entropy = 0
        # Iterate across i, and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                val = n_glcm[i][j]
                if val == 0.0:
                    entropy += 0
                else:
                    entropy -= val * np.log(val)
        entropy_list.append(entropy)
    # Return average entropy value
    return np.mean(entropy_list)


def glcm_correlation(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick correlation
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of corr_list
    '''
    # Initialize features list
    corr_list = []
    # Iterate across glcms i and j
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate mean values
        mux, muy = 0, 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                val = n_glcm[i][j]
                mux += i * val
                muy += j * val
        # Calculate standard deviations
        varx, vary = 0, 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                val = n_glcm[i][j]
                varx += (i-mux)**2 * val
                vary += (j-muy)**2 * val
        sigx = np.sqrt(varx)
        sigy = np.sqrt(vary)
        # Calculate correlation
        corr = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                val = n_glcm[i][j]
                corr += val * (i-mux)*(j-muy) / (sigx * sigy)
        corr_list.append(corr)
    # Return average entropy value
    return np.mean(corr_list)


def glcm_features(glcms_dict):
    '''
    Returns direction independent Haralick features
    https://doi.org/10.1155/2015/267807
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : features (as dictionary)
    '''
    # Get number of gray levels
    key0 = list(glcms_dict.keys())[0]
    levels = glcms_dict[key0].shape[0]
    # Get features, fill dictionary
    features = {}
    features['ASM'] = glcm_asm(glcms_dict, levels)
    features['Energy'] = np.sqrt(features['ASM'])
    features['Contrast'] = glcm_contrast(glcms_dict, levels)
    features['Homogeneity'] = glcm_homogeneity(glcms_dict, levels)
    features['Entropy'] = glcm_entropy(glcms_dict, levels)
    features['Correlation'] = glcm_correlation(glcms_dict, levels)
    # Return feature dictionary
    return features