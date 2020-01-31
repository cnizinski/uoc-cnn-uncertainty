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
    # Iterate across glcms
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


def glcm_energy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Harlick energy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of energy_list
    '''
    # Initialize features list
    energy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        asm = 0
        # Iterate across i and j
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                asm += (n_glcm[i][j]) * (n_glcm[i][j])
        energy_list.append(np.sqrt(asm))
    # Return average ASM value
    return np.mean(energy_list)


def glcm_homogeneity(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick homogeneity
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of homog_list
    '''
    # Initialize features list
    homog_list = []
    # Iterate across glcms
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


def glcm_entropy_calc(glcm, ngl):
    '''
    Returns entropy for single glcm
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
    Outputs : entropy
    '''
    entropy = 0
    # Iterate across i, and j
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            if val == 0.0:
                entropy += 0
            else:
                entropy -= val * np.log(val)
    return entropy


def glcm_entropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of entropy_list
    '''
    # Initialize features list
    entropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        entropy_list.append(glcm_entropy_calc(n_glcm, ngl))
    # Return average entropy value
    return np.mean(entropy_list)


def glcm_stat_calc(glcm, ngl):
    '''
    Returns means and standard deviations for single GLCM
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
    Outputs : entropy
    '''
    # Calculate mean values
    mux, muy = 0, 0
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            mux += i * val
            muy += j * val
    # Calculate standard deviations
    varx, vary = 0, 0
    for i in range(0, ngl-1):
        for j in range(0, ngl-1):
            val = glcm[i][j]
            varx += (i-mux)**2 * val
            vary += (j-muy)**2 * val
    sigx = np.sqrt(varx)
    sigy = np.sqrt(vary)
    return mux, muy, sigx, sigy


def glcm_correlation(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick correlation
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of corr_list
    '''
    # Initialize features list
    corr_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Get means and standard devs wrt x and y
        meanx, meany, stdx, stdy = glcm_stat_calc(n_glcm, ngl)
        # Calculate correlation
        inner = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                inner += (n_glcm[i][j] * i * j)
        corr = (inner - meanx*meany) / (stdx*stdy)
        corr_list.append(corr)
    # Return average entropy value
    return np.mean(corr_list)


def glcm_variance(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of var_list
    '''
    # Initialize features list
    var_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Get means and standard devs wrt x and y
        meanx, meany, _stdx, _stdy = glcm_stat_calc(n_glcm, ngl)
        meanxy = (meanx + meany) / 2.0
        # Calculate correlation
        var = 0
        for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                var += ((i - meanxy)**2 * n_glcm[i][j])
        var_list.append(var)
    # Return average entropy value
    return np.mean(var_list)


def pxpy_calc(glcm, ngl, k):
    '''
    Returns p_x+y(k)
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
              k (int, sum of i and j)
    Outputs : p_plus
    '''
    p_plus = 0
    # Add to p_x+y if i+j=k
    for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                if (i+j) == k:
                    p_plus += glcm[i][j]
                else:
                    p_plus += 0
    return p_plus


def pxmy_calc(glcm, ngl, k):
    '''
    Returns p_x-y(k)
    Inputs  : glcm (single normalized gray level co. matrix)
              ngl (int, # gray levels)
              k (int, abs diff of i and j)
    Outputs : p_minus
    '''
    p_minus = 0
    # Add to p_x+y if i+j=k
    for i in range(0, ngl-1):
            for j in range(0, ngl-1):
                if abs(i-j) == k:
                    p_minus += glcm[i][j]
                else:
                    p_minus += 0
    return p_minus


def glcm_contrast(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick contrast
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of contrast_list
    '''
    # Initialize features list
    contrast_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        contrast = 0
        # Iterate across k, call pxmy_calc
        for k in range(0, ngl-1):
            contrast += (k**2) * pxmy_calc(n_glcm, ngl, k)
        contrast_list.append(contrast)
    # Return average contrast value
    return np.mean(contrast_list)


def glcm_sumavg(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum average
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumavg_list
    '''
    # Initialize features list
    sumavg_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        sumavg = 0
        # Iterate across k, call pxpy_calc
        for k in range(2, 2*ngl):
            sumavg += k * pxpy_calc(n_glcm, ngl, k)
        sumavg_list.append(sumavg)
    # Return average contrast value
    return np.mean(sumavg_list)


def glcm_sumvar(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumvar_list
    '''
    # Initialize features list
    sumvar_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        sumavg = 0
        # Iterate across k, call pxpy_calc
        for k in range(2, 2*ngl):
            sumavg += k * pxpy_calc(n_glcm, ngl, k)
        # Calculate sum variance
        sumvar = 0
        for k in range(2, 2*ngl):
            sumvar += (k-sumavg)**2 * pxpy_calc(n_glcm, ngl, k)
        sumvar_list.append(sumvar)
    # Return average contrast value
    return np.mean(sumvar_list)


def glcm_sumentropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick sum entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumentropy_list
    '''
    # Initialize features list
    sumentropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate sum entropy
        sumentropy = 0
        for k in range(2, 2*ngl):
            val = pxpy_calc(n_glcm, ngl, k)
            if val == 0.0:
                sumentropy += 0
            else:
                sumentropy -= val * np.log(val)
        sumentropy_list.append(sumentropy)
    # Return average contrast value
    return np.mean(sumentropy_list)


def glcm_diffvar(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick difference variance
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of sumvar_list
    '''
    # Initialize features list
    diffvar_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        diffvar = 0
        # Iterate across k and l
        for k in range(0, ngl-1):
            inner = 0
            for l in range(0, ngl-1):
                inner += l * pxmy_calc(n_glcm, ngl, l)
            diffvar += (k - inner)**2 * pxmy_calc(n_glcm, ngl, k)
        diffvar_list.append(diffvar)
    # Return average contrast value
    return np.mean(diffvar_list)


def glcm_diffentropy(glcms_dict, ngl):
    '''
    Returns directionally-averaged Haralick difference entropy
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : mean of diffentropy_list
    '''
    # Initialize features list
    diffentropy_list = []
    # Iterate across glcms
    for key in glcms_dict:
        glcm = glcms_dict[key]
        n_glcm = glcm / np.sum(glcm)
        # Calculate sum entropy
        diffentropy = 0
        for k in range(0, ngl-1):
            val = pxmy_calc(n_glcm, ngl, k)
            if val == 0.0:
                diffentropy += 0
            else:
                diffentropy -= val * np.log(val)
        diffentropy_list.append(diffentropy)
    # Return average contrast value
    return np.mean(diffentropy_list)


# Feature pipeline
def glcm_features(glcms_dict):
    '''
    Returns direction independent Haralick features
    -- https://doi.org/10.1155/2015/267807
    -- https://doi.org/10.1016/j.patcog.2006.12.004
    Inputs  : glcms_dict (dict of directional matrices)
    Ouputs  : features (as dictionary)
    '''
    # Get number of gray levels
    key0 = list(glcms_dict.keys())[0]
    levels = glcms_dict[key0].shape[0]
    # Get features, fill dictionary
    features = {}
    features['ASM'] = glcm_asm(glcms_dict, levels)                   # f1
    features['Contrast'] = glcm_contrast(glcms_dict, levels)         # f2
    features['Correlation'] = glcm_correlation(glcms_dict, levels)   # f3
    features['Variance'] = glcm_variance(glcms_dict, levels)         # f4
    features['Homogeneity'] = glcm_homogeneity(glcms_dict, levels)   # IDM,f5
    features['Sum Average'] = glcm_sumavg(glcms_dict, levels)        # f6
    features['Sum Variance'] = glcm_sumvar(glcms_dict, levels)       # f7
    features['Sum Entropy'] = glcm_sumentropy(glcms_dict, levels)    # f8
    features['Entropy'] = glcm_entropy(glcms_dict, levels)           # f9
    features['Diff Variance'] = glcm_diffvar(glcms_dict, levels)     # f10
    features['Diff Entropy'] = glcm_diffentropy(glcms_dict, levels)  # f11
    # f12
    # f13
    # f14
    features['Energy'] = glcm_energy(glcms_dict, levels)  # SQRT(f1)
    # Return feature dictionary
    return features