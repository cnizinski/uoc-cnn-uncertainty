import numpy as np
import pandas as pd
import math
import re
from PIL import Image


def img_info(fname, fields):
    '''
    Gets condensed SEM image name and info
    Inputs  : fname (str, SEM image filename)
              fields (list of strings for data categories or "default")
    Outputs : info_dict (dictionary of from filename)
    '''
    #
    if fields == "default":
        fields = ["Material", "Magnification", "Resolution", "HFW",
                  "StartingMaterial", "CalcinationTemp", "CalcinationTime",
                  "AgingTime", "AgingTemp", "AgingHumidity", "AgingOxygen",
                  "Impurity", "ImpurityConcentration", "Detector",
                  "Coating", "Replicate", "Particle", "Image", "AcquisitionDate"]
    # Fill dictionary from filename and data fields
    info_dict = {}
    info = re.split('_', fname)
    # Correctly labeled images
    if (len(info) == len(fields)):
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Alpha and UO3 split by underscore
    elif (len(info) == len(fields)+1) and (info[0]=='Alpha'):
        info[0] = info[0] + '-' + info[1]
        info.remove(info[1])
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Am and UO3 split by underscore
    elif (len(info) == len(fields)+1) and (info[0]=='Am'):
        info[0] = info[0] + '-' + info[1]
        info.remove(info[1])
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Single missing field
    elif (len(info) == len(fields)-1):
        info.append('NA')
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # Date split by underscore or voltage included
    elif (len(info) > len(fields)) and (info[0]!='Alpha') and (info[0]!='Am'):
        info = info[0:19]
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = info[i]
    # No exception found
    else:
        print(fname, 'does not contain enough fields')
        for i in range(0, len(fields)-1):
            info_dict[fields[i]] = 'x'
    # 
    info_dict['Image'] = info_dict['Image'][0:3]
    info_dict['FileName'] = fname
    # Return image id and info as dictionary
    return info_dict


def convert_fname(fname, fields):
    '''
    Converts project 1 filenames to other scheme
    '''
    # Fill dictionary from filename and data fields
    idict = {}
    info = re.split('_', fname)
    for i in range(0, len(fields)-1):
        idict[fields[i]] = info[i]
    # Get HFW from magnification
    if idict['Magnification'] == '10000x':
        hfw = '30.6um'
    elif idict['Magnification'] == '25000x':
        hfw = '12.3um'
    elif idict['Magnification'] == '50000x':
        hfw = '6.13um'
    elif idict['Magnification'] == '100000x':
        hfw = '3.06um'
    else:
        hfw = 'NA'
    # Create new filename
    new_fname = idict['Material'] + '_' + idict['Magnification'] + '_'
    new_fname += '1024x934_' + hfw + '_' + idict['Precipitate'] + '_'
    new_fname += idict['CalcinationTemp'] + '_' + idict['CalcinationTime'] + '_'
    new_fname += 'NA_NA_NA_NA_'
    new_fname += idict['Ore'] + '-' + idict['Leach'] + '-' + idict['Purification']
    new_fname += '_NA_TLD_NoCoat_' + idict['Replicate'][-1] + '_' + idict['Particle'][-1] + '_'
    new_fname += idict['Image'][0:3] + '_' + 'NA.tif'
    # Return image id and info as dictionary
    return new_fname


def quick_filter(df, filt_dict):
    '''
    Returns filtered dataframe from info in dictionary
    Inputs  : df (pd DataFrame)
              filt_dict (dict['key']=[list, of, valid, values])
    Outputs : filt_df
    '''
    new_df = df
    for key in filt_dict:
        new_df = new_df[new_df[key]==filt_dict[key]]
    return new_df


def json2df(dpath, dfiles):
    '''
    Returns filtered dataframe from info in dictionary
    Inputs  : dpath (str, path to datafiles)
              dfiles (list of filenames to import)
    Outputs : concatenated dataframes
    '''
    df_list = []
    for item in dfiles:
        fname = dpath + '/' + item
        temp_df = pd.read_json(fname, orient='index', dtype=True)
        df_list.append(temp_df)
    return pd.concat(df_list)


def split_dataset(dataframe, test_split, k, seed):
    '''
    Splits dataset into test, train, and cross val. sets
    Inputs:  dataframe: df of split image names, labels
             test_split: size of test set (float 0 to 1)
             k: number of cv folds (integer)
    Outputs: test_df: df of all test ids, labels
             train_df: df of all train ids, labels 
             cv_df: train_df w/ fold labels added
    Usage:   te_df, tr_df, cv_df = split_dataset(all_ df, 0.2, 5, 42)
    '''
    # Split parent df, create train/test dfs
    train_df = dataframe.sample(frac=(1-test_split), random_state=seed)
    test_df = dataframe.drop(train_df.index)
    # Copy and shuffle train_df, reset indexes
    cv_df = train_df.sample(frac=1, random_state=seed)
    df_len = len(cv_df)
    # Create k=cv_folds cross validation sets
    val_list = [0] * df_len
    for fold in list(range(k)):
        idx1 = int(fold * df_len / k)
        idx2 = idx1 + int(df_len / k)
        for idx in range(idx1, idx2):
            val_list[idx] = fold
    cv_df['fold'] = val_list
    return test_df, train_df, cv_df


def stratified_split(df, label_col, test_split, balance, k, seed):
    '''
    Stratified train/test split with oversampled training data
    Inputs:  df: dataframe of split image names, labels
             label_col : column of dataframe (str)
             test_split: size of test set (float 0 to 1)
             balance: training data imbalance (float 0 to 1)
                      (# in each class / # largest class)
             k: number of cv folds (integer)
             seed : random state (integer)
    Outputs: test_df: df of all test ids, labels
             train_df: df of all train ids, labels 
             cv_df: train_df w/ fold labels added
    Usage:   te_df, tr_df, cv_df = stratified_split(df, 0.2, 0.8, 5, 42)
    '''
    # Stratified test split
    test_dfs = []
    for label in df[label_col].unique():
        temp_df = df[df[label_col]==label]
        test_dfs.append(temp_df.sample(frac=test_split,random_state=seed))
    test_df = pd.concat(test_dfs)
    test_df = test_df.sample(frac=1.0,random_state=seed)
    # Remove test images from df, leaving unique training images
    utrain_df = df.drop(test_df.index)
    # Create k stratified cross validation sets
    cv_dfs = []
    for fold in list(range(k)):
        label_dfs = []
        for label in df[label_col].unique():
            temp_df = utrain_df[utrain_df[label_col]==label]
            n_label = len(temp_df)
            n_sample = int(n_label / (k-fold))
            label_dfs.append(temp_df.sample(n=n_sample,random_state=seed))
        fold_df = pd.concat(label_dfs)
        utrain_df = utrain_df.drop(fold_df.index)
        fold_df['fold'] = fold
        cv_dfs.append(fold_df)
    cv_df = pd.concat(cv_dfs)
    # Oversample each fold, join
    oversampled_dfs = []
    for fold in list(range(k)):
        fold_df = cv_df[cv_df['fold']==fold]
        new_fold_df = oversample(fold_df, label_col, balance)
        new_fold_df = new_fold_df.sample(frac=1.0,random_state=seed)
        oversampled_dfs.append(new_fold_df)
    new_cv_df = pd.concat(oversampled_dfs)
    train_df = new_cv_df.drop(columns=['fold'])
    return test_df, train_df, new_cv_df


def oversample(df, label_col, balance):
    '''
    Oversamples df entries for balances datasets
    '''
    labels = df[label_col].unique()
    n_max = df[label_col].value_counts()[0]
    label_dfs = []
    for label in labels:
        temp_df = df[df[label_col]==label]
        n_label = len(temp_df)
        # While class is unbalanced, sample and concat
        while (n_label <= balance*n_max):
            samp_df = temp_df.sample(n=1)
            temp_df = pd.concat([temp_df, samp_df])
            n_label = len(temp_df)
        label_dfs.append(temp_df)
    # Join the oversampled dfs for each label
    balanced_df = pd.concat(label_dfs)
    return balanced_df


def drop_images(df, fname_col, img_path):
    '''
    Drops unreadable images from dataframe
    '''
    bad_imgs = []
    for idx in df.index:
        fname = img_path + '/' + df.loc[idx][fname_col]
        try:
            _img = Image.open(fname)
        except:
            bad_imgs.append(idx)
            print(idx, df.loc[idx][fname_col], 'dropped from df')
    return df.drop(bad_imgs)


def shannon_entropy(pred_list):
    '''
    Returns Shannon entropy (base 2) for set of predictions
    '''
    entropy = 0.0
    for pred in pred_list:
        if pred > 0.0:
            entropy -= pred * np.log2(pred)
        else:
            entropy += 0.0
    return entropy


def get_hfw(fname):
    '''
    Returns image horizontal field width (in um) from file name
    '''
    hfw_str = img_info(fname=fname, fields="default")['HFW']
    if "um" in hfw_str:
        hfw_num = np.float(hfw_str[:-2])
    elif "mm" in hfw_str:
        hfw_num = np.float(hfw_str[:-2])*1000.0
    elif ("HFW" in hfw_str) and ("pt" in hfw_str):
        hfw_str = hfw_str.split('HFW')[1].split('pt')
        hfw_num = np.float(hfw_str[0] + "." + hfw_str[1])
    else:
        hfw_num = "NA"
    return hfw_num


def get_scalebar(full_hfw, full_width, sub_width):
    '''
    Returns scalebar size for SEM images
    '''
    # Calculate pixels per micron
    bar_px = full_width / full_hfw
    bar_scale = 1.0
    # Set dimensions of scalebar
    ii = 0
    while (int(bar_px) > 0.9*sub_width) and (ii < 20):
        bar_px = bar_px / 2
        bar_scale = bar_scale / 2
        ii += 1
    # Convert from um to nm if necessary
    if bar_scale < 1.0:
        bar_scale = np.int(bar_scale * 1000.0)
        units = "nm"
    else:
        units = "um"
    return int(bar_px), np.round(bar_scale,2), units