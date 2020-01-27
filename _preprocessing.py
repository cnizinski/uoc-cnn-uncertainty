import numpy as np
import pandas as pd 


def sample_img(img, barh, rows, cols, region):
    '''
    Removes scalebar from image, returns img portion
    Inputs  : img (grayscale im as np array)
              barh (info bar height in px)
              rows, cols (number of rows and columns)
              region (img region to return)
    Outputs : imgs (specified img region)
    Usage   : img_arr = sample_img(img, 59, 3, 2, 1)
    '''
    # Check for grayscale image
    if len(img.shape) > 2:
        print("Input grayscale image")
        return
    # Check for valid region
    if region > (rows*cols):
        print("input valid region")
        return
    # Remove info bar from image
    if barh > 0:
        img = img[:-barh]
    else:
        img = img
    # Get region dimensions
    dims = img.shape
    img_reg = np.zeros((rows*cols, int(dims[0]/rows), int(dims[1]/cols), 1), dtype=int)
    index = 1
    for g1 in range(0, rows):
        for g2 in range(0, cols):
            # Boundaries of subimages
            y1 = int(g1 * dims[0] / rows)
            y2 = y1 + int(dims[0] / rows)
            x1 = int(g2 * dims[1] / cols)
            x2 = x1 + int(dims[1] / cols)
            sub_img = img[y1:y2, x1:x2]
            # Resize and save image to file
            if index == region:
                img_reg = sub_img
            index += 1
    # Return array of subdivided images
    return img_reg



