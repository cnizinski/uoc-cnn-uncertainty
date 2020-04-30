import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator


#
# Image cropping for CNN inputs
#

def random_crop(img, crop_size):
    '''
    Crops image from keras flow_from_x to crop_size
    Inputs:  img : image
             crop_size : int, one side of square
    Outputs : img : cropped image             
    '''
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    if (height <= crop_size) or (width <= crop_size):
        return img
    else:
        dy, dx = crop_size, crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(61, height - dy - 61)  # 60px ~ info bar height
        return img[y:(y+dy), x:(x+dx), :]
    


def center_crop(img, crop_size):
    '''
    Crops image from keras flow_from_x to crop_size
    Inputs:  img : image
             crop_size : int, one side of square
    Outputs : img : cropped image             
    '''
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    if (height <= crop_size) or (width <= crop_size):
        return img
    else:
        dy, dx = crop_size, crop_size
        x = int(width/2 + dx/2)
        y = int(height/2 + dy/2)
        return img[y:(y+dy), x:(x+dx), :]


def crop_generator(batches, crop_size, mode):
    """
    Performs random or center crop on keras ImageGen batch
    Inputs  : batches (keras image batch)
              crop_size (int, length of square to crop to)
              mode (str, "random" or "center")
    Outputs : cropped batch of images and labels
    """
    while True:
        batch_x, batch_y = next(batches)
        batch_crops = np.zeros((batch_x.shape[0],crop_size,crop_size,3))
        for i in range(batch_x.shape[0]):
            if mode == 'random':
                batch_crops[i] = random_crop(batch_x[i], 224)
            elif mode == 'center':
                batch_crops[i] = center_crop(batch_x[i], 224)
            else:
                print('Invalid crop mode')
                pass
        # "yield" kw needed for iterable generators
        yield (batch_crops, batch_y)


#
# Keras image generators
#

def train_gen(tr_df, img_dir, num_classes, batch_size):
    '''
    Returns Keras ImageDataGenerators for training set
    Inputs:  
    Outputs: train ImageDataGenerators
    Usage:   train_gen = train_gen(tr_df, img_dir,
                         num_classes, model_params)
    '''
    # Set class_mode by number of classes
    if num_classes == 2:
        class_mode = 'binary'
    elif num_classes > 2:
        class_mode = 'categorical'
    else:
        pass
    # Training ImageDataGenerator and datagen
    tr_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.0)
    tr_batches = tr_datagen.flow_from_dataframe(
        dataframe=tr_df,
        directory=img_dir,
        x_col='image',
        y_col='label',
        interpolation='bicubic',
        class_mode=class_mode,
        shuffle=True,
        target_size=((1024,943)),
        batch_size=batch_size)
    return tr_batches


def test_gen(vt_df, img_dir, num_classes, batch_size):
    '''
    Returns Keras ImageDataGenerators for val./test sets
    Inputs:  
    Outputs: test ImageDataGenerators
    Usage:   val_gen = test_gen(vt_df, img_dir,
                       num_classes, model_params)
    '''
    # Set class_mode by number of classes
    if num_classes == 2:
        class_mode = 'binary'
    elif num_classes > 2:
        class_mode = 'categorical'
    else:
        pass
    # Validation/Test ImageDataGenerator and datagen
    vt_datagen = ImageDataGenerator(rescale=1./255)
    vt_batches = vt_datagen.flow_from_dataframe(
        dataframe=vt_df,
        directory=img_dir,
        x_col='image',
        y_col='label',
        interpolation='bicubic',
        class_mode=class_mode,
        shuffle=False,
        target_size=((1024,943)),
        batch_size=batch_size)
    return vt_batches