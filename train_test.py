from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import optimizers
from keras import metrics
from .models import unfreeze_all
from .preprocessing import test_gen, crop_generator
from .helpers import shannon_entropy, get_hfw
import numpy as np
import pandas as pd
import time


def lr_decay(epoch, lr):
    # Learning rate decay scheduler
    if (epoch != 0):
        new_lr = lr / (1.0 + 0.001*epoch)
    else:
        new_lr = lr
    return new_lr


def train_2steps(train_df, train_gen, model, params):
    '''
    Inputs  : train_df (dataframe of training set)
              model (uncompiled keras model)
              params (dict of model parameters)
    Outputs : trained model, training history
    '''
    #
    # Setup callbacks
    #
    wts = params['data_path'] + '\\temp_wts_best.h5'
    checkpoint = ModelCheckpoint(wts, monitor='loss', verbose=0,
                                 save_best_only=True,mode='min')
    #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
    #                              patience=5, min_lr=1.0e-6, verbose=1)
    schedule = LearningRateScheduler(lr_decay, verbose=0)
    #
    # Compile model, training (part 1)
    #
    opt1 = optimizers.Adam(lr=params['lr1'])
    model.compile(loss='categorical_crossentropy',optimizer=opt1,metrics=['acc'])
    print('Training - Part 1/2...')
    h1 = model.fit_generator(train_gen,
                              epochs=params['eps1'],
                              steps_per_epoch=len(train_df)//params['batch_size'],
                              shuffle=True,
                              verbose=params['verbose'],
                              callbacks=[checkpoint, schedule])
    #
    # Unfreeze weights, compile model, training (part 2)
    #                   
    model = unfreeze_all(model)
    opt2 = optimizers.Adam(lr=params['lr2'])
    model.compile(loss='categorical_crossentropy',optimizer=opt2,metrics=['acc'])
    print('Training - Part 2/2...')
    h2 = model.fit_generator(train_gen,
                             epochs=params['eps2'],
                             steps_per_epoch=len(train_df)//params['batch_size'],
                             shuffle=True,
                             verbose=params['verbose'],
                             callbacks=[checkpoint, schedule])
    #
    # Return trained model and training history
    #
    print('Training - Complete')
    train_hist = {'Part1':h1, 'Part2':h2}
    return model, train_hist


def mc_predict_image(test_gen, model, n):
    '''
    OBSOLETE: Use mc_predict_df which is faster and mode thread safe
    Monte Carlo (MC) dropout predictions for single image
    Inputs  : test_gen (keras image data generator for single image)
              model (trained keras model, loaded and compiled)
              n (# of predictions)
    Outputs : mean, variance
    '''
    mc_predictions = []
    for _i in range(n):
        y_p = model.predict_generator(test_gen,steps=1,verbose=0,workers=1)
        mc_predictions.append(y_p)
        # Time delay needed between predictions to prevent Keras thread safe warning
        time.sleep(0.1)  
    preds = np.array(mc_predictions)
    return np.mean(preds, axis=0)[0], np.var(preds, axis=0)[0]


def mc_predict_df(test_df, img_path, label_idxs, model, n, crop):
    '''
    Performs MC dropout predictions on test images;
    returns results
    Inputs  : test_df (dataframe of test images, filenames as "image")
              model (trained keras model, loaded and compiled)
              label_idxs (output of train_gen.class_indices)
              crop (str, "center" or "random" crop)
    Outputs : results_df
    '''
    num_classes = len(label_idxs)
    num_imgs = len(test_df)
    results_dict = {}
    # Cycle through images in test_df
    test_df = test_df.drop_duplicates()
    copy_df = test_df
    for i in range(0, num_imgs):
        print('Predicting image {0:4d} of {1:4d}'.format(i+1, num_imgs))
        img_df = copy_df.sample(n=1)
        copy_df = copy_df.drop(img_df.index)
        img_dict = {}
        img_dict['image'] = img_df.iloc[0]['image']
        if (crop == "center") or (crop == "random"):
            # Get image data generators
            img_batches = test_gen(img_df, img_path, num_classes, n)
            img_crops = crop_generator(img_batches, 224, crop, None, None, None)
            # Make n MC predictions, get probabilites and variances
            preds = model.predict_generator(img_crops,steps=n,verbose=1,workers=1)
            preds = np.array(preds)
        elif (crop == "pseudorandom"):
            preds = np.empty((0,num_classes))
            for ii in range(0, n):
                # Get image data generators
                img_batches = test_gen(img_df, img_path, num_classes, n)
                img_crops = crop_generator(img_batches, 224, crop, i+ii, None, None)
                # Make n MC predictions, get probabilites and variances
                cpreds = model.predict_generator(img_crops,steps=n,verbose=0,workers=1)
                cpreds = np.array(cpreds)
                preds = np.concatenate((preds, cpreds), axis=0)
        elif (crop == "adaptive random"):
            # Get image data generators
            hfw = get_hfw(img_dict['image'])
            img_batches = test_gen(img_df, img_path, num_classes, n)
            img_crops = crop_generator(img_batches, 224, crop, None, 6.13, hfw)
            # Make n MC predictions, get probabilites and variances
            preds = model.predict_generator(img_crops,steps=n,verbose=1,workers=1)
            preds = np.array(preds)
        elif (crop == "adaptive pseudorandom"):
            preds = np.empty((0,num_classes))
            hfw = get_hfw(img_dict['image'])
            for ii in range(0, n):
                # Get image data generators
                img_batches = test_gen(img_df, img_path, num_classes, n)
                img_crops = crop_generator(img_batches, 224, crop, i+ii, 6.13, hfw)
                # Make n MC predictions, get probabilites and variances
                cpreds = model.predict_generator(img_crops,steps=n,verbose=0,workers=1)
                cpreds = np.array(cpreds)
                preds = np.concatenate((preds, cpreds), axis=0)
        else:
            print("Invalid crop mode")
        # Convert predictions to numpy array, get prediction mean and variance
        probs = np.mean(preds, axis=0)
        uncs = np.var(preds, axis=0)
        # Append results to img_dict
        high_prob = np.argmax(probs)
        for key, value in label_idxs.items():
            if value == high_prob:
                pred = key
            img_dict[key+'_prob'] = probs[value]
            img_dict[key+'_unc'] = uncs[value]
        img_dict['sum_unc'] = np.sum(uncs)
        img_dict['entropy'] = shannon_entropy(probs)
        img_dict['true_label'] = img_df.iloc[0]['label']
        img_dict['pred_label'] = pred
        # Append results for image to results_dict
        results_dict[img_df.index[0]] = img_dict
    # Convert results dict to pandas dataframe
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    return results_df

