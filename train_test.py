from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import optimizers
from keras import metrics
from .dropout import unfreeze_all
from .preprocessing import test_gen, crop_generator
from .helpers import shannon_entropy
import numpy as np
import pandas as pd
import time


def lr_decay(epoch, init_lr):
    # Learning rate decay scheduler
    return init_lr/(1.0 + epoch*0.01)


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
    wts = params['data_path'] + '/temp_wts_best.h5'
    checkpoint = ModelCheckpoint(wts,monitor='loss',verbose=0,save_best_only=True,mode='min')
    #stopping = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=0)
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1.0e-5)
    schedule = LearningRateScheduler(lr_decay)
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
    results_dict = {}
    # Cycle through images in test_df
    test_df = test_df.drop_duplicates()
    copy_df = test_df
    for _i in range(0, len(test_df)):
        img_df = copy_df.sample(n=1)
        copy_df = copy_df.drop(img_df.index)
        img_dict = {}
        img_dict['image'] = img_df.iloc[0]['image']
        # Get image data generators
        img_batches = test_gen(img_df, img_path, num_classes, n)
        img_crops = crop_generator(img_batches, 224, crop)
        # Make n MC predictions, get probabilites and variances
        preds = model.predict_generator(img_crops,steps=n,verbose=0,workers=1)
        preds = np.array(preds)
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

