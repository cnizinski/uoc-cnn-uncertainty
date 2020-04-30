from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras import optimizers
from keras import metrics
from .dropout import unfreeze_all
import numpy as np
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
    wts = params['data_path'] + '/best_wts.h5'
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
    Monte Carlo (MC) dropout predictions for single image
    Inputs  : test_df (dataframe of training set)
              test_gen (keras image data generator)
              model (trained keras model)
              n (# of predictions)
    Outputs : mean, variance
    '''
    mc_predictions = []
    for _i in range(n):
        y_p = model.predict_generator(test_gen,steps=1,verbose=0,workers=1)
        mc_predictions.append(y_p)
        time.sleep(0.1)
    preds = np.array(mc_predictions)
    return np.mean(preds, axis=0), np.var(preds, axis=0)