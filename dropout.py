from keras import backend as K
from keras.models import Sequential, Model, Input
from keras.applications import VGG16, ResNet50
from keras.layers import Dense, Dropout, Flatten, concatenate
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D
from classification_models.keras import Classifiers
import numpy as np


def dropout_layer(input_tensor, p=0.5, mc=False):
    '''
    Returns dropout layer for keras models
    Inputs  : input_tensor, p(do probability), mc(bool)
    Outputs : keras layer
    '''
    if mc is True:
        return Dropout(p)(input_tensor, training=True)
    else:
        return Dropout(p)(input_tensor)


def get_VGG16(num_classes, img_size, mc_inf, p):
    '''
    Returns keras model capable of MC dropout during inference
    Inputs  : num_classes (int, # of classes)
              img_size (tuple, image size, default=(224,224,3))
              mc_inf (bool, keep dropout on inference? default=True)
              p (float (0-1), dropout probability )
    Outputs : keras model
    '''
    # Get base model
    model = VGG16(weights='imagenet', include_top=True)
    inp = Input(img_size)
    x = inp
    # Add dropout layers after conv and pooling layers
    for idx in range(1,len(model.layers)-3):
        x = model.layers[idx](x)
        #if 'conv' in model.layers[idx].name and mc_inf is True:
        #    x = Dropout(p)(x, training=True)
        #elif 'pool' in model.layers[idx].name and mc_inf is True:
        #    x = Dropout(p)(x, training=True)
        #else:
        #    pass
    # Return model with num_classes
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dropout(p)(x, training=mc_inf)
    x = Dense(512, activation='relu', name='fc2')(x)
    out = Dense(num_classes, activation='softmax', name='predictions')(x)
    return Model(inputs=inp, outputs=out)


def get_ResNet50(num_classes, img_size, mc_inf, p, frozen):
    '''
    Returns keras model capable of MC dropout during inference
    ResNet50 from keras.applications
    Inputs  : num_classes (int, # of classes)
              img_size (tuple, image size, default=(224,224,3))
              mc_inf (bool, keep dropout on inference? default=True)
              p (float (0-1), dropout probability )
              frozen
    Outputs : keras model
    '''
    # Get base model
    base = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)
    if frozen is True:
        for layer in base.layers:
            layer.trainable == False
    else:
        for layer in base.layers:
            layer.trainable == True
    # Add top
    pool1 = GlobalMaxPooling2D(name='GMP')(base.output)
    pool2 = GlobalAveragePooling2D(name='GAP')(base.output)
    head = concatenate([pool1, pool2])
    head = Dropout(p)(head, training=mc_inf)
    head = Dense(400, name='fc')(head)
    out = Dense(num_classes, activation='softmax', name='predictions')(head)
    return Model(inputs=base.input, outputs=out)


def get_ResNet18(num_classes, img_size, mc_inf, p, frozen):
    '''
    Returns keras model capable of MC dropout during inference
    ResNet18 from classifier zoo
    Inputs  : num_classes (int, # of classes)
              img_size (tuple, image size, default=(224,224,3))
              mc_inf (bool, keep dropout on inference? default=True)
              p (float (0-1), dropout probability )
              frozen
    Outputs : keras model
    '''
    # Get base model
    ResNet18, _pro = Classifiers.get('resnet18')
    base = ResNet18(weights='imagenet', include_top=False, input_shape=img_size)
    if frozen is True:
        for layer in base.layers:
            layer.trainable == False
    else:
        for layer in base.layers:
            layer.trainable == True
    # Add top
    pool1 = GlobalMaxPooling2D(name='GMP')(base.output)
    pool2 = GlobalAveragePooling2D(name='GAP')(base.output)
    head = concatenate([pool1, pool2])
    if mc_inf is True:
        head = Dropout(p)(head, training=True)
    else:
        head = Dropout(p)(head)
    head = Dense(1000, name='fc')(head)
    out = Dense(num_classes, activation='softmax', name='predictions')(head)
    return Model(inputs=base.input, outputs=out)


def get_ResNet34(num_classes, img_size, mc_inf, p, frozen):
    '''
    Returns keras model capable of MC dropout during inference
    ResNet34 from classifier zoo
    Inputs  : num_classes (int, # of classes)
              img_size (tuple, image size, default=(224,224,3))
              mc_inf (bool, keep dropout on inference? default=True)
              p (float (0-1), dropout probability )
              frozen
    Outputs : keras model
    '''
    # Get base model
    ResNet34, _pro = Classifiers.get('resnet34')
    base = ResNet34(weights='imagenet', include_top=False, input_shape=img_size)
    if frozen is True:
        for layer in base.layers:
            layer.trainable == False
    else:
        for layer in base.layers:
            layer.trainable == True
    # Add top
    pool1 = GlobalMaxPooling2D(name='GMP')(base.output)
    pool2 = GlobalAveragePooling2D(name='GAP')(base.output)
    head = concatenate([pool1, pool2])
    if mc_inf is True:
        head = Dropout(p)(head, training=True)
    else:
        head = Dropout(p)(head)
    head = Dense(1000, name='fc')(head)
    out = Dense(num_classes, activation='softmax', name='predictions')(head)
    return Model(inputs=base.input, outputs=out)


def get_ResNet50x(num_classes, img_size, mc_inf, p, frozen):
    '''
    Returns keras model capable of MC dropout during inference
    ResNet50 from classifier zoo
    Inputs  : num_classes (int, # of classes)
              img_size (tuple, image size, default=(224,224,3))
              mc_inf (bool, keep dropout on inference? default=True)
              p (float (0-1), dropout probability )
              frozen
    Outputs : keras model
    '''
    # Get base model
    ResNet50, _pro = Classifiers.get('resnet50')
    base = ResNet50(weights='imagenet', include_top=False, input_shape=img_size)
    if frozen is True:
        for layer in base.layers:
            layer.trainable == False
    else:
        for layer in base.layers:
            layer.trainable == True
    # Add top
    head = GlobalMaxPooling2D(name='GMP')(base.output)
    #pool1 = GlobalMaxPooling2D(name='GMP')(base.output)
    #pool2 = GlobalAveragePooling2D(name='GAP')(base.output)
    #head = concatenate([pool1, pool2])
    if mc_inf is True:
        head = Dropout(p)(head, training=True)
    else:
        head = Dropout(p)(head)
    head = Dense(400, name='fc')(head)
    out = Dense(num_classes, activation='softmax', name='predictions')(head)
    return Model(inputs=base.input, outputs=out)


def unfreeze_all(model):
    '''
    Sets all layers of model to trainable
    '''
    new_model = model
    for layer in new_model.layers:
        layer.trainable == True
    return new_model