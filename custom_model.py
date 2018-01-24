# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tqdm import tqdm
import zipfile, shutil
import cv2,h5py
import os, sys, glob
import itertools

from keras import backend
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, Lambda
from keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam, Adadelta, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, Callback, CSVLogger,  History, ModelCheckpoint, EarlyStopping
from keras.applications.resnet50 import ResNet50, preprocess_input as renet50_process
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_process
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_process
from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_process
from keras.applications.xception import Xception, preprocess_input as xception_process
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_resnet2_process

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def get_params_element(model):
    trainable = int(np.sum([backend.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable = int(np.sum([backend.count_params(p) for p in set(model.non_trainable_weights)]))
    return trainable, non_trainable

def vgg_16(img_height, img_width=None, dropout = 0.25):
    
    if not img_width:
        img_width = img_height
        
    base_model = VGG16(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    for layers in base_model.layers:
        layers.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    model_vgg16 = Model(inputs=base_model.input, outputs=x)
    model_vgg16.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    # model_vgg16.summary()
    print('VGG16 has %d layers.' % len(model_vgg16.layers))
    return model_vgg16



def resnet_50(img_height, img_width=None, dropout = 0.25):
    if not img_width:
        img_width = img_height
        
    base_model = ResNet50(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False)
    
    for layers in base_model.layers:
        layers.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    model_resnet50 = Model(inputs=base_model.input, outputs=x)
    model_resnet50.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    # model_resnet50.summary()
    print('ResNet50 has %d layers.' % len(model_resnet50.layers))
    return model_resnet50



def xception(img_height, img_width=None, dropout = 0.25):
    if not img_width:
        img_width = img_height
        
    input_tensor = Input(shape=(img_height, img_width, 3))
    input_tensor = Lambda(xception_process)(input_tensor)
    base_model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    for layers in base_model.layers:
        layers.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    model_xception = Model(inputs=base_model.input, outputs=x)
    model_xception.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    # model_xception.summary()
    print('Xception has %d layers.' % len(model_xception.layers))
    return model_xception



def inception_v3(img_height, img_width=None, dropout = 0.25):
    if not img_width:
        img_width = img_height
    input_tensor = Input(shape=(img_height, img_width, 3)) 
    input_tensor = Lambda(inception_v3_process)(input_tensor)
    base_model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    for layers in base_model.layers:
        layers.trainable = False
        
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    model_inceptionV3 = Model(inputs=base_model.input, outputs=x)
    model_inceptionV3.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

    # model_inceptionV3.summary()
    print('model_inceptionV3 has %d layers.' % len(model_inceptionV3.layers))
    return model_inceptionV3


def inception_resnetv2(img_height, img_width=None, dropout=0.25):
    if not img_width:
        img_width = img_height
    input_tensor = Input(shape=(img_height, img_width, 3))
    input_tensor = Lambda(inception_resnet2_process)(input_tensor)  # 以为加了，补上
    base_model = InceptionResNetV2(input_tensor=input_tensor, weights='imagenet', include_top=False)
    
    for layers in base_model.layers:
        layers.trainable = False
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    
    model_inceptionResNetV2 = Model(inputs=base_model.input, outputs=x)
    model_inceptionResNetV2.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    # model_inceptionResNetV2.summary()
    print('model_inceptionResNetV2 has %d layers.' % len(model_inceptionResNetV2.layers))
    return model_inceptionResNetV2




def fine_tuning(ft_model, X_train, y_train, X_valid, y_valid, img_height, freeze_layer, batch_size=64, epochs=3, dropout=0.25, weights=None):

    for layer in ft_model.layers:
        layer.trainable = True
    for layer in ft_model.layers[:freeze_layer]:
        layer.trainable = False
        
    ft_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    
    if weights:
        ft_model.load_weights(weights)

    logs_file = 'finetune-%s-{val_loss:.4f}.h5'%str(freeze_layer)
    path = os.getcwd()
    path_logs = os.path.join(path, logs_file)

    early_stop = EarlyStopping(monitor='val_loss', patience=2)
    model_check = ModelCheckpoint(path_logs, monitor='val_loss', save_best_only=True)
    
    ft_model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=[early_stop, model_check])
    return ft_model




def fine_tune(ft_model, preprocess, X_train, y_train, X_valid, y_valid, img_height, freeze_layer, batch_size=64, epochs=3, dropout=0.25, weights=None):

    x = Input(shape=(img_height, img_height, 3))
    x = Lambda(preprocess)(x)

    # Base Model
    base_model = ft_model(input_tensor=x, weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = True
    for layer in base_model.layers[:freeze_layer]:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)

    model = Model(inputs=base_model.input, outputs=x, name='Transfer_Learning')
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    print('Trainable: %d, Non-Trainable: %d' % get_params_element(model))
    
    if weights:
        model.load_weights(weights)
    
    logs_file = 'finetune-%s-{val_loss:.4f}.h5'%str(ft_model.__name__)
    path = os.getcwd()
    path_logs = os.path.join(path, logs_file)

    # early_stop = EarlyStopping(monitor='val_loss', patience=20)
    # learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=10, verbose=0,epsilon=0.0001,factor=0.5, min_lr=0)
    model_check = ModelCheckpoint(path_logs, monitor='val_loss', save_best_only=True)
    
    model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_valid, y_valid),
              callbacks=[model_check])
    return model



def get_features(ft_model, preprocess, img_height, train_data, test_data, train_label=None, batch_size=64):
    inputs = Input(shape=(img_height, img_height, 3))
    in_tensor = Lambda(preprocess)(inputs)
    base_model = ft_model(input_tensor=in_tensor, weights='imagenet', include_top=False)
    x = GlobalAveragePooling2D()(base_model.output)
    model = Model(inputs=inputs, outputs=x)
    train_feature = model.predict(train_data, batch_size=batch_size)
    test_feature = model.predict(test_data, batch_size=batch_size)
    
    with h5py.File("feature_%s.h5" % ft_model.__name__, 'w') as f:
        f.create_dataset('train', data=train_feature)
        f.create_dataset('test', data=test_feature)
        f.create_dataset('label', data=train_label)
    return train_feature, test_feature



class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        
    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        
    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()  


# def show_loss(Model):
#     fig, ax = plt.subplots(2,2)
#     ax = ax.flatten()
#     his_model = Model.history
#     history = his_model.history
    
#     ax[0].plot(history['acc'],color='b',label="Train acc")
#     legend = ax[0].legend(loc='best', shadow=True)
    
#     ax[1].plot(history['val_acc'],color='r',label="Validation acc")
#     legend = ax[1].legend(loc='best', shadow=True)
    
#     ax[2].plot(history['loss'],color='g',label="Train loss")
#     legend = ax[2].legend(loc='best', shadow=True)
#     ax[3].plot(history['val_loss'],color='c',label="Valid loss")
#     legend = ax[3].legend(loc='best', shadow=True)

def show_loss(Model):
    fig, ax = plt.subplots(2,1)
    his_model = Model.history
    history = his_model.history
    ax[0].plot(history['loss'], color='b', label="loss")
    ax[0].plot(history['val_loss'], color='r', label="val loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history['acc'], color='g', label="accuracy")
    ax[1].plot(history['val_acc'], color='c',label="val accuracy")
    legend = ax[1].legend(loc='best', shadow=True)