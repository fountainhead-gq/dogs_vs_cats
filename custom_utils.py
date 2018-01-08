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


# train
def get_train_data(img_width, img_height=None):
    if not img_height:
        img_height = img_width 
    
    path = os.getcwd()
    path_cat = os.path.join(path, r'cat')
    path_dog = os.path.join(path, r'dog')

    cat_size = len(glob.glob(path_cat+'/*.jpg'))
    dog_size = len(glob.glob(path_dog+'/*.jpg'))
    train_size = cat_size + dog_size
    
    X_train =  np.zeros((train_size, img_width, img_height, 3), dtype=np.uint8)
    y_train = np.array([0] * cat_size + [1] * dog_size)
    i_train = 0

    for filenames in tqdm(os.listdir(path_cat)): 
        img = cv2.imread(os.path.join(path_cat, filenames))
        X_train[i_train] = cv2.resize(img,(img_width, img_height))
        i_train += 1
    

    for filenames in tqdm(os.listdir(path_dog)): 
        img = cv2.imread(os.path.join(path_dog, filenames))
        X_train[i_train] = cv2.resize(img,(img_width, img_height))
        i_train += 1  
    return X_train, y_train


# test 
def get_test_data(img_width, img_height=None):
    if not img_height:
        img_height = img_width
    
    path = os.getcwd()
    path_test = os.path.join(path, r'test')
    test_size = len(glob.glob(path_test+'/*.jpg'))
    
    X_test = np.zeros((test_size, img_width, img_height, 3), dtype=np.uint8)
    j_test = 0
    for j_test in tqdm(range(test_size)):
        get_img = path_test + r'/%d.jpg' % (j_test+1)
        img = cv2.imread(get_img)
        X_test[j_test] = cv2.resize(img, (img_width, img_height))
    return X_test

