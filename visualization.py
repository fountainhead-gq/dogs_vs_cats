# coding: utf-8

import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm



# pick out the small size  pictures < 100
def pick_small_size(path):
    img_list = os.listdir(path)
    small_size = []
    for img_name in tqdm(img_list):
        img = load_img(os.path.join(path, img_name))
        x = img_to_array(img)
        if x.shape[0] < 100 and x.shape[1] < 100:
            small_size.append(img_name)
    return small_size
    

    
# explore the size of the pictures
def exp_pic_size(path):
    heights, widths = [], []
    for filenames in tqdm(os.listdir(path)):
        img = load_img(os.path.join(path, filenames))
        x = img_to_array(img) # this is a Numpy array with shape (n_c, n_w, n_h)
        heights.append(x.shape[0])
        widths.append(x.shape[1])
        
    x = np.array(widths)
    y = np.array(heights)
    
    plt.scatter(x, y, c='r', alpha=1, marker = 'o')
    plt.title('the size of pictures')
    plt.xlabel('width')
    plt.ylabel('height')
    plt.show()

