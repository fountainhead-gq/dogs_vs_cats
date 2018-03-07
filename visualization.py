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


def visualise_image(path_img):
    plt.figure(figsize=(15, 12))
    for i in range(15):
        plt.subplot(3, 5, i+1)
        img = load_img(os.path.join(path_img, os.listdir(path_img)[i]))
        plt.title(os.listdir(path_img)[i])
        plt.imshow(img) 	
		
		
# predict_img(model_inceptionV3, path_test, X_test_299.shape[0])
def predict_img(MODE, path_test, X_test_num, img_size=None):
    import random
    if not img_size:
        img_size = (299, 299)
        
    plt.figure(figsize=(15, 12))
    for i in range(1, 21): 
        img = cv2.imread(os.path.join(path_test, '%d.jpg'% random.randint(1, X_test_num)))
        img = cv2.resize(img, img_size)
        x = img.copy()
        x.astype(np.float32)
        prediction = MODE.predict(np.expand_dims(x, axis=0))[0]
        plt.subplot(4, 5, i)
        
        if prediction < 0.5:
            plt.title('cat %.2f%%' % (100 - prediction*100))
        else:
            plt.title('dog %.2f%%' % (prediction*100))
        
        plt.axis('off')
        plt.imshow(x[:,:,::-1]) # convert BGR to RGB		