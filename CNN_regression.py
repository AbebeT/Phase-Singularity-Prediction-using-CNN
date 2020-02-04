# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:34:03 2019

@author: User
"""

#KERAS

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from keras.optimizers import Adam
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
import models
from sklearn.model_selection import train_test_split

path = r'dataset'
path2 = r'processed'
label_path = r'C:\Users\User\Desktop\Research\heart_het'

#input_image dimensions
img_rows, img_cols = 64, 64

# number of channels
img_channels = 1

listing = os.listdir(path)
num_samples=size(listing)
print (num_samples)

for file in listing :
    im = Image.open(path+'\\'+file)
    img = im.resize((img_rows,img_cols))
    gray = img.convert('L')
    gray.save(path2 +'\\' +  file, "JPEG")
    
imlist = os.listdir(path2)
# create matrix to store all flattened images
immatrix = array([array(Image.open(path2 + '\\' + im2)).flatten()
              for im2 in imlist],'f')
    
    


label = pd.read_csv(label_path+'\\'+'label.csv')
label_scaler = MinMaxScaler(feature_range = (0,1))
label = label_scaler.fit_transform(label)
data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

img=immatrix[65].reshape(img_rows,img_cols)
#plt.imshow(img)
#plt.imshow(img,cmap='gray')



(X, y) = (train_data[0],train_data[1])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')




model = models.create_cnn(64, 64, 1, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test),
	epochs=200, batch_size=15)
