# -*- coding: utf-8 -*-


!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle

!kaggle datasets download -d ahmedhamada0/brain-tumor-detection

import zipfile
zip_ref = zipfile.ZipFile('/content/brain-tumor-detection.zip','r')
zip_ref.extractall()
zip_ref.close()

import os 
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras .utils import normalize
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Activation,Flatten

image_directory='/content/'
no_tumor_image = os.listdir(image_directory+'no/')
yes_tumor_image = os.listdir(image_directory+'yes/')
print(no_tumor_image)

print(yes_tumor_image)

datasets = []
label = []
for i,image_name in enumerate(no_tumor_image):
  if(image_name.split('.')[1]=='jpg'):
    image=cv2.imread(image_directory+'no/'+image_name)
    image= Image.fromarray(image,"RGB")
    image = image.resize((64,64))
    datasets.append(np.array(image))
    label.append(0)

for i,image_name in enumerate(yes_tumor_image):
  if(image_name.split('.')[1]=='jpg'):
    image=cv2.imread(image_directory+'yes/'+image_name)
    image= Image.fromarray(image,"RGB")
    image = image.resize((64,64))
    datasets.append(np.array(image))
    label.append(1)

print(len(datasets))
print(len(label))

datasets=np.array(datasets)
labels =np.array(label)

X_train,X_test,Y_train,Y_test = train_test_split(datasets,labels,test_size=0.2,random_state=0)

print(datasets.shape)
print(X_train.shape)
print(X_test.shape)

X_train = normalize(X_train,axis=1)
X_test = normalize(X_test,axis=1)

model = Sequential()
input_size = 64

model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size=(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

history = model.fit(X_train,Y_train,
  batch_size=16,
  verbose=1,
  epochs=10,
  validation_data=(X_test,Y_test),
  shuffle=False)

loss,accuracy = model.evaluate(X_test,Y_test)
print(accuracy)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'],label='train_accuracy')
plt.plot(history.history['val_accuracy'],label='val_accuracy')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

plt.plot(history.history['loss'],label='train_loss')
plt.plot(history.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

image = cv2.imread('/content/images.jpeg')
image = Image.fromarray(image)
img = image.resize((64,64))
img = np.array(img)
input_image = np.expand_dims(img,axis=0)
result = model.predict(input_image)
if result[0] == 1:
  print("The person is suffer from brain tumors")
else:
  print("The person is not suffer from brain tumors")

model.save('brain_tumor_model.h5')

