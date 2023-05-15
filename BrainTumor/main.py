from joblib import load
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

model = load('/home/linux/brain-tumor/BrainTumor/brain/templates/model (1).joblib')

image = cv2.imread('a-Original-MRI-brain-tumor-image-b-Colored-MRI-image.png')
image = Image.fromarray(image)
img = image.resize((64,64))
img = np.array(img)
input_image = np.expand_dims(img,axis=0)
result = model.predict(input_image)
print(result)