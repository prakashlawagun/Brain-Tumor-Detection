from django.shortcuts import render, redirect
import keras
from PIL import Image
import numpy as np
import os
from .models import Photo

model = keras.models.load_model('/home/linux/brain-tumor/BrainTumor/brain/templates/brain_tumor.h5')

def makepredictions(path):
    img = Image.open(path)

    img_d = img.resize((64, 64))

    if len(np.array(img_d).shape) < 4:
        rgb_img = Image.new("RGB", img_d.size)
        rgb_img.paste(img_d)
    else:
        rgb_img = img_d
    
    rgb_img = np.array(rgb_img, dtype=np.float64)
    rgb_img = rgb_img.reshape(1, 64, 64, 3)

    predictions = model.predict(rgb_img)
    if predictions[0] == 1:
        return "Tumor"
    else:
        return "No Tumor"

def index(request):
    if request.method == 'POST' and 'upload' in request.FILES:
        f = request.FILES['upload']
        uploaded_image = Photo(image=f)
        uploaded_image.save()

        file_url = uploaded_image.image.url
        predictions = makepredictions(uploaded_image.image.path)
        
        return render(request, "index.html", {"pred": predictions, "file_url": file_url})
    elif request.method == 'POST':
        err = "No image selected"
        return render(request, "index.html", {'err': err})
    else:
        return render(request, "index.html")
