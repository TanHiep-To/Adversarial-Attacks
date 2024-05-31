import imagenet_stubs
import numpy as np
import os
from imagenet_stubs.imagenet_2012_labels import name_to_label, label_to_name
from keras.preprocessing import image

def load_images:
    
    images_list = []
    for (i,image_path) in enumerate(imagenet_stubs.get_image_paths()):
        img = image.load_img(image_path, target_size=(224, 224))
        img_name = image_path.split('\\')[-1]
        img.save('../data/images/' + img_name)
    
load_images()