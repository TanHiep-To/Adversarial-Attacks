import os
import numpy as np
import warnings
warnings.filterwarnings('ignore') 
from PIL import Image
from keras.preprocessing import image

class Dataset():
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = self.load_data()
        self.labels = self.load_labels()

    def load_data(self):
        images_list = []
        for image_path in os.listdir(self.data_dir):
            img = image.load_img(self.data_dir + image_path, target_size=(224, 224))
            img = image.img_to_array(img)
            images_list.append(img)
            
        data = np.array(images_list)
        data = data/255
        return data
    
    def load_labels(self):
        labels = []
        for image_path in os.listdir(self.data_dir):
            label = image_path.split('.jpg')[0]
            labels.append(label)
        return labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

