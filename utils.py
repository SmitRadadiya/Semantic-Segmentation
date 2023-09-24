import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class MakeData(Dataset):

    def __init__(self, path) -> None:
        self.img_dir = path
        self.images = os.listdir(path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = cv2.imread(self.img_dir+self.images[index])
        x = img[0:256, 0:256]  #see the image
        y = img[0:256, 256:512]
        x = TF.to_tensor(x)
        x = x[:3]
        y = TF.to_tensor(y)
        y = y[:3]
        return x, y
        


# def get_data(path):
#     x = []
#     y = []

#     for image in os.listdir(path):
#         img = cv2.imread(path+image)
#         i_train = img[0:256, 0:256]  #see the image
#         i_val = img[0:256, 256:512]
#         x.append(i_train)
#         y.append(i_val)
    
#     return x, y
    pass


