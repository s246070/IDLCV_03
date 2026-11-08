import torch
import os
import glob
from PIL import Image

class DRIVE(torch.utils.data.Dataset):
    def __init__(self, train, transform):
        'Initialization'
        self.transform = transform
        self.DATA_PATH = '/dtu/datasets1/02516/DRIVE'
        if train:
            data_path = os.path.join(self.DATA_PATH, 'training')
            self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
            self.label_paths = sorted(glob.glob(data_path + '/1st_manual/*.gif'))
        else:
            data_path = os.path.join(self.DATA_PATH, 'test')
            self.image_paths = sorted(glob.glob(data_path + '/images/*.tif'))
            self.label_paths = sorted(glob.glob(data_path + '/mask/*.gif'))


    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y

class PH2(torch.utils.data.Dataset):
    def __init__(self, transform):
        'Initialization'
        self.transform = transform
        self.DATA_PATH = '/dtu/datasets1/02516/PH2_Dataset_images'
        # data_path = os.path.join(self.DATA_PATH)
        self.image_paths = sorted(glob.glob(self.DATA_PATH +  "/IMD*/IMD*_Dermoscopic_Image/*.bmp"))
        self.label_paths = sorted(glob.glob(self.DATA_PATH + "/IMD*/IMD*_lesion/*.bmp"))

    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(image_path)
        label = Image.open(label_path)
        Y = self.transform(label)
        X = self.transform(image)
        return X, Y