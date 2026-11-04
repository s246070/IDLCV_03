import torch
import os
import glob
from PIL import Image
import torchvision.transforms as transforms

# DATA_PATH = '/dtu/datasets1/02516/phc_data'

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

if __name__ == "__main__":
    size = 128
    train_transform = transforms.Compose([transforms.Resize((size, size)),
                                    transforms.ToTensor()])
    trainset = DRIVE(train=True, transform=train_transform)
    print("Number of samples:", len(trainset))            # >0 ?
    print("First 3 image paths:", trainset.image_paths[:3])
    print("First 3 label paths:", trainset.label_paths[:3])

    X, Y = trainset[0]
    print("Image tensor:", X.shape, X.dtype, X.min().item(), X.max().item())
    print("Label tensor:", Y.shape, Y.dtype, Y.unique())