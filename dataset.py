import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from PIL import Image
from keras.preprocessing import image
import tensorflow




class ImageDataset(Dataset):
    def __init__(self, train, test):
        #self.csv = csv
        self.train = train
        self.test = test

        train_df = pd.read_csv('input/CheXpert-v1.0-small/train.csv')
        train_df = train_df.fillna(0)
        train_df = train_df.replace(-1, 1)
        valid_df = pd.read_csv('input/CheXpert-v1.0-small/valid.csv')
        test_df = pd.read_csv('input/CheXpert-v1.0-small/test.csv')
        self.train_image_names = train[:]['Path']
        self.valid_image_names = valid_df[:]['Path']
        self.test_image_names = test_df[:]['Path']

        # set the training data images and labels
        if self.train == True:
            print(f"Number of training images: {len(train.index)}")
            self.image_names = np.array(self.train_image_names)
            self.labels = np.array(train.drop(['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'],axis=1))
            # define the training transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=45),
                transforms.ToTensor(),
            ])
        # set the validation data images and labels
        elif self.train == False and self.test == False:
            print(f"Number of validation images: {len(valid_df.index)}")
            self.image_names = np.array(self.valid_image_names)
            self.labels = np.array(valid_df.drop(['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], axis=1))
            # define the validation transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((200, 200)),
                transforms.ToTensor(),
            ])
        # set the test data images and labels, only last 10 images
        # this, we will use in a separate inference script
        elif self.test == True and self.train == False:
            self.image_names = np.array(self.test_image_names)
            self.labels = np.array(test_df.drop(['Path', 'Sex', 'Age', 'Frontal/Lateral', 'AP/PA'], axis=1))
            # define the test transforms
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(f"input/{self.image_names[index]}")
        # convert the image from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # apply image transforms
        image = self.transform(image)
        targets = self.labels[index]
        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(targets, dtype=torch.float32)
        }