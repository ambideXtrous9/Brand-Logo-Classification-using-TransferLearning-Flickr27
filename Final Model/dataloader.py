import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
import config
from PIL import Image


# initialize our data augmentation functions
resize = transforms.Resize(size=(config.WIDTH,config.HEIGHT))
hFlip = transforms.RandomHorizontalFlip(p=0.25)
vFlip = transforms.RandomVerticalFlip(p=0.25)
rotate = transforms.RandomRotation(degrees=15)
coljtr = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
raf = transforms.RandomAffine(degrees=40, translate=None, scale=(1, 2), shear=15)
rrsc = transforms.RandomResizedCrop(size=config.WIDTH, scale=(0.8, 1.0))
ccp  = transforms.CenterCrop(size=config.WIDTH)  # Image net standards
nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards

trainTransforms = transforms.Compose([resize,hFlip,vFlip,rotate,raf,rrsc,ccp,coljtr,transforms.ToTensor(),nrml])
valTransforms = transforms.Compose([resize,transforms.ToTensor(),nrml])

class LogoDataModule(pl.LightningDataModule):
    def __init__(self, data_folder, batch_size=32, num_workers=4, val_split=0.2):
        super(LogoDataModule, self).__init__()
        self.data_folder = data_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.train_transform = trainTransforms
        self.val_transform = valTransforms

    def setup(self, stage=None):
        # Create dataset without transformations
        self.dataset = ImageFolder(root=self.data_folder)

        # Split dataset into training and validation sets
        val_size = int(len(self.dataset) * self.val_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # Apply transformations to the datasets
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        # For testing, you can use the validation set or a separate test set if available
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


