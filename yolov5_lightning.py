
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn as nn  


# Datasets

class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file=None, labels=None, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        if label_file:
            self.labels = pd.read_csv(label_file)
        elif labels is not None:
            self.labels = labels
        else:
            raise ValueError("Either label_file or labels should be provided.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        value = int(self.labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, value

transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
])

image_folder_path = "/home/shihong/data/mock_lightning/train/image"
label_file_path = "/home/shihong/data/mock_lightning/train/label.csv"

dataset = CustomDataset(image_folder=image_folder_path, label_file=label_file_path, transform=transform)


# Loader

train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Module

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, train_image_folder, train_label_file, batch_size=32, val_split=0.2):
        super(CustomDataModule, self).__init__()
        self.image_folder = train_image_folder
        self.label_file = train_label_file
        self.batch_size = batch_size

        full_dataset = CustomDataset(image_folder=self.image_folder, label_file=self.label_file)
        train_len = int((1.0 - val_split) * len(full_dataset))
        val_len = len(full_dataset) - train_len
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)


# Model architecture

class YOLOv5Regression(pl.LightningModule):
    def __init__(self):
        super(YOLOv5Regression, self).__init__()
        self.yolov5_backbone = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.features = self.yolov5_backbone.model[0]
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1280 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        self.criterion = nn.MSELoss()

    def forward(self, x):
        x = self.features(x)
        return self.regression_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x)  
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


data_module = CustomDataModule(train_image_folder="/home/shihong/data/mock_lightning/train/image", train_label_file="/home/shihong/data/mock_lightning/train/label.csv")
model = YOLOv5Regression() 
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule=data_module)