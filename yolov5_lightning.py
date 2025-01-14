import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import pytorch_lightning as pl
from torch.optim import Adam
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint

# Dataset Directory Structure:

# CustomDataset Class: fetch individual image-label pairs (given an index)


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data, torch.utils.data.Subset):
            idx = self.data.indices[idx]
            actual_data = self.data.dataset
        else:
            actual_data = self.data

        img_path = actual_data.iloc[idx]["image_path"]
        label_value = actual_data.iloc[idx]["label"]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label_value


# CustomDataModule Class: split the dataset into training and validation subsets and provides DataLoaders for them


class CustomDataModule(pl.LightningDataModule):
    def __init__(
        self, image_folder, label_file, transform, batch_size=4, val_split=0.2, test_split=0.1
    ):
        super(CustomDataModule, self).__init__()
        self.image_folder = image_folder
        self.label_file = label_file
        self.batch_size = batch_size
        self.transform = transform

        # Read labels into a DataFrame and add full image paths
        all_data = pd.read_csv(label_file)
        all_data["image_path"] = all_data["filename"].apply(
            lambda x: os.path.join(self.image_folder, x)
        )

        train_len = int((1.0 - val_split - test_split) * len(all_data))
        val_len = int(val_split * len(all_data))
        
        self.train_data, self.val_data, self.test_data = torch.utils.data.random_split(
            all_data, [train_len, val_len, len(all_data) - train_len - val_len]
        )

    def train_dataloader(self):
        return DataLoader(
            CustomDataset(self.train_data, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            CustomDataset(self.val_data, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            CustomDataset(self.test_data, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
        )


# Model architecture: custom regression model built on top of a pretrained YOLOv5 backbone


class YOLOv5Regression(pl.LightningModule):
    def __init__(self):
        super(YOLOv5Regression, self).__init__()
        self.yolov5_backbone = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True
        )

        self.features = nn.Sequential(
            *list(self.yolov5_backbone.model.children())[:-1][:24]
        )

        # a custom head for regression
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 60 * 40, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )
        self.criterion = nn.MSELoss()

        # Initialize val_losses as an empty list
        self.val_losses = []

    def forward(self, x):
        x = self.features(x)
        return self.regression_head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(
            x
        ).squeeze()  # You might need to squeeze your outputs to match target dimensions
        loss = self.criterion(y_hat, y.float())  # Ensure y is a float tensor
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        self.val_losses.append(loss.detach())
        return {"val_loss": loss.detach()}  # detach the loss from the computation graph

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(torch.tensor(self.val_losses))
        self.log("avg_val_loss", avg_loss)
        self.val_losses = [] 

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
    
        if not hasattr(self, 'test_losses'):
            self.test_losses = []
        self.test_losses.append(loss.detach())
    
        return {"test_loss": loss.detach()}
    
    
    def on_test_epoch_end(self):
        if hasattr(self, 'test_losses'):
            avg_test_loss = torch.stack(self.test_losses).mean()
            print(f"Average Test Loss: {avg_test_loss}")
            
            del self.test_losses

transform = transforms.Compose(
    [
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

PATH_DATA = os.path.join("/home", "shihong", "yolov5_lightning", "data", "mock_lightning", "train")
PATH_IMAGE = os.path.join(PATH_DATA, "image")
PATH_LABEL = os.path.join(PATH_DATA, "label.csv")

# Initialize the CustomDataModule
data_module = CustomDataModule(
    image_folder=PATH_IMAGE, label_file=PATH_LABEL, transform=transform, batch_size=4
)

model = YOLOv5Regression()

checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",  # Save the model with the best validation loss
    dirpath="/home/shihong/yolov5_lightning/data/mock_lightning/saved_models/",  # Directory where you want to save the model
    filename="model-{epoch:02d}-{avg_val_loss:.2f}",  # Naming scheme
    save_top_k=3,  # Save only the top 3 models
    mode="min",  # 'min' for metrics where lower is better (like loss), 'max' for metrics where higher is better (like accuracy)
)

# Train the model
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=data_module)

# Evaluate the model on the test dataset
trainer.test(model, datamodule=data_module)
