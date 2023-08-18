
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

# Datasets

""" 

- Dataset Directory Structure:
train/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels.csv 

- Labels File Format:
image_name,label_value
image1.jpg,23.4
image2.jpg,45.1
...

"""

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Check if self.data is a Subset
        if isinstance(self.data, torch.utils.data.Subset):
            idx = self.data.indices[idx]  # Convert Subset index to original DataFrame index
            actual_data = self.data.dataset
        else:
            actual_data = self.data

        # Now access the data
        img_path = actual_data.iloc[idx]["image_path"]
        label_value = actual_data.iloc[idx]["label"]

        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label_value


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, image_folder, label_file, transform, batch_size=4, val_split=0.2):
        super(CustomDataModule, self).__init__()
        self.image_folder = image_folder
        self.label_file = label_file
        self.batch_size = batch_size
        self.transform = transform

        # Read labels into a DataFrame and add full image paths
        all_data = pd.read_csv(label_file)
        all_data["image_path"] = all_data["filename"].apply(lambda x: os.path.join(self.image_folder, x))

        train_len = int((1.0 - val_split) * len(all_data))
        self.train_data, self.val_data = torch.utils.data.random_split(all_data, [train_len, len(all_data) - train_len])

    def train_dataloader(self):
        return DataLoader(CustomDataset(self.train_data, transform=self.transform), batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(CustomDataset(self.val_data, transform=self.transform), batch_size=self.batch_size, shuffle=False)




# Model architecture

class YOLOv5Regression(pl.LightningModule):
    def __init__(self):
        super(YOLOv5Regression, self).__init__()
        self.yolov5_backbone = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        
        # Use only the layers up to index 23 (excluding the detection head)
        self.features = nn.Sequential(*list(self.yolov5_backbone.model.children())[:-1][:24])
        
        self.regression_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 60 * 40, 512),  # Adjusted the input size
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
        y_hat = self(x).squeeze()  # You might need to squeeze your outputs to match target dimensions
        loss = self.criterion(y_hat, y.float())  # Ensure y is a float tensor
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch  
        y_hat = self(x).squeeze()
        loss = self.criterion(y_hat, y.float())
        if not hasattr(self, "val_losses"):
            self.val_losses = []
        self.val_losses.append(loss)
        self.log('val_loss_step', loss)
        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = torch.mean(torch.stack(self.val_losses))
        self.log('avg_val_loss', avg_loss)
        # reset the val_losses for the next epoch
        self.val_losses = []

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


#----- Now you are the user -----#

transform = transforms.Compose([
    transforms.Resize((640, 640)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

PATH_DATA = os.path.join("/home", "shihong", "data", "mock_lightning", "train")
PATH_IMAGE = os.path.join(PATH_DATA, "image")
PATH_LABEL = os.path.join(PATH_DATA, "label.csv")

# Initialize the CustomDataModule
data_module = CustomDataModule(image_folder=PATH_IMAGE, label_file=PATH_LABEL, transform=transform, batch_size=4)

model = YOLOv5Regression() 

checkpoint_callback = ModelCheckpoint(
    monitor='avg_val_loss',   # Save the model with the best validation loss
    dirpath='/home/shihong/data/mock_lightning/saved_models/',  # Directory where you want to save the model
    filename='model-{epoch:02d}-{avg_val_loss:.2f}',  # Naming scheme
    save_top_k=3,  # Save only the top 3 models (based on the monitor metric, i.e., 'avg_val_loss' here)
    mode='min',  # 'min' for metrics where lower is better (like loss), 'max' for metrics where higher is better (like accuracy)
)

# Train the model
trainer = pl.Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, datamodule=data_module)


# After training
trainer.fit(model, datamodule=data_module)

# Evaluate the model on the test dataset
trainer.test(model, datamodule=data_module)