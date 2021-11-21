import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torch.optim.lr_scheduler import MultiStepLR

class SegmentModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.save_hyperparameters()

        self.model = smp.FPN(
            encoder_name='resnet34',
            encoder_weights='imagenet',
            in_channels=3,
            classes=19,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.model(x)
        x = torch.argmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.criterion(y_hat, y)

        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=7e-3)
        scheduler = MultiStepLR(optimizer, milestones=[10, 60, 90], gamma=0.1)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "valid_loss",
            }
        }