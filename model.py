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
            encoder_name='timm-efficientnet-b4',
            encoder_weights=None,
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


if __name__ == "__main__":
    import cv2
    import numpy as np
    from torchvision.transforms import transforms

    normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalize)
    ])
    

    ckpt_path = "checkpoints/epoch=89-step=224999.ckpt"
    model = SegmentModel.load_from_checkpoint(ckpt_path)
    print('pretrained model loaded')

    img = cv2.imread("base64_sample/0.jpg")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)

    out = model(img).permute(1, 2, 0)
    out = out.detach().cpu().numpy()
    out = np.uint8(out)

    print(out.shape)
    print(out)

    cv2.namedWindow("out", cv2.WINDOW_NORMAL)
    cv2.imshow("out", out * 20)
    cv2.waitKey(0)

    print(img.shape)
    print(out.shape)

