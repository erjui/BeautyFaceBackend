import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from model import SegmentModel
model = SegmentModel()

ckpt_path = "checkpoints/epoch=10-step=10999.ckpt"
model.load_from_checkpoint(ckpt_path)

print(model)


def inference(event, context):
    pass

if __name__ == '__main__':
    inference(None, None)