
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

from my_dl_framework.training.utils import get_lossfunction, get_model, get_optimizer

class PLWrapper(pl.LightningModule):
    def __init__(self, config: Dict, 
                 training_set_len: int = 0,
                 clearml_logger = None):
        self.config = config
        self.modules = get_model(config=self.config)
        self.loss_function = get_lossfunction(config=self.config)
        self.clearml_logger = clearml_logger
        self.training_set_len = training_set_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.modules(x)
        return res

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        outputs = self.model(images)
        loss = self.loss_function(outputs, targets)
        # Track loss
        if batch_idx % self.config["loss_log_freq"] == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            if self.clearml_logger is not None:
                self.clearml_logger.report_scalar(title="Loss", 
                                                  series="Train Loss", 
                                                  value=loss.detach().cpu().numpy(),
                                                  iteration=self.current_epoch * self.training_set_len + batch_idx)
        return loss

    def validation_step(self, val_batch, batch_idx):
        images, targets = val_batch
        outputs = F.softmax(model(images), dim=1)

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
    
