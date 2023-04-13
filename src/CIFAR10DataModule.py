import pytorch_lightning as pl 
import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import random_split, DataLoader

pl.seed_everything(42)

class CIFAR10DataModule(pl.LightningDataModule):
  
  def __init__(
    self, 
    data_dir="../data", 
    batch_size=128, 
    num_workers=int(os.cpu_count() / 2)
  ):
    
    super().__init__()
    
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.transform = transforms.Compose([transforms.ToTensor()])

  def prepare_data(self):
    datasets.CIFAR10(root=self.data_dir, train=True, download=True)
    datasets.CIFAR10(root=self.data_dir, train=False, download=True)

  def setup(self, stage=None):
    if stage == "fit" or stage is None:
      cifar10_full = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform)
      self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [0.8, 0.2])

      data = self.cifar10_train.dataset.data / 255 # data is numpy array
      self.mean = data.mean(axis = (0,1,2)) 
      self.std = data.std(axis = (0,1,2))

      print(f"Train split mean: {self.mean}")
      print(f"Train split std : {self.std}")

    if stage == "test" or stage is None:
      self.cifar10_test = datasets.CIFAR10(self.data_dir, train=False, transform=self.transform)

  def train_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)
  

def __main__():
  import plotly.express as px

  BATCH_SIZE = 25000

  cifar10_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE)
  cifar10_datamodule.prepare_data()
  cifar10_datamodule.setup()

  train_dl = cifar10_datamodule.train_dataloader()

  for x, y in train_dl:

    fig = px.imshow(x[1].permute(1, 2, 0))
    fig.show()

    pass


if __name__ == "__main__":
  __main__()