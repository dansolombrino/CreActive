import pytorch_lightning as pl
import torch
from torch.nn import Conv2d, MaxPool2d, Linear, functional as F, ReLU
from torchmetrics.functional import accuracy
from torch.optim.lr_scheduler import OneCycleLR

pl.seed_everything(42)

class VGGModule(pl.LightningModule):

  def __init__(self, batch_size, lr):

    super(VGGModule, self).__init__()

    self.save_hyperparameters()

    self.conv_0 = Conv2d(in_channels=3, out_channels=32, kernel_size=3)

    self.conv_0_act = ReLU()

    self.conv_1 = Conv2d(in_channels=32, out_channels=32, kernel_size=3)

    self.conv_1_act = ReLU()

    self.pool_1 = MaxPool2d(kernel_size=2)

    self.fc_0 = Linear(in_features=6272, out_features=128)

    self.fc_0_act = ReLU()

    self.fc_1 = Linear(in_features=128, out_features=10) 

    self.fc_1_act = torch.nn.Softmax(dim=1)

  
  def forward(self, x):

    x = self.conv_0(x)

    x = self.conv_0_act(x)

    x = self.conv_1(x)

    x = self.conv_1_act(x)

    x = self.pool_1(x)

    x = torch.flatten(x, start_dim=1, end_dim=-1)

    x = self.fc_0(x)

    x = self.fc_0_act(x)

    x = self.fc_1(x)

    x = self.fc_1_act(x)

    return x
  
  def _common_step(self, batch, batch_idx, step_name):
    x, y = batch
    
    logits = self.forward(x)

    y_one_hot = F.one_hot(y, 10).float()
    loss = F.cross_entropy(input=logits, target=y_one_hot)

    preds = torch.argmax(logits, dim=1)
    acc = accuracy(preds=preds, target=y, task="multiclass", num_classes=10)

    if step_name != "predict":
      self.log(f"loss/{step_name}", loss)
      self.log(f"loss_{step_name}", loss)
      self.log(f"acc/{step_name}", acc)
      self.log(f"acc_{step_name}", acc)

    return {
      "loss": loss, 
      "acc": acc
    }
  

  def training_step(self, batch, batch_idx):
    return self._common_step(batch=batch, batch_idx=batch_idx, step_name="train")
  

  def validation_step(self, batch, batch_idx):
    return self._common_step(batch=batch, batch_idx=batch_idx, step_name="val")
  

  def test_step(self, batch, batch_idx):
    return self._common_step(batch=batch, batch_idx=batch_idx, step_name="test")
  

  def predict_step(self, batch, batch_idx):
    return self._common_step(batch=batch, batch_idx=batch_idx, step_name="predict")
  

  def configure_optimizers(self):
    
    optimizer = torch.optim.SGD(
        self.parameters(),
        lr=self.hparams.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    
    steps_per_epoch = 40192 // self.hparams.batch_size
    
    scheduler_dict = {
        "scheduler": OneCycleLR(
            optimizer,
            0.1,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=steps_per_epoch,
        ),
        "interval": "step",
    }
    return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
  

def __main__():
  from CIFAR10DataModule import CIFAR10DataModule

  BATCH_SIZE = 4

  cifar10_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE)
  cifar10_datamodule.prepare_data()
  cifar10_datamodule.setup()

  train_dl = cifar10_datamodule.train_dataloader()

  batch = next(iter(train_dl))

  model = VGGModule()

  model.training_step(batch=batch, batch_idx=69)


if __name__ == "__main__":
  __main__()