from datetime import datetime
from VGGModule import VGGModule
import pytorch_lightning as pl 
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from CIFAR10DataModule import CIFAR10DataModule

pl.seed_everything(42)

TRAIN_ID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
CKPT_DIR = f"../ckpts/{TRAIN_ID}"

BATCH_SIZE = 256

LR = 0.05
MAX_EPOCHS = 60

PRECISION = "32-true"

model = VGGModule(batch_size=BATCH_SIZE, lr=LR)

logger = WandbLogger(entity="dansolombrino", project="CreActive", save_dir="../wandb")

logger.experiment.config.update({
  "train_id": TRAIN_ID   
})

trainer = Trainer(
  max_epochs=MAX_EPOCHS,
  accelerator="gpu",
  devices=1,
  precision=PRECISION,
  logger=logger,
  callbacks=[
    LearningRateMonitor(logging_interval="step"), 
    RichProgressBar(
      theme=RichProgressBarTheme(
        progress_bar="#C71585",
        progress_bar_finished="bold #008000",
        batch_progress="bold #008080",
        time="bold #1E90FF",
        processing_speed="bold #8B0000"
      )
    ),
    ModelCheckpoint(
      dirpath=CKPT_DIR, monitor="loss_val", filename="loss_train_{epoch}"
    ),
    ModelCheckpoint(
      dirpath=CKPT_DIR, monitor="loss_train", filename="loss_val_{epoch}"
    )
  ],
  enable_checkpointing=True,
  default_root_dir="../ckpts"
)

cifar10_dm = CIFAR10DataModule(batch_size=BATCH_SIZE)

trainer.fit(model, cifar10_dm)
trainer.test(model, cifar10_dm)