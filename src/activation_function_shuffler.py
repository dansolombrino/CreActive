from VGGModule import VGGModule
from CIFAR10DataModule import CIFAR10DataModule
import pytorch_lightning as pl
from pytorch_lightning import Trainer 
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.loggers import WandbLogger
from rich import print
import torch

pl.seed_everything(42)

def get_model_stats(trainer, model, model_type, val_dl, test_dl, logger: WandbLogger):
  val_preds = trainer.predict(model, val_dl)
  test_preds = trainer.predict(model, test_dl)

  acc_dict = {}
  avg_acc_dict = {}

  for preds, stage in zip([val_preds, test_preds], ["val", "test"]):
      
    acc_dict[stage] = []
      
    for pred_idx, pred in enumerate(preds):
      acc_dict[stage].append(pred["acc"])

      logger.log_metrics(
        metrics={f"{model_type}/acc/{stage}": acc_dict[stage][-1]},
        step=pred_idx
      )

    avg_acc_dict[stage] = torch.tensor(acc_dict[stage]).mean()
    logger.log_metrics(
      metrics={f"{model_type}/acc/avg/{stage}": avg_acc_dict[stage]},
      step=0
    )

  logger.log_metrics(
    metrics={
      f"{model_type}/acc/avg/val_and_test": torch.tensor(list(avg_acc_dict.values())).mean()
    },
    step=0
  )

def __main__():

  BATCH_SIZE = 256

  LR = 0.05

  TRAIN_ID_TO_LOAD = "2023_04_14_14_35_46"
  CKPT_PATH = f"../ckpts/{TRAIN_ID_TO_LOAD}/loss_val_epoch=59.ckpt"

  model = VGGModule(batch_size=BATCH_SIZE, lr=LR).load_from_checkpoint(CKPT_PATH)
  og_model_schema = dict(model.__dict__["_modules"])

  cifar10_datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE)
  cifar10_datamodule.prepare_data()
  cifar10_datamodule.setup()

  val_dl = cifar10_datamodule.val_dataloader()
  test_dl = cifar10_datamodule.test_dataloader()

  logger = WandbLogger(
    entity="dansolombrino", project="CreActive_activ_func_shuffle", 
    save_dir="../wandb"
  )

  trainer = Trainer(
    accelerator="gpu", devices=1, logger=logger,
    callbacks=[
      RichProgressBar(
        theme=RichProgressBarTheme(
          progress_bar="#C71585",
          progress_bar_finished="bold #008000",
          batch_progress="bold #008080",
          time="bold #1E90FF",
          processing_speed="bold #8B0000"
        )
      ),
    ],
    enable_checkpointing=True, default_root_dir="../ckpts"
  )

  get_model_stats(
    trainer=trainer, model=model, model_type="original", 
    val_dl=val_dl, test_dl=test_dl, logger=logger
  )

  model.conv_0_act = torch.nn.Tanh()
  model.conv_1_act = torch.nn.Tanh()
  model.fc_0_act = torch.nn.Tanh()
  shuffled_model_schema = dict(model.__dict__["_modules"])

  get_model_stats(
    trainer=trainer, model=model, model_type="shuffled", 
    val_dl=val_dl, test_dl=test_dl, logger=logger
  )

  logger.experiment.config.update({
    "og_model_schema": og_model_schema,
    "shuffled_model_schema": shuffled_model_schema
  })




if __name__ == "__main__":
  __main__()