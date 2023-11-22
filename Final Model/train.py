from model import CNNModel
from EfficientNetB0 import EfficientNet
from Xception import XceptionNet
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import LogoDataModule
from lightning.pytorch import Trainer, seed_everything
import config

seed_everything(42, workers=True)

# model = CNNModel(num_classes=config.NUM_CLASSES,lr=config.LR)

# model = EfficientNet(num_classes=config.NUM_CLASSES,lr=config.LR)

model = XceptionNet(num_classes=config.NUM_CLASSES,lr=config.LR)


checkpoint_callback = ModelCheckpoint(
    dirpath = 'checkpoints',
    filename = config.CHECKPOINT_NAME,
    save_top_k = 1,
    verbose = True,
    monitor = 'mean_val',
    mode = 'min'
)

data_module = LogoDataModule(data_folder=config.DATA_FOLDER,
                            batch_size=config.BATCH_SIZE,
                            val_split=config.VAL_SPLIT)

data_module.setup()

trainer = pl.Trainer(devices=-1, 
                  accelerator="gpu",
                  check_val_every_n_epoch=5,
                  callbacks=[checkpoint_callback],
                  max_epochs=config.MAX_EPOCHS)


trainer.fit(model=model,datamodule=data_module)
