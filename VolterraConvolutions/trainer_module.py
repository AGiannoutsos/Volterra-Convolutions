import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from VolterraConvolutions.util import artifact_cleaner, ArtifactCleaner, get_model_from_logger
from VolterraConvolutions.data_module import CIFARDataModule
from VolterraConvolutions.model_module import ModelModule


os.environ["WANDB_RESUME"]  = "allow"

def trainer(LOGGER_CONFIG, TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG):

    # setting same seed for ddp
    if TRAINING_CONFIG.gpus > 1:
        pl.seed_everything(42)

    DATA_MODULE = CIFARDataModule(data_config=DATA_CONFIG)
    MODEL_MODULE = ModelModule(model_config=MODEL_CONFIG)

    logger = None
    ckpt_path = None
    debug = TRAINING_CONFIG.debug
    del TRAINING_CONFIG.debug
    training = TRAINING_CONFIG.training
    del TRAINING_CONFIG.training

    if training and LOGGER_CONFIG.logger is not None and debug is False:
        print(f"INITIALIZING LOGGER WITH NAME {LOGGER_CONFIG.experiment_name} IN PROJECT {LOGGER_CONFIG.project_name}")
        logger = WandbLogger(log_model='all', project=LOGGER_CONFIG.project_name, resume=LOGGER_CONFIG.experiment_name, name=LOGGER_CONFIG.experiment_name)
        TRAINING_CONFIG["logger"] = logger
        # run = LOGGER_CONFIG.logger.init(project=LOGGER_CONFIG.project_name, resume=LOGGER_CONFIG.experiment_name) 
        # run.name = LOGGER_CONFIG.experiment_name
        if not LOGGER_CONFIG.load_saved_model:
            logger.experiment.tags  = LOGGER_CONFIG.tags
            logger.experiment.notes = LOGGER_CONFIG.notes
        if LOGGER_CONFIG.delete_artifacts:
            artifact_cleaner(LOGGER_CONFIG.project_name, LOGGER_CONFIG.experiment_name)
        if LOGGER_CONFIG.artifact_cleaner:
            TRAINING_CONFIG.callbacks.append(ArtifactCleaner(LOGGER_CONFIG.project_name, LOGGER_CONFIG.experiment_name, verbose=False))
        if LOGGER_CONFIG.watch_model:
            logger.watch(MODEL_MODULE, log="all", log_freq=TRAINING_CONFIG.log_every_n_steps)
        # if training:

    # loggers and callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=LOGGER_CONFIG.experiment_name, every_n_epochs=LOGGER_CONFIG.log_interval, monitor="val_accuracy", mode="max", auto_insert_metric_name=True, save_last=True)
    TRAINING_CONFIG.callbacks.append(checkpoint_callback)

    # fix trainer kwargs
    if debug:
        print("DEBUGGING")
        TRAINING_CONFIG.val_check_interval = 1
    else:
        del TRAINING_CONFIG.overfit_batches

    # check for gpus
    if not torch.cuda.is_available():
        print("GPU NOT AVAILABLE\nREMOVING GPUS")
        del TRAINING_CONFIG.gpus

    # load model from cloud
    if LOGGER_CONFIG.load_saved_model: 
        ckpt_path = get_model_from_logger(LOGGER_CONFIG.from_cloud, 
                                          LOGGER_CONFIG.project_owner,
                                          LOGGER_CONFIG.project_name, 
                                          LOGGER_CONFIG.experiment_name if LOGGER_CONFIG.model_name is "experiment_name" else LOGGER_CONFIG.model_name, 
                                          LOGGER_CONFIG.saved_model_version)
            
    # init trainer
    trainer = pl.Trainer(**TRAINING_CONFIG)

    if training:
        trainer.fit(MODEL_MODULE, datamodule=DATA_MODULE, ckpt_path=ckpt_path)
    else:
        trainer.test(MODEL_MODULE, datamodule=DATA_MODULE, ckpt_path=ckpt_path)

    if training and LOGGER_CONFIG.logger is not None and debug is False:
        LOGGER_CONFIG.logger.finish()

    return trainer
