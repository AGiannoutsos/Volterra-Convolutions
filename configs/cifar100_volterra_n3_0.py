import os
import wandb
import torch
from VolterraConvolutions.trainer_module import trainer
from VolterraConvolutions.util import AttrDict, WDChanger, ConfigArgparse
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T



########### WANDB #############################
LOGGER_CONFIG         = AttrDict({
                            "logger":              wandb,
                            "project_owner":       "andreas_giannoutsos",
                            "project_name":        "Volterra-Convolutions",
                            "experiment_name":     "cifar100_volterra_n3_0",
                            "notes":               "baseline wide resnet 160 channels cifar100 volterra n3 convolution",
                            "tags":                ["cifar100", "volterra_n3", "time_0"],
                            "model_name":          "experiment_name",
                            "from_cloud":          True,
                            "load_saved_model":    True,
                            "saved_model_version": "latest",
                            "watch_model":         False,
                            "artifact_cleaner":    True,
                            "log_interval":        1,
                            "delete_artifacts":    False,
                        })

########### TRAINING PARAMS ######################
TRAINING_CONFIG       = AttrDict({
                            "min_epochs":          4,
                            "max_epochs":          220,
                            "val_check_interval":  1.0,
                            "log_every_n_steps":   50,
                            "callbacks":           [LearningRateMonitor("epoch"), WDChanger(wd=0.0, epoch=200)],
                            "gpus":                1,
                            "benchmark":           True,
                            "training":            True,
                            "debug":               False,
                            "overfit_batches":     1
                        })
########### DATA PARAMS ##########################
DATA_CONFIG           = AttrDict({
                            "dataset_type":        CIFAR100,
                            "batch_size":          128,
                            "val_batch_size":      200,
                            "dataset_dir":         "./",
                            "num_workers":         os.cpu_count(),
                            "pin_memory":          True,
                            "mean":                [0.5071, 0.4867, 0.4408],
                            "std":                 [0.2675, 0.2565, 0.2761],                            
                            "transforms":          [
                                                       T.RandomCrop(size=(32,32), padding=4, padding_mode="reflect"),
                                                       T.RandomHorizontalFlip(p=0.5),
                                                   ],
                        })
########### MODEL PARAMS ##########################
MODEL_CONFIG          = AttrDict({
                            "depth":                28,
                            "widen_factor":         10,
                            "dropout_rate":         0.3,
                            "num_classes":          100,
                            "init_bn":              True,
                            "init_channels":        160,
                            "stride":               1,
                            "padding":              1,
                            "dilation":             1,
                            "conv_type":            "volterra_conv_n3",
                            "masking":              True,
                            "scaling":              True,
                            "optimizer":            torch.optim.SGD,
                            "optimizer_params":     AttrDict({
                                "lr":           0.1,
                                "momentum":     0.9,
                                "weight_decay": 0.0005,
                                "nesterov":     True,
                            }),
                            "lr_scheduler":         torch.optim.lr_scheduler.MultiStepLR,
                            "lr_scheduler_params":  AttrDict({
                                "milestones":   [60, 120, 160, 200],
                                "gamma":        0.2,
                                "verbose":      False,
                            }),
                        })
#################################################
CONFIG = AttrDict({"LOGGER_CONFIG": LOGGER_CONFIG,
                   "TRAINING_CONFIG": TRAINING_CONFIG,
                   "DATA_CONFIG": DATA_CONFIG,
                   "MODEL_CONFIG": MODEL_CONFIG})
#################################################


if __name__ == "__main__":
    CONFIG = ConfigArgparse().get_config(CONFIG)
    trainer(CONFIG.LOGGER_CONFIG, CONFIG.TRAINING_CONFIG, CONFIG.DATA_CONFIG, CONFIG.MODEL_CONFIG)