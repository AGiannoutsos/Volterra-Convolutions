import os
import wandb
import torch
from VolterraConvolutions.trainer_module import trainer
from VolterraConvolutions.util import AttrDict, WDChanger
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms as T


########### WANDB #############################
LOGGER_CONFIG         = AttrDict({
                            "logger":              wandb,
                            "project_owner":       "andreas_giannoutsos",
                            "project_name":        "test_mnist_ptyxiakh",
                            "experiment_name":     "test_test",
                            "notes":               "test baseline wide resnet from github with 16 init channels old MEN STD",
                            "tags":                ["cifar10", "volterra2nd_SCALIGN FACTOR" "baseline", "wide_resnet" "n28", "k10"],
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
                            "dataset_type":        CIFAR10,
                            "batch_size":          128,
                            "val_batch_size":      200,
                            "dataset_dir":         "./",
                            "num_workers":         os.cpu_count(),
                            "pin_memory":          True,
                            "mean":                [0.4914, 0.4822, 0.4465],
                            "std":                 [0.2673, 0.2564, 0.2762],
                            "transforms":          [
                                                       T.RandomCrop(size=(32,32), padding=4, padding_mode="reflect"),
                                                       T.RandomHorizontalFlip(p=0.5),
                                                   ],
                        })
########### MODEL PARAMS ##########################
MODEL_CONFIG          = AttrDict({
                            "depth":                10,
                            "widen_factor":         1,
                            "dropout_rate":         0.3,
                            "num_classes":          10,
                            "stride":               1,
                            "padding":              1,
                            "dilation":             1,
                            "conv_type":            "linear_conv",
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


if __name__ == __main__:
    trainer(LOGGER_CONFIG, TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG)