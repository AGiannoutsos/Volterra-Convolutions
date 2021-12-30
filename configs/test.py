import os

import argparse
from benedict import benedict 
from typing import Union


class AttrDict(benedict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__.update(self)
    def change_nested_value(self, key, value):
        keypath = [path for path in self.keypaths() if key in path]
        self[keypath[0]] = value
        

class ConfigArgparse(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ConfigArgparse, self).__init__(*args, **kwargs)
  
        self.add_argument('--depth',               type=int)    
        self.add_argument('--widen_factor',        type=int)    
        self.add_argument('--nesterov',            type=str)
        self.add_argument('--padding',             type=str)    

    
    def str_to_bool(self, value):
        if isinstance(value, bool):
            return value
        if value.lower() in {'False','false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    def get_config(self, config):

        args = self.parse_args()
        print(args)


        if args.depth:
            config.change_nested_value("depth", args.depth)
        if args.widen_factor:
            config.change_nested_value("widen_factor", args.widen_factor)
        if args.nesterov:
            config.change_nested_value("nesterov", self.str_to_bool(args.nesterov))
        if args.padding:
            config.change_nested_value("padding", args.padding)

        # next time I hope I fix this...
        CONFIG                 = AttrDict(config)
        CONFIG.LOGGER_CONFIG   = AttrDict(CONFIG.LOGGER_CONFIG)
        CONFIG.TRAINING_CONFIG = AttrDict(CONFIG.TRAINING_CONFIG)
        CONFIG.DATA_CONFIG     = AttrDict(CONFIG.DATA_CONFIG)
        CONFIG.MODEL_CONFIG    = AttrDict(CONFIG.MODEL_CONFIG)
        CONFIG.MODEL_CONFIG.optimizer_params = AttrDict(CONFIG.MODEL_CONFIG.optimizer_params)
        CONFIG.MODEL_CONFIG.lr_scheduler_params = AttrDict(CONFIG.MODEL_CONFIG.lr_scheduler_params)

        return CONFIG

########### WANDB #############################
LOGGER_CONFIG         = AttrDict({
                            "logger":              0,
                            "project_owner":       "andreas_giannoutsos",
                            "project_name":        "Volterra-Convolutions",
                            "experiment_name":     "linear_0",
                            "notes":               "baseline wide resnet 160 channels cifar10 linear convolution",
                            "tags":                ["cifar10", "linear", "time_0"],
                            "model_name":          "experiment_name",
                            "from_cloud":          True,
                            "load_saved_model":    True,
                            "saved_model_version": "latest",
                            "watch_model":         False,
                            "artifact_cleaner":    True,
                            "log_interval":        1,
                            "log_activations":     False,
                            "delete_artifacts":    False,
                        })

########### TRAINING PARAMS ######################
TRAINING_CONFIG       = AttrDict({
                            "min_epochs":          4,
                            "max_epochs":          220,
                            "val_check_interval":  1.0,
                            "log_every_n_steps":   50,
                            "callbacks":           [0],
                            "gpus":                1,
                            "benchmark":           True,
                            "training":            True,
                            "debug":               False,
                            "overfit_batches":     1
                        })
########### DATA PARAMS ##########################
DATA_CONFIG           = AttrDict({
                            "dataset_type":        0,
                            "batch_size":          128,
                            "val_batch_size":      200,
                            "dataset_dir":         "./",
                            "num_workers":         os.cpu_count(),
                            "pin_memory":          True,
                            "mean":                [0.4914, 0.4822, 0.4465],
                            "std":                 [0.2673, 0.2564, 0.2762],
                            "transforms":          [
                                                       
                                                   ],
                        })
########### MODEL PARAMS ##########################
MODEL_CONFIG          = AttrDict({
                            "depth":                28,
                            "widen_factor":         10,
                            "dropout_rate":         0.3,
                            "num_classes":          10,
                            "init_bn":              True,
                            "init_channels":        160,
                            "kernel_size":          3,
                            "stride":               1,
                            "padding":              1,
                            "dilation":             1,
                            "conv_type":            "linear_conv",
                            "masking":              True,
                            "scaling":              True,
                            "optimizer":            0,
                            "optimizer_params":     AttrDict({
                                "lr":           0.1,
                                "momentum":     0.9,
                                "weight_decay": 0.0005,
                                "nesterov":     True,
                            }),
                            "lr_scheduler":         0,
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




def trainer(LOGGER_CONFIG, TRAINING_CONFIG, DATA_CONFIG, MODEL_CONFIG):
    print(MODEL_CONFIG, MODEL_CONFIG["depth"], MODEL_CONFIG.depth, MODEL_CONFIG.optimizer_params.nesterov)


if __name__ == "__main__":
    CONFIG = ConfigArgparse().get_config(CONFIG)
    # print(CONFIG.__dict__)
    # print(CONFIG.MODEL_CONFIG.__dict__)

    

    # print(CONFIG.MODEL_CONFIG.depth)
    # print(CONFIG.MODEL_CONFIG.optimizer_params.nesterov)

    # print( AttrDict(CONFIG.MODEL_CONFIG).depth)
    # print( CONFIG.MODEL_CONFIG.optimizer_params.__dict__)
    # print( AttrDict(AttrDict(CONFIG.MODEL_CONFIG).optimizer_params).nesterov)
    print(CONFIG.MODEL_CONFIG, CONFIG.MODEL_CONFIG["depth"], CONFIG.MODEL_CONFIG.padding)

    # trainer(CONFIG.LOGGER_CONFIG, CONFIG.TRAINING_CONFIG, CONFIG.DATA_CONFIG, CONFIG.MODEL_CONFIG)