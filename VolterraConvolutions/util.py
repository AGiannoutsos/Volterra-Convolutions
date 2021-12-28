import os
import wandb
import torch
import argparse
import pytorch_lightning as pl
from benedict import benedict 

os.environ["WANDB_RESUME"]  = "allow"


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

        self.add_argument('--init_channels', type=int)    
        self.add_argument('--kernels',       type=int)    
        self.add_argument('--dilation',      type=int)    
        self.add_argument('--stride',        type=int)    
        self.add_argument('--init_bn',       type=str)
        self.add_argument('--padding',       type=int)    
        self.add_argument('--masking',       type=str)  
        self.add_argument('--scaling',       type=str)  
        self.add_argument('--conv_type',     type=str)    
        self.add_argument('--depth',         type=int)    
        self.add_argument('--widen_factor',  type=int)    
        self.add_argument('--nesterov',      type=str)
    
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

        if args.init_channels:
            config.change_nested_value("init_channels", args.init_channels)
        if args.kernels:
            config.change_nested_value("kernels", args.kernels)
        if args.dilation:
            config.change_nested_value("dilation", args.dilation)
        if args.stride:
            config.change_nested_value("stride", args.stride)
        if args.init_bn:
            config.change_nested_value("init_bn", self.str_to_bool(args.init_bn))
        if args.padding:
            config.change_nested_value("padding", args.padding)
        if args.masking:
            config.change_nested_value("masking", self.str_to_bool(args.masking))
        if args.scaling:
            config.change_nested_value("scaling", self.str_to_bool(args.scaling))
        if args.conv_type:
            config.change_nested_value("conv_type", args.conv_type)
        if args.depth:
            config.change_nested_value("depth", args.depth)
        if args.widen_factor:
            config.change_nested_value("widen_factor", args.widen_factor)
        if args.nesterov:
            config.change_nested_value("nesterov", self.str_to_bool(args.nesterov))

        return config


class LRChanger(pl.Callback):
    def __init__(self, lr=1e-3, verbose=False):
        self.lr = lr
        self.verbose = verbose

    def on_train_epoch_start(self, trainer, pl_module):
        if self.verbose:
            print("Change Learning Rate to ", self.lr)
        for param in trainer.optimizers[0].param_groups:
            param['lr'] = self.lr

class WDChanger(pl.Callback):
    def __init__(self, wd=0.0, epoch=200, verbose=True):
        self.wd = wd
        self.verbose = verbose
        self.epoch = epoch
        
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch >= self.epoch:
            if self.verbose:
                print("Change Weight Decay to ", self.wd, trainer.current_epoch)
            for param in trainer.optimizers[0].param_groups:
                param['weight_decay'] = self.wd
        
            
def artifact_cleaner(project_name, run_name, verbose=True):
    os.system("wandb artifact cache cleanup 0.5GB >> ./wandb/cleanup.log")
    try:
        api_run = wandb.Api().run(project_name+"/"+run_name)
        for artifact in api_run.logged_artifacts():
            if len(artifact.aliases) == 0:
                if verbose:
                    print("DELETING ARTIFACT ", artifact)
                artifact.delete()
    except Exception as e: 
        print(e)

class ArtifactCleaner(pl.Callback):
    def __init__(self, project_name, run_name, verbose=True):
        self.project_name = project_name
        self.run_name     = run_name
        self.verbose      = verbose
    def on_train_epoch_end(self, trainer, pl_module):
        artifact_cleaner(self.project_name, self.run_name, self.verbose)

def fix_scheduler_last_epoch(path):
    # fix last epoch and save again model
    model = torch.load(path)
    model["lr_schedulers"][0]["_step_count"] = model["epoch"]
    model["lr_schedulers"][0]["last_epoch"] = model["epoch"]
    torch.save(model, path)

def get_model_from_logger(run, wandb_entity, project_name, experiment_name, version, type="model"):
    print(f"LOAD SAVED MODEL {experiment_name}:{version}")
    try:
        if run: # get model from wandb
            api = wandb.Api()
            artifact = api.artifact(f"{wandb_entity}/{project_name}/model-{experiment_name}:{version}")
            model_directory = os.path.join(artifact.download(),"model.ckpt")
        else:
            model_directory = os.path.join(experiment_name,version+".ckpt")
    except:
        print("NO MODEL FOUND\nINITIALIZING NEW MODEL")
        return None
    #### FIX EPOCH BUG
    fix_scheduler_last_epoch(model_directory)
    return model_directory