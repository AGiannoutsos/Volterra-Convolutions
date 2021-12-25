import os
import wandb
import torch
import pytorch_lightning as pl

os.environ["WANDB_RESUME"]  = "allow"


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

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

            # artifact = run.use_artifact(("model-%s:%s"%(experiment_name, version)), type=type)
        else:
            model_directory = os.path.join(experiment_name,version+".ckpt")
    except:
        print("NO MODEL FOUND\nINITIALIZING NEW MODEL")
        return None
    #### FIX EPOCH BUG
    fix_scheduler_last_epoch(model_directory)
    return model_directory