import os
import wandb
import torch
import numpy as np
import torchmetrics
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from VolterraConvolutions.wide_resnet import Wide_ResNet


class ModelModule(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.best_val_accuracy = []
        self.save_hyperparameters()

        self.model = Wide_ResNet(self.model_config)
        self.model.apply(self.init_weights)

        self.loss = nn.CrossEntropyLoss() 

        # metrics 
        self.Accuracy = torchmetrics.Accuracy(num_classes=self.model_config.num_classes)
        self.F1       = torchmetrics.F1(num_classes=self.model_config.num_classes)
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight, gain=np.sqrt(2))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight, 1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    
    def loss_function (self, logits, target):
        return self.loss(logits, target)

    def metrics(self, predictions, target):
        accuracy = self.Accuracy(predictions, target)
        f1 = self.F1(predictions, target)
        return accuracy, f1

    def forward(self, x):
        return self.model(x)

    def training_step(self, training_batch, training_batch_idx):
        data, labels = training_batch
        logits = self(data)
        loss = self.loss_function(logits, labels)
        accuracy, f1 = self.metrics(logits, labels)
        self.log("train_loss",     loss,     on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1",       f1,       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, training_batch, training_batch_idx):
        data, labels = training_batch
        logits = self(data)
        loss = self.loss_function(logits, labels)
        accuracy, f1 = self.metrics(logits, labels)
        self.log("val_loss",     loss,     on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1",       f1,       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return accuracy

    def test_step(self, training_batch, training_batch_idx):
        data, labels = training_batch
        logits = self(data)
        loss = self.loss_function(logits, labels)
        accuracy, f1 = self.metrics(logits, labels)
        self.log("test_loss",     loss,     on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1",       f1,       on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return accuracy

    def configure_optimizers(self):
        optimizer_params = self.model_config.optimizer_params.copy()
        optimizer_params.update({"params":self.parameters()})
        optimizer = self.model_config.optimizer(**optimizer_params)

        lr_scheduler_params = self.model_config.lr_scheduler_params.copy()
        lr_scheduler_params.update({"optimizer":optimizer})
        lr_scheduler = self.model_config.lr_scheduler(**lr_scheduler_params)
        ### Fixing checkpont scheduler problem after shutdown of training https://github.com/PyTorchLightning/pytorch-lightning/issues/4655  
        for epoch in range(self.trainer.current_epoch):
            print("Lacking epoch", epoch)
            lr_scheduler.step()
        
        return {"optimizer" : optimizer, "lr_scheduler": lr_scheduler}

    def predict_step(self, batch):
        return self(batch).argmax(dim=1)

    def validation_epoch_end(self, outputs):
        # log best validation accuracy epoch
        val_accuracy = torch.mean(torch.tensor((outputs)))
        if self.best_val_accuracy:
            if self.best_val_accuracy[-1] < val_accuracy:
                self.best_val_accuracy.append(val_accuracy)
            else:
                self.best_val_accuracy.append(self.best_val_accuracy[-1])
        else:
            self.best_val_accuracy.append(val_accuracy)

        self.log("best_val_accuracy", self.best_val_accuracy[-1], prog_bar=True, logger=True)


    @classmethod
    def get_model_from_wandb(cls, wandb_entity, project_name, experiment_name, version="latest"):
        api = wandb.Api()
        artifact = api.artifact(f"{wandb_entity}/{project_name}/model-{experiment_name}:{version}")
        artifact.download()
        model_artifact_directory = os.path.join(artifact.download(),"model.ckpt")
        # load model
        return cls.load_from_checkpoint(model_artifact_directory)

    
    def print_parameter_histograms(self, save_dir, bins=50):
        for name, parameter in tqdm(self.named_parameters()):
            data = parameter.float().ravel().detach().cpu().numpy()
            plt.hist(data, density=True, bins=bins) 
            plt.ylabel('Probability Density')
            plt.xlabel('Value of Weights')
            # plt.title(f"{name}")
            filename_png = f"{save_dir}/{name}.png"
            filename_svg = f"{save_dir}/{name}.svg"
            os.makedirs(os.path.dirname(filename_png), exist_ok=True)
            os.makedirs(os.path.dirname(filename_svg), exist_ok=True)
            plt.savefig(filename_png)
            plt.savefig(filename_svg)
            plt.close()
        

    # printing results works only after trainer.fit() for validation data
    def print_results(self, data=None, labels=None):
        images_count = 10
        channels_count, images_height, images_width = self.trainer.datamodule.dataset[0][0].size()
        for index, (data, labels) in enumerate(self.trainer.datamodule.val_dataloader()):
            plt.figure(figsize = (20,20)) 
            grid_image_tensor = make_grid(data[0:images_count], images_count)           
            print("Classes     ", labels[0:images_count])
            plt.imshow(grid_image_tensor.permute((1,2,0)).cpu().numpy())
            print("Predictions ", self.predict_step(data)[0:images_count])
            break

    def activations(self, x):
        return self.model.activations(x)