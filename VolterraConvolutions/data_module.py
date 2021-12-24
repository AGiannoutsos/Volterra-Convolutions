import matplotlib.pyplot as plt
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self,data_config):

        super().__init__()
        self.data_config = data_config
        
        self.mean = {
            CIFAR10: (0.49139923, 0.4821585, 0.44653007),
            CIFAR100: (0.5071, 0.4867, 0.4408),
        }
        self.std = {
            CIFAR10: (0.2023, 0.1994, 0.2010),
            CIFAR100: (0.2675, 0.2565, 0.2761),
        }
        self.basic_transforms = [
            T.ToTensor(),
            T.Normalize(self.data_config.mean, self.data_config.std),
        ]
        self.train_transforms_ = self.data_config.transforms + self.basic_transforms
        self.train_transforms_ = T.Compose(self.train_transforms_)
        self.test_transforms_ = self.basic_transforms
        self.test_transforms_ = T.Compose(self.test_transforms_)
        self.save_hyperparameters()

    def prepare_data(self):
        self.data_config.dataset_type(self.data_config.dataset_dir, train=True, download=True)
        self.data_config.dataset_type(self.data_config.dataset_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset = self.data_config.dataset_type(
                self.data_config.dataset_dir, train=True, transform=self.train_transforms_
            )
            self.dataset_len = len(self.dataset)
            self.train_dataset = self.dataset
            self.val_dataset = self.data_config.dataset_type(
                self.data_config.dataset_dir, train=False, transform=self.test_transforms_
            )

        if stage == "test" or stage is None:
            self.test_dataset = self.data_config.dataset_type(
                self.data_config.dataset_dir, train=False, transform=self.test_transforms_
            )

    def __len__(self):
        return len(self.train_dataset)

    def size(self):
        return len(self.train_dataset), len(self.val_dataset), len(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.data_config.batch_size, num_workers=self.data_config.num_workers, pin_memory=self.data_config.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.data_config.val_batch_size, num_workers=self.data_config.num_workers, pin_memory=self.data_config.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.data_config.val_batch_size, num_workers=self.data_config.num_workers, pin_memory=self.data_config.pin_memory)

    def get_images(self):
        images_count = 10
        channels_count, images_height, images_width = self.dataset[0][0].size()
        for index, (data, labels) in enumerate(self.train_dataloader()):
            plt.figure(figsize=(20, 20))
            grid_image_tensor = make_grid(data[0:images_count], images_count)
            print("Classes ", labels[0:images_count])
            plt.imshow(grid_image_tensor.permute((1, 2, 0)).cpu().numpy())
            break
