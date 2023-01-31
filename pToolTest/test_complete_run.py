import logging
from pathlib import Path
from typing import List

import numpy as np
import toml
import torch
from torch.autograd import Variable
from torch import optim
import torch.nn as nn

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from constructor import DLPlotter


cwd = Path(__file__).resolve().parent
logging.basicConfig(level=logging.INFO,
                    filename=f'{cwd}/std.log',
                    format="[%(asctime)s] %(levelname)s [%(name)s.%(module)s.%(funcName)s:%(lineno)d] %(message)s",
                    filemode='w')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.layer2 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = x.view(x.size(0), -1)
        output = self.layer2(x)
        return output


def train(num_epochs: int,
          cnn: nn.Module,
          loaders: dict,
          loss_func: nn.CrossEntropyLoss,
          optimizer: optim,
          mode: str,
          config: dict,
          plotter: DLPlotter,
          exp_dir: Path,
          num: str):

    val_loss = 0
    val_steps = 0

    cnn.train()
    total_step = len(loaders[mode])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders[mode]):

            b_x = Variable(images)
            b_y = Variable(labels)
            output = cnn(b_x)
            loss = loss_func(output, b_y)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if mode == 'train':
                weights = dict()
                for l in range(3):
                    if l != 2:
                        name = cnn._modules[f'layer{l}'][0].__class__.__name__
                        weights[f'({l}){name}'] = cnn._modules[f'layer{l}'][0].weight.detach().clone()
                    else:
                        name = cnn._modules[f'layer{l}'].__class__.__name__
                        weights[f'({l}){name}'] = cnn._modules[f'layer{l}'].weight.detach().clone()

                plotter.collect_weights(f"exp{num}", len(loaders[mode]), epoch + 1, i, weights)

            val_steps += 1
            val_loss += loss.item()

            plotter.collect_loss(f"exp{num}", len(loaders[mode]), epoch + 1, i, loss.item(), mode)

        logger.info(f"{mode.upper()} Data for Epoch [{epoch} / {num_epochs-1}] collected.")

    if mode == 'train':
        output_file_name = exp_dir / f"exp{num}.toml"
        with open(output_file_name, "w") as toml_file:
            toml.dump(config, toml_file)

        plotter.collect_parameter(f"exp{num}", config, val_loss / val_steps)

        logger.info(f"Hyperparameters for Experiment exp{num} collected.")


def run(epochs: List[int], lr: List[float], batch_size: List[int], exp_names: List[str]):
    cwd = Path().absolute()
    exp_dir = cwd / 'exp'
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_data = datasets.MNIST(
        root='data',
        train=True,
        transform=ToTensor(),
        download=True)

    val_data = datasets.MNIST(
        root='data',
        train=False,
        transform=ToTensor()
    )

    train_data = torch.utils.data.Subset(train_data, np.arange(0, 2560, 1))
    val_data = torch.utils.data.Subset(val_data, np.arange(0, 1280, 1))

    plotter = DLPlotter()
    for e, l, b, n in zip(epochs, lr, batch_size, exp_names):

        config = {
            "epoch": e,
            "lr": l,
            "batch_size": b
        }

        loaders = {
            'train': torch.utils.data.DataLoader(train_data,
                                                 batch_size=b,
                                                 shuffle=True,
                                                 num_workers=1),

            'validation': torch.utils.data.DataLoader(val_data,
                                               batch_size=b,
                                               shuffle=True,
                                               num_workers=1),
        }

        cnn = CNN()
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=l)

        for mode in ['train', 'validation']:
            train(e, cnn, loaders,  loss_func, optimizer, mode, config, plotter, exp_dir, n)

    plotter.construct(port=8050)


if __name__ == "__main__":
    test_data = dict(epochs=[2, 1, 1],
                     lr=[0.002, 0.01, 0.0005],
                     batch_size=[32, 64, 128],
                     exp_names=['0001', '0002', '0003'])
    run(**test_data)