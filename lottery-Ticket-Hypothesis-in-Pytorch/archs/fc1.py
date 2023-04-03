import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=10, setup:str ="mnist_small"):
        super(fc1, self).__init__()
        if setup == "mnist_large":
            self.classifier = nn.Sequential(
                nn.Linear(28*28, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, num_classes),
            )
        elif setup == "mnist_small":
            self.classifier = nn.Sequential(
                nn.Linear(28*28, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, num_classes),
            )
        elif setup == "cifar_small":
            self.classifier = nn.Sequential(
                nn.Linear(3*32*32, 300),
                nn.ReLU(inplace=True),
                nn.Linear(300, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, num_classes),
            )
        elif setup == "cifar_large":
            self.classifier = nn.Sequential(
                nn.Linear(3*32*32, 4000),
                nn.ReLU(inplace=True),
                nn.Linear(4000, 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, 4000),
                nn.ReLU(inplace=True),
                nn.Linear(4000, num_classes),
            )
        else:
            print("invalid setup choice for architecture")
            print(f"setup choice: {setup}")
            raise

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    