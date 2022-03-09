import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os.path
from tqdm import tqdm
import click
import pandas as pd

# Set current dir
curr_path = os.getcwd()

def loadModel(path):
    return torch.load(path)

@click.command()
@click.option("--model", default='resnet', help="Set choice of model (resnet, googlenet, alexnet)")
def main(model):
    model_name = model
    model = loadModel(f'{curr_path}\\models\\trained_{model_name}_train_epochs_25.pth')
    #print(model)
    model = models.alexnet(pretrained=True)
    print(model)

if __name__ == "__main__":
    main()