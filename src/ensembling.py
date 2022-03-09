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

'''
ATT GÖRA:
    * Skriva showImage funktionen
    * Testa resNet
    *   
    
'''

# Set current dir
curr_path = os.getcwd()

# Use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

def majorityPrediction(predA, predR, predG):
    #print(predA, predG, predR)
    if torch.eq(predA, predR):
        return predA
    elif torch.eq(predA, predG):
        return predA
    else:
        return predR

def evaluate(dataloader, models, loss_fn, device):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, ncols=100, ascii=True, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            predAlexnet = models['alexnet'](X).argmax(1)
            predResnet = models['resnet'](X).argmax(1)
            predGooglenet = models['googlenet'](X).argmax(1)
            pred = majorityPrediction(predAlexnet, predResnet, predGooglenet)            
            #test_loss += loss_fn(pred, y).item()
            correct += (pred == y).type(torch.float).sum().item()
    #test_loss /= size
    correct /= size
    print(f"Accuracy: {(100*correct):>0.2f}%")
    return (100*correct) #, test_loss

def loadModel(path):
    return torch.load(path)

def main():
    
    print("-------------------------")
    print("        ENSEMBLING   ")
    print("-------------------------")

    # Load the models
    # alexnet = loadModel(f'{curr_path}\\models\\trained_alexnet_train_epochs_25.pth')
    # resnet = loadModel(f'{curr_path}\\models\\trained_resnet_train_epochs_25.pth')
    # googlenet = loadModel(f'{curr_path}\\models\\trained_googlenet_train_epochs_25.pth')

    alexnet = loadModel(f'{curr_path}\\models\\fine_tuned_model_alexnet_train_epochs_25_tuned_epochs_10.pth')
    resnet = loadModel(f'{curr_path}\\models\\fine_tuned_model_resnet_train_epochs_25_tuned_epochs_10.pth')
    googlenet = loadModel(f'{curr_path}\\models\\fine_tuned_model_googlenet_train_epochs_25_tuned_epochs_10.pth')

    alexnet.eval()
    resnet.eval()
    googlenet.eval()

    models = {'alexnet': alexnet, 'resnet': resnet, 'googlenet': googlenet}

    # Chest X-rays
    test_dir = f'{curr_path}\\datasets\\chest_xray\\test\\'

    # Pre-processing
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get training data
    test_data = datasets.ImageFolder(test_dir, transform=preprocess)
    
    # Data loaders
    test_dataloader = DataLoader(test_data)
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Ensemble 
    evaluate(test_dataloader, models, loss_fn, device)
    
if __name__ == "__main__":
    main()