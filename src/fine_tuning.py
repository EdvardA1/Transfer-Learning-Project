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
from DLFunctions import training_model, evaluate

# Set current dir
curr_path = os.getcwd()

# Use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"

def loadModel(path):
    return torch.load(path)

def saveModel(model, out_name):
    if not os.path.isdir(f'{curr_path}\\models\\'):
        os.mkdir(f'{curr_path}\\models\\')    
    torch.save(model, (f'{curr_path}\\models\\{out_name}.pth'))

def fine_tuning(t_dataloader, v_dataloader, model, loss_fn, epochs, plots, device):
    # Fine tuning
    # Unfreeze some layers

    # print(f"\nUnfreezing {unfreeze}")
    # for name, child in model.named_children():
    #     if name in unfreeze:
    #         #print(name + ' is unfrozen')
    #         for param in child.parameters():
    #           param.requires_grad = True
    #     else:
    #       #print(name + ' is frozen')
    #       for param in child.parameters():
    #           param.requires_grad = False

    # So gradients are freezed for other layers

    for param in model.parameters():
        param.requires_grad = True

    low_eta = 1e-5
    optimizer_ft = optim.SGD(model.parameters(), lr=low_eta, momentum=0.9)
    
    model, train_acc, train_loss, val_acc, val_loss = training_model(t_dataloader, v_dataloader, model, loss_fn, optimizer_ft, epochs, device, plots)
    return model, train_acc, train_loss, val_acc, val_loss

def plot(x, y, x_label, y_label, title, out_name, epochs):
    if not os.path.isdir(f'{curr_path}\\plots\\'):
        os.mkdir(f'{curr_path}\\plots\\')
    plt.plot(epochs, x, label='Training')
    plt.plot(epochs, y, label='Validation')
    plt.legend(loc='best')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(f'{curr_path}\\plots\\{out_name}.png', dpi=600)
    plt.clf()

@click.command()
@click.option("--model", default='resnet', help="Set choice of model (resnet, googlenet, alexnet)")
def main(model):

    model_name = model

    # Load model
    model = loadModel(f'{curr_path}\\models\\trained_{model_name}_train_epochs_25.pth')

    # Chest X-rays
    train_dir = f'{curr_path}\\datasets\\chest_xray\\train\\'
    val_dir = f'{curr_path}\\datasets\\chest_xray\\val\\'
    test_dir = f'{curr_path}\\datasets\\chest_xray\\test\\'

    # Hyperparameters
    batch_size = 32

    # Pre-processing
    preprocess_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Get training data
    training_data = datasets.ImageFolder(train_dir, transform=preprocess_train)
    validation_data = datasets.ImageFolder(val_dir, transform=preprocess)
    test_data = datasets.ImageFolder(test_dir, transform=preprocess)
    
    # Data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(validation_data, shuffle=True)
    test_dataloader = DataLoader(test_data, shuffle=True)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    print("Evaluating trained model...")

    acc, _ = evaluate(test_dataloader, model, loss_fn, device)
    print("Accuracy before fine tuning:", acc)

    # Fine tune model
    tuned_epochs = 10
    plots = True # Not printing train and val loss
    print("Fine tuning...")
    fine_tune_model, train_acc, train_loss, val_acc, val_loss = fine_tuning(train_dataloader, val_dataloader, model, loss_fn, tuned_epochs, plots, device)

    print("Evaluating fine-tuned model...")
    acc, _ = evaluate(test_dataloader, fine_tune_model, loss_fn, device)
    print("Accuracy after fine tuning:", acc)

    # Saving tuned accuracies
    # tuned_df = pd.DataFrame({'Accuracy': tuned_acc})
    # tuned_df.to_csv(f'{curr_path}\\tables\\tuned_acc_{model_name}.csv')

    # print("Visualize fine tuned model...")
    # visualizeModel(val_dataloader, fine_tune_model, f"images_{model_name}_epochs_{epochs}")

    # print("Plotting fine tuned accuracy...")
    # fine_tune_plot(unfreeze_layers, tuned_acc, model_name)

    # print("Saving fine tuned model...")
    saveModel(fine_tune_model,f"fine_tuned_model_{model_name}_train_epochs_25_tuned_epochs_{tuned_epochs}")

    print("Plotting accuracy...")
    plot(train_acc, val_acc, "Epochs", "Accuracy", f"Training vs. Validation Accruacy", f"acc_finetuned_{model_name}_epochs_10", range(1, 11))

    print("Plotting loss...")
    plot(train_loss, val_loss, "Epochs", "Loss", f"Training vs. Validation Loss", f"loss_finetuned_{model_name}_epochs_10", range(1, 11))

    # print("Saving values...")
    # saveValues(train_acc, train_loss, val_acc, val_loss, f"values_{model_name}_epochs_{epochs}")

if __name__ == "__main__":
    main()