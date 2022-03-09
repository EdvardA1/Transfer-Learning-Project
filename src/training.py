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

'''
ATT GÖRA:
    * Skriva showImage funktionen
    * Testa resNet
    *   
    
'''

# Set current dir
curr_path = os.getcwd()

# Set plotting style
plt.style.use('ggplot')

# Use GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
#print("Using {} device".format(device))

def showImage(image):
    plt.imshow(image[0], cmap="gray")
    #plt.savefig(f'{curr_path}/sample_image.png', dpi=600)
    plt.show()
    
def imshow(inp, out_name, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.savefig(f'{curr_path}\\images\\{out_name}.png')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.clf()

def saveModel(model, out_name):
    if not os.path.isdir(f'{curr_path}\\models\\'):
        os.mkdir(f'{curr_path}\\models\\')    
    torch.save(model, (f'{curr_path}\\models\\{out_name}.pth'))

def saveValues(train_acc, train_loss, val_acc, val_loss, out_name):
    if not os.path.isdir(f'{curr_path}\\tables\\'):
        os.mkdir(f'{curr_path}\\tables\\')
    df = pd.DataFrame({'Train acc': train_acc, 'Train loss': train_loss, 'Val acc': val_acc, 'Val loss': val_loss})
    df.to_csv(f'{curr_path}\tables\{out_name}.csv')

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

def fine_tune_plot(x,y, model_name):
    if not os.path.isdir(f'{curr_path}\\plots\\'):
        os.mkdir(f'{curr_path}\\plots\\')
    plt.clf()
    plt.plot(x,y, label="Accuracy")
    plt.legend(loc='best')
    plt.title("Test Accuracy Over Unfrozen Layers")
    plt.xlabel("Unfrozen Layers")   
    plt.ylabel("Test Accuracy")
    plt.savefig(f'{curr_path}\\plots\\test_acc_fine_tuned_{model_name}.png', dpi=600)
    plt.clf() 

def visualizeModel(dataloader, model, out_name, num_images=6):
    if not os.path.isdir(f'{curr_path}\\images\\'):
        os.mkdir(f'{curr_path}\\images\\')
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    class_names = ["NORMAL","PNEUMONIA"]

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j], out_name)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


@click.command()
@click.option("--model", default='resnet', help="Set choice of model (resnet, googlenet, alexnet)")
@click.option("--n_epochs", default=2, help="Set number of epochs")
@click.option("--batch", default=100, help="Set batch size")
@click.option("--lr", default=0.1, help="Set desired learning rate")
@click.option("--un_freeze", default=0, help="Set number of layers to un-freeze on pre-trained model (last X layers excluding output layer)")
def main(model, n_epochs, batch, lr, un_freeze):
    
    print("-------------------------")
    print("        PARAMETERS    ")
    print("-------------------------")
    print(f"Device: {device}\nCurrent model: {model}\nEpochs: {n_epochs}\nBatch size: {batch}\nLearning rate: {lr}")
    print("-------------------------\n")

    model_name = model
    if model == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif model == 'resnet':
        model = models.resnet18(pretrained=True)
    else:
        print("Could not load model, please try again!")

    # Set data paths
    # Cats and Dogs
    #train_dir = f'{curr_path}\\datasets\\CatsAndDogs\\train'
    #val_dir = f'{curr_path}\\datasets\\CatsAndDogs\\val'
    #test_dir = f'{curr_path}\\datasets\\test_set\\'

    # ImageNet
    #train_dir = f'{curr_path}\\datasets\\ImageNet\\train\\'
    #val_dir = f'{curr_path}\\datasets\\ImageNet\\val\\'
    #test_dir = f'{curr_path}\\datasets\\ImageNet\\test\\'

    # Chest X-rays
    train_dir = f'{curr_path}\\datasets\\chest_xray\\train\\'
    val_dir = f'{curr_path}\\datasets\\chest_xray\\val\\'
    test_dir = f'{curr_path}\\datasets\\chest_xray\\test\\'

    # Hyperparameters
    batch_size = batch
    epochs = n_epochs
    eta = lr

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
    
    # So gradients are freezed for other layers
    for param in model.parameters():
        param.requires_grad = False

    if model_name == 'alexnet':
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 2)
        model = model.to(device)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=eta, momentum=0.9)

    # Scheduler 
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)

    # Loss
    loss_fn = nn.CrossEntropyLoss()
    #loss_fn = nn.BCELoss()
    
    # Just train model
    print("Training model...")
    plots = True
    model, train_acc, train_loss, val_acc, val_loss = training_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, device, plots, scheduler)

    # Save trained model
    saveModel(model, f"trained_{model_name}_train_epochs_{epochs}")
    
    tuned_acc = []

    print("Evaluating trained model...")
    acc, _ = evaluate(test_dataloader, model, loss_fn, device)
    tuned_acc.append(acc)

    print("Plotting accuracy...")
    plot(train_acc, val_acc, "Epochs", "Accuracy", f"Training vs. Validation Accruacy", f"acc_{model_name}_epochs_{epochs}", range(1, epochs+1))

    print("Plotting loss...")
    plot(train_loss, val_loss, "Epochs", "Loss", f"Training vs. Validation Loss", f"loss_{model_name}_epochs_{epochs}", range(1, epochs+1))
    
if __name__ == "__main__":
    main()