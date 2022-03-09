import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    loss_avg = 0
    model.train()
    for batch, (X, y) in enumerate(tqdm(dataloader, ncols=100, ascii=True, desc="Training  ")):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        loss_avg += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_avg / len(dataloader.dataset)


def training_model(t_dataloader, v_dataloader, model, loss_fn, optimizer, epochs, device, plots = True, scheduler=None):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for t in range(epochs):
        print("\n--------------")
        print(f"Epoch {t+1} of {epochs}")
        print("--------------")
        train(t_dataloader, model, loss_fn, optimizer, device)

        if plots:
            t_acc, t_loss = evaluate(DataLoader(t_dataloader.dataset, shuffle=True), model, loss_fn, device)
            v_acc, v_loss = evaluate(v_dataloader, model, loss_fn, device)
            train_acc.append(t_acc)
            train_loss.append(t_loss)
            val_acc.append(v_acc)
            val_loss.append(v_loss)

        if scheduler != None:
            scheduler.step()

    if plots:
        # Convert to numpy array
        train_acc = np.array(train_acc)
        train_loss = np.array(train_loss)
        val_acc = np.array(val_acc)
        val_loss = np.array(val_loss)

    return model, train_acc, train_loss, val_acc, val_loss 


def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, ncols=100, ascii=True, desc="Evaluating"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Accuracy: {(100*correct):>0.2f}%\nAvg loss: {test_loss:>8f} \n")
    return (100*correct), test_loss