import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from engine import train, validate
from dataset import ImageDataset
from torch.utils.data import DataLoader
import models

matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intialize the model
model = models.model(pretrained=True, requires_grad=True).to(device)
# learning parameters
lr = 0.0001
epochs = 10
batch_size = 32
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

# train dataset
train_data = ImageDataset(
    train=True, test=False
)
# validation dataset
valid_data = ImageDataset(
    train=False, test=False
)
# train data loader
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False
)
# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size,
    shuffle=False
)

# start the training and validation
train_loss = []
valid_loss = []
train_acc, valid_acc = [], []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, train_loader, optimizer, criterion, train_data, device
    )
    valid_epoch_loss = validate(
        model, valid_loader, criterion, valid_data, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')



# save the trained model to disk
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, 'outputs/model.pth')
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss.png')
plt.show()

plt.figure(figsize=(10, 7))
plt.plot(train_acc, color='orange', label='train accuracy')
plt.plot(valid_acc, color='red', label='validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/accuracy.png')
plt.show()