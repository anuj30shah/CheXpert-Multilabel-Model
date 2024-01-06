import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataset import ImageDataset
from torch.utils.data import DataLoader

# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#intialize the model
model = models.model(pretrained=True, requires_grad=True).to(device)
# load the model checkpoint
checkpoint = torch.load('outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
train_csv = pd.read_csv('input/CheXpert-v1.0-small/train.csv')
diagnoses = train_csv.columns.values[5:]
# prepare the test dataset and dataloader
test_data = ImageDataset(
    train=False, test=True
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    shuffle=False
)
for counter, data in enumerate(test_loader):
    image, target = data['image'].to(device), data['label']
    # get all the index positions where value == 1
    target_indices = [i for i in range(len(target[0])) if target[0][i] == 1]
    # get the predictions by passing the image through the model
    outputs = model(image)
    #print(outputs)
    outputs = torch.sigmoid(outputs)
    #print(outputs)
    outputs = outputs.detach().cpu()
    #print(outputs[0][0].tolist())
    sorted_indices = np.argsort(outputs[0])
    #print(sorted_indices)
    string_predicted = ''
    string_actual = ''
    for i in range(len(sorted_indices)):
        if (outputs[0][i].tolist() > 0.4):
            string_predicted += f"{diagnoses[i]}    "
    for i in range(len(target_indices)):
        string_actual += f"{diagnoses[target_indices[i]]}    "
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"PREDICTED: {string_predicted}\nACTUAL: {string_actual}")
    plt.savefig(f"outputs/inference_{counter}.jpg")
    plt.show()