import torchvision.models as models
import torch 
import torchvision
from torchvision import datasets
from torchvision import transforms
import os 
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
from torch.utils.data import Dataset, DataLoader


transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

data_dir = '/content/gdrive/MyDrive/cartoondataset' 
                                  
trainset = datasets.ImageFolder(os.path.join(data_dir , 'training'), transforms)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)     

testset = datasets.ImageFolder(os.path.join(data_dir, 'Validation'), transforms)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=True)  

def imshow(inp, title=None):
  
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  
inputs, classes = next(iter(trainloader))
out = torchvision.utils.make_grid(inputs)
imshow(out, title=[classes[x] for x in classes])

