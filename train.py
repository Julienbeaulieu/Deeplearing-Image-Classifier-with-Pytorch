# Imports
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import  Image
import numpy as np
import json
import pandas as pd
import argparse

# Import for keeping our session alive
from workspace_utils import active_session
import utils

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--layer',  action='append',
                    dest = 'layer',
                    help='Add hidden layers to the Network. Input multiple times for extra layers', type=int, default=[500, 200])

parser.add_argument('--lr',  action='store',
                    dest='learning_rate',
                    help='Select a learning rate', type=float, default = 0.001)

parser.add_argument('--arch',  action='store',
                    dest='arch',
                    help='Densenet121 can be used if specified, otherwise vgg16 will be used', default = 'vgg16')

parser.add_argument('--device',  action='store',
                    dest='device',
                    help='Choose between "gpu" and "cpu"', default='gpu')

parser.add_argument('--epochs',  action='store',
                    dest='epochs',
                    help='Number of epochs to train on', type = int, default = 7)

results = parser.parse_args()
hidden_layers = results.layer
arch = results.arch
learning_rate = results.learning_rate
device = results.device
epochs = results.epochs

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Define transforms for the training, validation, and testing sets

# Training data transforms with data augmentation
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=0.3),
                                       transforms.RandomRotation(50),
                                       transforms.RandomResizedCrop(224),
                                       transforms.ColorJitter(brightness = 2, contrast = 2),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean, std)
                                    ])
# Validation transforms
validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean,std)
                                           ])

# Testing transforms
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,std)
                                     ])

# Load the datasets with ImageFolder
training_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform = validation_transforms)
testing_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

# Define dataloaders using the image datasets and the trainforms
trainloader = torch.utils.data.DataLoader(training_dataset, batch_size = 64, shuffle=True) 
validloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 64, shuffle=True)
testloader = torch.utils.data.DataLoader(testing_dataset, batch_size = 64, shuffle=True)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Create model
model, optimizer, criterion, device = utils.setup(hidden_layers, learning_rate, arch, device)

# training the model
def train(model, trainloader, validloader, criterion, optimizer, epochs):
      
    epochs = epochs
    print_every = 10
    running_loss = 0
    steps = 0
    
    with active_session():
        for e in range(epochs):
            for images, labels in trainloader:
                steps += 1
                # Move input and label tensors to the device
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad() 

                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    validation_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for images, labels in validloader:
                            images, labels = images.to(device), labels.to(device)
                            log_probs = model(images)
                            batch_loss = criterion(log_probs, labels)

                            validation_loss += batch_loss.item()

                            # Calculate the accuracy
                            ps = torch.exp(log_probs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(F'Epoch number: {e+1}/{epochs}, '        
                          F'Train loss: {running_loss/print_every:.3f}, '
                          F'Validation loss: {validation_loss/len(validloader):.3f}, '
                          F'The accuracy = {accuracy/len(validloader):.3f}')
                    running_loss = 0
                    model.train()
                   
    return model

model = train(model, trainloader, validloader, criterion, optimizer, epochs) 

# save model
def save_model(path='checkpoint.pth', structure=arch):
    model.class_to_idx = training_dataset.class_to_idx

    checkpoint = {'structure': arch,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx
                 }

    torch.save(checkpoint, path)

save_model(path='checkpoint.pth', structure=arch)
# Save the checkpoint 

