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

# Import for keeping our session alive
from workspace_utils import active_session

# Label Mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Quick check of data in json file
df = pd.DataFrame({'flower_type': cat_to_name})
df.head(10)

# Define our classifier Class
class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, dropout = 0.2):
        
        super().__init__()     
        
        # self.input_size = nn.Linear(input_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
                                  
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)
                                  
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        for hidden_layer in self.hidden_layers:
            x = self.dropout(F.relu(hidden_layer(x)))
        x = F.log_softmax(self.output(x), dim=1)
        
        return x

def setup(hidden_layers, learning_rate, arch, device='gpu',  dropout = 0.3):
    
    if arch == 'vgg16':
        input_size = 25088
        # Use VGG16 model from Torchvision
        model = models.vgg16(pretrained=True)
        
    elif arch == 'densenet121':
        input_size = 1024
        # Use Densenet121 from Torchvision
        model = models.densenet121(pretrained=True)
    
    else:
        print("Sorry, this is not a valid model. Try 'vgg16' or 'densenet121'")
    
    # Freeze the parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # output_size = 102 flower categories
    output_size = 102
  
    classifier = Classifier(input_size, output_size, hidden_layers, dropout = 0.3)
    
    # Update the model's classifier    
    model.classifier = classifier
    
    # Select between gpu and cpu
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')

    # Bring model to device
    model.to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
    
    # Define loss function
    criterion = nn.NLLLoss()
    
    return model, optimizer, criterion, device

# testing the model and returning the accuracy on new data
def testset_accuracy(model, testloader):
    
    # Bring model to GPU
    model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        model.to(device)
        model.eval()
        for data in testloader:
            images, labels = data



# Function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath='checkpoint.pth'):
      
    checkpoint = torch.load(filepath)
    
    model = models.vgg16(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    im = Image.open(image)
    
    width, height = im.size
    # Resize image to make the shortest side 256 pixels
    if im.width > im.height:   
        (width, height) = (im.width, 256)
    elif im.width < im.height:
        (width, height) = (256, im.height)
    else:
        (width, height) = (256, 256)
    
    im.thumbnail((width, height), Image.ANTIALIAS)
    
    # new size of image
    width, height = im.size
    
    # Crop at center, make image 224x224
    reduce = 224
    left = (width - reduce)/2 
    top = (height - reduce)/2
    right = left + 224 
    bottom = top + 224

    im = im.crop((left, top, right, bottom))

    np_image = np.array(im)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    np_image = (np_image - mean) / std
    
    image = np_image.transpose((2, 0, 1))
    
    return image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
def predict(image_path, model, device = 'gpu', topk=1):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image = process_image(image_path)
    
    # Convert image to a FloatTensor and add a 'batch_size' dimension with .unsqueeze_(0)
    image = torch.from_numpy(image).type(torch.FloatTensor).unsqueeze_(0)
    
    # Select between gpu and cpu
    if device == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda:0') 
    else:
        device = torch.device('cpu')

    # Bring model to device
    model.to(device)   
    
    with torch.no_grad():
        model.eval()
        output = model.forward(image.cuda())
        ps = torch.exp(output)
        probs, idx = ps.topk(topk, dim=1)
        
    probs, idx = probs.to('cpu'), idx.to('cpu')
    probs = probs.numpy () # converting both to numpy array
    idx = idx.numpy () 
    
    probs = probs.tolist () [0] # converting both to list
    idx = idx.tolist () [0]
    
    
    mapping = {val: key for key, val in
                model.class_to_idx.items()
                }
    
    classes = [mapping [item] for item in idx]
    
    class_names = [cat_to_name [item] for item in classes]
    class_names = np.array(class_names)
   
    classes = np.array(classes) # converting to Numpy array 
    
    return print(probs, class_names)


