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

parser.add_argument('--image',  action='store',
                    dest='image',
                    help='number of classes', default='flowers/test/12/image_03994.jpg')

parser.add_argument('-k',  action='store',
                    dest='topk',
                    help='number of classes', type=int, default = 3)

parser.add_argument('--device',  action='store',
                    dest='device',
                    help='Choose between "gpu" and "cpu"', default = 'gpu')

parser.add_argument ('--category_names', help = 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str)

args = parser.parse_args()
topk = args.topk
device = args.device
image = args.image
category_names = args.category_names


# Load the mapping if provided
if category_names:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        pass

# Load model
model = utils.load_checkpoint('checkpoint.pth')

utils.predict(image, model, device, topk)






