import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn


def data_setup(data_dir,batch_size):
    # Train trasforms
    train_transforms = transforms.Compose([transforms.RandomRotation(10),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor()])

    # Test trasforms
    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor()])

    train_datasets = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    validation_datasets = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_datasets = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    
    return {'train': torch.utils.data.DataLoader(train_datasets, batch_size=batch_size,shuffle=True),
            'validation': torch.utils.data.DataLoader(validation_datasets, batch_size=batch_size),
            'test': torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)}

def map_labels(file):
    
    with open(file, 'r') as f:
        cat_to_name = json.load(f,strict=False)

    return cat_to_name 

def create_network(arch="VGG",hidden_units=512):
    
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
            
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(4096, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(1000, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc4', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
       
    
        for param in model.parameters():
            param.requires_grad = False
                                  
        model.classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 1000)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(1000, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    return model