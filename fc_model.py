import torch
from torchvision import datasets, transforms, models
import json

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