import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn,optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    device = torch.device("mps")

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

def train(model, trainloader, testloader,epochs=1,print_every=10,learning_rate=0.001):

    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.classifier.parameters(),learning_rate)
    steps = 0
    running_loss = 0

    model.to(device)    

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                
                model.train()
                
def validation(model,testloader):
    criterion = nn.NLLLoss()
    model.to(device);
    accuracy = 0
    test_loss = 0
    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        model.eval()
        for inputs, labels in testloader:

            inputs, labels = inputs.to(device), labels.to(device)

            log_ps = model(inputs)
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    model.train()

    print("Accuracy: {:.3f}".format(accuracy/len(testloader)))

def save_model(model,file,dataset,epochs=1,arch='VGG',learning_rate=0.01,hidden_units=512):
    optimizer = optim.AdamW(model.classifier.parameters(),learning_rate)
    checkpoint = {
            'hidden_units': hidden_units,
            'architecture': arch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs,
            'classifier': model.classifier,
            'class_to_idx': dataset.class_to_idx,
            'learning_rate': learning_rate
        }

    torch.save(checkpoint, file)

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    model = create_network(checkpoint['architecture'],checkpoint['hidden_units'])
    
    model.load_state_dict(checkpoint['state_dict'])
        
    return model,checkpoint['class_to_idx'],checkpoint['epochs'],checkpoint['learning_rate'],checkpoint['optimizer_state_dict']