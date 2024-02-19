import torch
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
from torch import nn,optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


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


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(image)
#     input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def predict_top_k_classes(image_path,model, k=5):
    # Pass model to cpu
    model.to('cpu')

    # Load and preprocess the image
    input_tensor = process_image(image_path)
    
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the GPU if available
    if torch.cuda.is_available():
        input_batch = input_batch.cuda()

    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        output = model(input_batch)

    # Get the top k predicted classes and probabilities
    probabilities, predicted_classes = torch.topk(nn.functional.softmax(output, dim=1), k)

    # If on CUDA, move tensors to CPU before converting to NumPy
    if torch.cuda.is_available():
        probabilities = probabilities.cpu()
        predicted_classes = predicted_classes.cpu()

    # Convert tensor results to Python lists
    probabilities = probabilities.squeeze().numpy().tolist()
    predicted_classes = predicted_classes.squeeze().numpy().tolist()

    return probabilities, predicted_classes

def plot_probabilities(image_path, probabilities,class_to_idx, predicted_classes, class_labels):
    # Load and show the input image
    img = process_image(image_path)
    imshow(img)

    # Convert predicted class indices to class names using the provided dictionary
    idx_to_class = {x: y for y, x in class_to_idx.items()}
    top_classes = [idx_to_class[x] for x in predicted_classes]

    predicted_labels = [class_labels.get(str(cls), f'Class {cls}') for cls in top_classes]

    # Plot the probabilities as a bar graph
    fig, ax = plt.subplots()
    y_pos = np.arange(len(probabilities))
    ax.barh(y_pos, probabilities, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predicted_labels)
    ax.invert_yaxis()  # Invert y-axis for better visualization
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Predicted Classes')

    plt.show()
