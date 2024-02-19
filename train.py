import torch
import torch.nn.functional as F
import argparse
from torch import optim
import fc_model

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument('--dir', type=str, default='flowers/',
                        help='path to folder of images')
    parser.add_argument('--arch', default='VGG', choices=['VGG', 'Densenet'])
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_units', default=512)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--mode', default='cpu')
    parser.add_argument('--save_file', default='model_checkpoint.pth')

    # Parse the arguments
    inputs = parser.parse_args()

    # Print the configurations
    print("Here are the configurations to be used to train the model \n", inputs)

    # Setup dataset
    dataset = fc_model.data_setup(inputs.dir, 32)

    # Create the neural network model
    model = fc_model.create_network(inputs.arch, inputs.hidden_units)

    # Define loss function (criterion) and optimizer
    optimizer = optim.Adam(model.classifier.parameters(), lr=inputs.learning_rate)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() and inputs.mode == 'gpu' else 'cpu')

    # Check if Mac MPS is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    # Move the model to the selected device
    model.to(device)

    # Train the model
    fc_model.train(model, dataset['train'], dataset['validation'],inputs.epochs,inputs.learning_rate)

    # Test the model
    fc_model.validation(model, dataset['test'])

    # Save the trained model
    fc_model.save_model(model, inputs.save_file)

# Call the function to run the script
if __name__ == "__main__":
    # Call the function to run the script
    get_input_args()

