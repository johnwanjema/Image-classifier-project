import torch
import torch.nn.functional as F
import argparse
import fc_model

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Define command line arguments
    parser.add_argument('--image', default='flowers/test/1/image_06743.jpg', type=str, help='path to image')
    parser.add_argument('--checkpoint', default='model_checkpoint.pth')
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('--mode', default='cpu')
    parser.add_argument('--top_k', default=3)

    # Print the configurations
    print("Here are the configurations to be used to predict the type of flower \n", parser.parse_args())

    # Parse the arguments
    inputs = parser.parse_args()

    # Load checkpoint
    model, class_to_idx, epochs, learning_rate, optimizer_state_dict = fc_model.load_checkpoint(inputs.checkpoint)

    # Use GPU if it's available
    device = torch.device('cuda' if torch.cuda.is_available() and inputs.mode == 'gpu' else 'cpu')

    # Move the model to the selected device
    model.to(device)

    # Predict top k classes
    top_k_probabilities, top_k_classes = fc_model.predict_top_k_classes(inputs.image, model, inputs.top_k)

    # Map labels from category names
    labels = fc_model.map_labels(inputs.category_names)

    # Map indices to class labels
    idx_to_class = {x: y for y, x in class_to_idx.items()}

    # Map top classes to their corresponding labels
    top_classes = [idx_to_class[x] for x in top_k_classes]
    predicted_labels = [labels.get(str(cls), f'Class {cls}') for cls in top_classes]

    # Print predicted labels
    print(predicted_labels)

# Call the function to run the script
if __name__ == "__main__":
    # Call the function to run the script
    get_input_args()
