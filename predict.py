import argparse
import numpy as np
import pandas as pd
import torch
import json
from PIL import Image

# Main execution of the program.
def main():
    # Get parameters from command line.
    in_arg = get_input_args()

    # Extract them to variables.
    path_to_image = in_arg.path_to_image
    checkpoint = in_arg.checkpoint
    top_k = in_arg.top_k
    cat_to_name = in_arg.cat_to_name
    gpu = in_arg.gpu

    # Set hyper variables.
    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

    # Load the model from directory.
    model = load_checkpoint(checkpoint)

    # Predict the image's category.
    probs, classes = predict(path_to_image, model, device, int(top_k))
    
    # Print results.
    view_classify(probs, classes, cat_to_name)

# Get paramerters from command line.
def get_input_args():
    """
    Get paramerters from command line.
    Parameters:
        None
    Returns:
        parse_args() - data structure
    """
    # Creates parse 
    parser = argparse.ArgumentParser(description = 'Predicting image with NN options')

    # Creates arguments
    parser.add_argument(
        'path_to_image',
        action = 'store',
        help = 'path to a single image file'
    )
    parser.add_argument(
        'checkpoint',
        action = 'store',
        help = 'path to the checkpoint file'
    )
    parser.add_argument(
        '--top_k',
        type = int,
        dest = 'top_k',
        default = 5,
        help = 'top k most likely classes'
    )
    parser.add_argument(
        '--category_names ',
        type = str,
        dest = 'cat_to_name',
        help = 'path to the JSON file containing category names'
    )
    parser.add_argument(
        '--gpu',
        dest = 'gpu',
        action = 'store_true',
        help = 'use the GPU for training'
    )

    # returns parsed argument collection
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    for param in model.parameters():
        param.requires_grad = False
    return model

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = torch.from_numpy(process_image(image_path))
    image = image.unsqueeze(0).float()
    model, image = model.to(device), image.to(device)
    model.eval()
    model.requires_grad = False
    outputs = torch.exp(model.forward(image)).topk(topk)
    probs, classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    return probs, classes

def view_classify(probs, classes, cat_to_name):
    if cat_to_name is None:
        name_classes = classes
    else:
        with open(cat_to_name, 'r') as f:
            cat_to_name_data = json.load(f)
        name_classes = [cat_to_name_data[i] for i in classes]
    df = pd.DataFrame({
        'classes': pd.Series(data = name_classes),
        'values': pd.Series(data = probs, dtype='float64')
    })
    print(df)

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    size = 256, 256
    im = Image.open(image)
    im = im.resize(size)
    im = im.crop((16, 16, 240, 240))
    np_image = np.array(im)
    np_image_norm = ((np_image / 255) - ([0.485, 0.456, 0.406])) / ([0.229, 0.224, 0.225])
    np_image_norm = np_image_norm.transpose((2, 0, 1))
    return np_image_norm

if __name__ == '__main__':
    main()
