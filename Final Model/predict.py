from dataloader import LogoDataModule,nrml,resize,transforms
import config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import CNNModel
from EfficientNetB0 import EfficientNet
from Xception import XceptionNet
import torch
import os
import random
import warnings
warnings.filterwarnings("ignore")

data_module = LogoDataModule(data_folder=config.DATA_FOLDER,
                            batch_size=config.BATCH_SIZE,
                            val_split=config.VAL_SPLIT)

data_module.setup()

cnnmodel = XceptionNet(num_classes=config.NUM_CLASSES,lr=config.LR)

transform_norm = transforms.Compose([
            transforms.ToTensor(),
            resize,
            nrml])

cppath = 'checkpoints/Efficient-Best.ckpt'

checkpoint = torch.load(cppath)
cnnmodel.load_state_dict(checkpoint['state_dict'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def readimage(text_file_path,logo_folder_path):
    with open(text_file_path, 'r') as file:
        lines = file.readlines()

    file_class_pairs = [line.strip().split('\t') for line in lines]

    # Randomly choose a (filename, class) pair
    random_pair = random.choice(file_class_pairs)
    filename, image_class = random_pair
    image_path = os.path.join(logo_folder_path, filename)
    
    return image_path, image_class


def predimg(path,true_class):
    image = Image.open(path).convert('RGB')
    

    cnnmodel.eval()
    
    input_tensor = transform_norm(image).unsqueeze(0)  # Add batch dimension


    with torch.no_grad():
        output = cnnmodel(input_tensor)
        
        probs = torch.exp(output)

        # Get the predicted class index
        index = torch.argmax(output).item()
        
    predicted_class = data_module.dataset.classes[index]
    predprob = round(probs[0][index].item(),2)
    if(predprob < 0.80) : predicted_class = 'None'
    
    plt.imshow(image)
    plt.axis("off")
    # Print the predicted class on the image in red color
    plt.text(20, 20, f'ACTUAL : {true_class}', color='green', fontsize=15, weight='bold')
    plt.text(20, 40, f'PRED : {predicted_class}', color='red', fontsize=15, weight='bold')
    plt.text(20, 60, f'PROB : {predprob}', color='blue', fontsize=15, weight='bold')

    plt.show()

if __name__ == "__main__":
    # Path to the text file containing filenames and classes
    text_file_path = config.QUERY_SET_PATH
    # Path to the 'Logo Folder' containing images
    logo_folder_path = config.MAIN_LOGO_FOLDER
    path,true_class = readimage(text_file_path,logo_folder_path)
        
    predimg(path,true_class)