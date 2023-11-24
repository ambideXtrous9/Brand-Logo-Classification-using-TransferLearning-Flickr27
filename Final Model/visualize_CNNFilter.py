from model import CNNModel
import pytorch_lightning as pl 
import matplotlib.pyplot as plt
import torch
import config

cnnmodel = CNNModel(num_classes=config.NUM_CLASSES,lr=config.LR)

cppath = 'checkpoints/Efficient-Best.ckpt'

checkpoint = torch.load(cppath)
cnnmodel.load_state_dict(checkpoint['state_dict'])

cnnmodel.freeze()
cnnmodel.eval()

# Get the weights of the first convolutional layer
conv1_weights = cnnmodel.conv[0].weight.data.cpu()

# Visualize the filters in a 4x8 matrix
fig, axs = plt.subplots(4, 8, figsize=(15, 8))

for i in range(4):
    for j in range(8):
        filter_index = i * 8 + j
        axs[i, j].imshow(conv1_weights[filter_index].numpy().transpose(1, 2, 0))
        axs[i, j].axis('off')

plt.show()