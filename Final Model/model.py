from torch import nn,optim
import torch.functional as F
import pytorch_lightning as pl
import torch
import torchmetrics


class CNNModel(pl.LightningModule):
    def __init__(self,num_classes,lr):
        super().__init__()
        
        self.conv = nn.Sequential(   # 3x224x224
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5), # 32x220x220
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 32x110x110 # 32x110x110
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5), # 64x106x106
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  #64x53x53
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5), # 128x49x49
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))   # 128x24x24
            
        
        self.linear = nn.Sequential(
                nn.Linear(128*24*24,512), # FC1
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(512,256),      # FC2
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(256),
                nn.Linear(256,out_features=num_classes), # FC3
                nn.LogSoftmax(dim=1))
                
        
        self.Lr = lr
        self.lossfn = nn.NLLLoss()
        self.acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)

        
        self.validation_step_outputs = []
        
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
    
    def training_step(self,batch,batch_idx):
        input, label = batch
        output = self(input)
        loss = self.lossfn(output,label)
        return loss
    
    def validation_step(self,batch,batch_idx):
        input, label = batch
        output = self(input)
        loss = self.lossfn(output,label)
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss)
        y_pred = torch.argmax(output,dim=1)
        #pred = output.data.max(1, keepdim=True)[1]
        self.acc.update(y_pred, label)
        
    def on_validation_epoch_end(self):
        mean_val = torch.mean(torch.tensor(self.validation_step_outputs))
        self.log('mean_val', mean_val)
        self.validation_step_outputs.clear()  # free memory
        val_accuracy = self.acc.compute()
        self.log("val_accuracy", val_accuracy)
        # reset all metrics
        self.acc.reset()
        print(f"\nVal Accuracy: {val_accuracy:.4} "\
        f"Val Loss: {mean_val:.4}")
        
        
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(),lr=self.Lr)
        
    
        
        
        
        
        
        
        
        
        
        