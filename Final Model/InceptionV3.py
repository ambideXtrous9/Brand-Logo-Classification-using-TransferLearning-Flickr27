from torch import nn,optim
import pytorch_lightning as pl
import timm
import torch
import torchmetrics



class InceptionV3(pl.LightningModule):
    def __init__(self, num_classes, lr):
        super(InceptionV3, self).__init__()
        
        self.Lr = lr
        self.validation_step_outputs = []
        self.lossfn = nn.NLLLoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

        # Load the pretrained InceptionV3 model
        self.model = timm.create_model('inception_v3', pretrained=True)

        # Freeze training for all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the classifier head with a new one
        self.model.fc = nn.Sequential(
                            nn.BatchNorm1d(self.model.fc.in_features),
                            nn.Linear(self.model.fc.in_features, 256),
                            nn.Dropout(0.2),
                            nn.ReLU(inplace=True),
                            nn.BatchNorm1d(256),
                            nn.Linear(256, num_classes),
                            nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        out = self.model(x)
        return out
    
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