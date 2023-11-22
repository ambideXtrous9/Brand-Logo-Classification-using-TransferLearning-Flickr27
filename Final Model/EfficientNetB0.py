from torch import nn,optim
import pytorch_lightning as pl
import timm
import torch
import torchmetrics



class EfficientNet(pl.LightningModule):
    def __init__(self,num_classes,lr):
        super(EfficientNet, self).__init__()
        
        
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.aux_logits=False
        
        self.Lr = lr
        
        self.validation_step_outputs = []
        
        self.lossfn = nn.NLLLoss()
        
        self.acc = torchmetrics.Accuracy(task="multiclass",num_classes=num_classes)

        # Freeze training for all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.classifier = nn.Sequential(
                    nn.BatchNorm1d(self.model.classifier.in_features),
                    nn.Linear(self.model.classifier.in_features, 256), 
                    nn.Dropout(0.2),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(256),
                    nn.Linear(256, num_classes),
                    nn.LogSoftmax(dim=1))
        # add metrics
        
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