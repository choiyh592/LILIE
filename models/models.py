import torch
from torch import optim, nn, utils, Tensor
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from timm.layers import Mlp

from .pool.poolers import AttentiveDelta, NNDelta, Delta

class LILIE(L.LightningModule):
    '''
    Learning-based Infernce of Longitudinal Intra-patient EEGs
    
    '''
    def __init__(self, input_dim, embedding_size, num_classes, pool_method="Attentive", clf_method="Linear"):
        super().__init__()
        
        if pool_method == "Attentive":
            self.pooler = AttentiveDelta(input_dim=input_dim, embed_dim=embedding_size, num_heads=8)
        elif pool_method == "NN":
            self.pooler = NNDelta(input_shape = (0, 32, 256), embed_dim = 256)
        elif pool_method == "Linear":
            self.pooler = Delta()
        elif pool_method == "Raw":
            self.pooler = Delta()
        else:
            print("Error Initializing")

        if clf_method == "Linear":
            self.clf = nn.Linear(embedding_size, num_classes)
        elif clf_method == "NN":
            self.clf = Mlp(in_features=embedding_size, hidden_features=embedding_size * 2, out_features=num_classes, act_layer=nn.GELU, drop=0.15)

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.val_auroc = BinaryAUROC()

    def training_step(self, batch, batch_idx):
        x_0, x_1, y = batch
        pooled_x = self.pooler(x_0, x_1)
        y_pred = self.clf(pooled_x)
        loss = nn.functional.cross_entropy(y_pred, y)

        preds = torch.argmax(y_pred, dim=1)
        self.train_acc(preds, y)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        probs = torch.softmax(y_pred, dim=1)[:, 1]
        self.train_auroc.update(probs, y)
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x_0, x_1, y = batch
        
        y_pred = self.clf(self.pooler(x_0, x_1))
        
        loss = nn.functional.cross_entropy(y_pred, y)
        preds = torch.argmax(y_pred, dim=1)
        
        self.val_acc(preds, y)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True, sync_dist=True)

        probs = torch.softmax(y_pred, dim=1)[:, 1]
        
        self.val_auroc(probs, y)
        self.log("val_auroc", self.val_auroc, on_epoch=True, prog_bar=True, sync_dist=True)

    def get_embeddings(self, x_0, x_1):
        return self.pooler(x_0, x_1)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-5, weight_decay=0.0005)
        return optimizer

