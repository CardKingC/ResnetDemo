from models.fusion_model.resnet import *
from models.fusion_model.vit import ViT
import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet=resnet18(useCli=False)
        self.vit=ViT(
            image_size=32,
            patch_size=4,
            num_classes=1,
            channels=1,
            dim=512,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        # self.layer_norm=[]
        # for i,net in enumerate(nets):
        #     self.layer_norm.append(nn.LayerNorm(in_features[i]))
        self.fc=nn.Linear(in_features=1024,out_features=1)
        self.sigmoid=nn.Sigmoid()
    def forward(self, x):
        x_1=self.resnet(x)
        x_2=self.vit(x)
        x_cat=torch.cat((x_1,x_2),dim=1)
        output=self.fc(x_cat)
        output=self.sigmoid(output)
        return output


