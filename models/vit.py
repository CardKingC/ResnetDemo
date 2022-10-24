from vit_pytorch import ViT
import torch
import torch.nn as nn
from conf import global_settings as gs

class myVit(ViT):
    def __init__(self,image_size = 32,
            patch_size = 4,
            num_classes = 1,
            channels=1,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1):
        super().__init__(image_size = image_size,
            patch_size = patch_size,
            num_classes = num_classes,
            channels=channels,
            dim = dim,
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            dropout = dropout,
            emb_dropout = emb_dropout)
        self.sigmoid=nn.Sigmoid()

    def forward(self, img):
        x=super().forward(img)
        x=self.sigmoid(x)
        return x
