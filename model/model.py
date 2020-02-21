from torch import nn, tensor
import torch
import torch.nn.functional as F
from model.layers import MLP

class GAN(MLP):
    def __init__(self, input_size, embed_size, hidden, output_size, classes, mode = "cGAN", dropout = 0.2):
        if mode == "cGAN":
            input_ = input_size + embed_size
        else:
            input_ = input_size
        super(GAN, self).__init__(input_ ,hidden, output_size, dropout)
        '''
        input_size : input data dimension
        embed_size : class embeding dimension
        hidden : array of hidden layer number ex) [3,2,4]
        output_size: layer output dimension
        classes : class num of data
        mode : cGAN vs GAN
        dropout : dropout ratio
        '''
        self.mode = mode
        self.embed = nn.Embedding(classes,embed_size)
        

        #self.init_params()
    def init_params(self):
        for p in self.parameters():
            if(p.dim() > 1):
                nn.init.xavier_normal_(p)
        else:
            nn.init.uniform_(p, 0.1, 0.2)

    def forward(self, x, y):
        if self.mode == "cGAN":
            class_emb = self.embed(y)
            x = torch.cat([x,class_emb], dim = 1)

        out = self.linear(x)
        out = F.sigmoid(out)

        return out



