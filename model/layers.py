from torch import nn, tensor
import torch
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, hidden, output_size, dropout):
        super(MLP, self).__init__()
        

        self.input = input_size
        self.output_size = output_size
        self.hidden = hidden

        module = [
            nn.Linear(self.input,self.hidden[0]),
            nn.ReLU(inplace = True),
            nn.Dropout(dropout)
        ]
        for i in range(len(self.hidden) - 1):
            module.append(nn.Linear(self.hidden[i], self.hidden[i+1]))
            module.append(nn.ReLU(inplace = True))
            module.append(nn.Dropout(dropout))

        module.append(nn.Linear(self.hidden[-1], output_size))
        module = nn.ModuleList(module)
        

        self.linear = nn.Sequential(
            *module
        )

    def forward(self, x):
        '''
        input : x (N,D)
        output : out (N,O)
        '''

        output = self.linear(x)
        output = F.sigmoid(output)

        return output