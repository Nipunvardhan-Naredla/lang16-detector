import torch.nn as nn


class LanguageModel(nn.Module):
    def __init__(self):
        '''Defines model architecture


        '''
        super().__init__()
        self.embedding = nn.Embedding(120000, 15)
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.LSTM(input_size = 15,
                          hidden_size = 240,
                          num_layers = 6,
                          batch_first = True,
                          )

        self.fc2 = nn.Linear(240,16)#change 16 depending on amount of classes

    def forward(self, x):
        '''the forward pass
        '''
        x = self.embedding(x).long()
        x = x.squeeze(1).float()
        x, _ = self.fc1(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        
        return x