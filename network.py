import numpy as np
import torch.nn as nn

class RadianceField2D(nn.Module):

    def __init__(self, config):
        super().__init__()
        if not config.use_posenc:
            first_layer = nn.Linear(2, config.hidden_size)
        else:
            first_layer = nn.Linear(4*config.n_freqs, config.hidden_size)
        intermediate_layers = []
        for _ in range(config.n_layers-2):
            intermediate_layers.append(nn.Linear(config.hidden_size, config.hidden_size))
        final_layer = nn.Linear(config.hidden_size, 3)
        self.layers = nn.ModuleList([first_layer] + intermediate_layers + [final_layer])
        self.relu = nn.ReLU()
    
    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.relu(x)
        x = self.layers[-1](x)
        return x


# optimizations:
# 1) positional encoding
# 2) normalization
# 3) activation