import numpy as np
import torch.nn as nn
import torch


class RFBase(nn.Module):
    def __init__(self, config, final_size):
        super().__init__()
        if not config.use_posenc:
            first_layer = nn.Linear(2, config.hidden_size)
        else:
            first_layer = nn.Linear(4*config.n_freqs, config.hidden_size)
        intermediate_layers = []
        for _ in range(config.n_layers-2):
            if config.layer_norm:
                intermediate_layers.append(nn.LayerNorm(config.hidden_size))
            elif config.batch_norm:
                intermediate_layers.append(nn.BatchNorm1d(config.hidden_size, affine=False))
            intermediate_layers.append(nn.Linear(config.hidden_size, config.hidden_size))
        final_layer = nn.Linear(config.hidden_size, final_size)
        self.layers = nn.ModuleList([first_layer] + intermediate_layers + [final_layer])
        if config.dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=config.dropout)
        else:
            self.dropout = None
        self.relu = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = self.relu(x)
            if self.dropout is not None:
                x = self.dropout(x)
        x = self.layers[-1](x)
        return x
    
class RFPalette(nn.Module):

    def __init__(self, config):
        super().__init__()
        palette = torch.zeros((config.palette_size, 3),dtype=torch.float32)
        torch.nn.init.uniform_(palette,a=0.0,b=1.0)
        self.palette = nn.parameter.Parameter(palette)

    def forward(self, raw_weights):
        color_indices = torch.argmax(raw_weights, dim=1)
        final_color = self.palette[color_indices]
        return final_color


class RadianceField2D(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.base = RFBase(config, 3)
    
    def forward(self, x):
        return self.base(x)


class RadianceField2DPalette(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.base = RFBase(config, config.palette_size)
        self.palette = RFPalette(config)        
    
    def forward(self, x):
        raw_weights = self.base(x)
        final_color = self.palette(raw_weights)
        return raw_weights, final_color

    # def compute_radiance(self, x):
    #     '''
    #     forward() only computes the palette weights. This function selects the radiance
    #     corresponding the largest palette weight.        
    #     '''
    #     x = self.forward(x)
    #     color_indices = torch.argmax(x, dim=1)
    #     final_color = self.palette[color_indices]
    #     return final_color

# optimizations:
# 1) positional encoding
# 2) normalization
# 3) activation