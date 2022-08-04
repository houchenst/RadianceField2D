import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import math


def pixel_to_continuous_coordinate(dims, y_idx, x_idx, x_offset=0.5, y_offset=0.5):
    '''
    This function takes in an image and a pixel index.
    The output is the continuous coordinates of the specified pixel after
    the image has been rescaled so that the largest dimension has length 1
    and the image is centered within the unit square (between 0 and 1)

    The offsets determine where in the pixel box the continuous positions are located. 
    An offset of 0.5 is the pixel center, while 0 or 1 would be on the pixel boundaries.
    '''
    height, width = dims
    large_dim = max(height, width)
    y_coord = pixel_to_continuous_1d(height, y_idx, (large_dim-height)/large_dim, y_offset)
    x_coord = pixel_to_continuous_1d(width, x_idx, (large_dim-width)/large_dim, x_offset)
    y_coord = 1-y_coord #flip image vertically because the image origin is in the upper left
    return y_coord, x_coord
    

def pixel_to_continuous_1d(dim_size, idx, padding, offset):
    '''
    Outputs the continuous coordinate (in 1 dimension) given that the image
    dimension, the pixel index, and the amount padding within the unit square.
    '''
    pos = (idx+offset)/dim_size*(1.-padding) + padding/2.
    pos = 2*pos - 1. #translate/scale to be between -1 and 1
    return pos

def positional_encoding(yx, config):
    '''
    Applies NeRF style positional encoding to the coordinate
    '''
    y_coord, x_coord = yx
    encoder = lambda p: [y for x in range(config.n_freqs) for y in [math.sin(2**x*math.pi*p), math.cos(2**x*math.pi*p)]]
    if config.use_posenc:
        encoded_y = encoder(y_coord)
        encoded_x = encoder(x_coord)
        return encoded_y + encoded_x
    else:
        return [y_coord, x_coord]

class ImageDataset(Dataset):

    def __init__(self, config):
        img = Image.open(config.imagepath)
        img = np.array(img)
        if config.drop_alpha:
            img = img[:,:,:-1]
        img = img / 255.
        self.colors = []
        self.coords = []
        config.ydim = img.shape[0]
        config.xdim = img.shape[1]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                self.coords.append(torch.tensor(positional_encoding(pixel_to_continuous_coordinate(img.shape[:2], i, j), config), dtype=torch.float32))
                self.colors.append(torch.tensor(img[i,j], dtype=torch.float32))
        # print(len(self))
        
    def __len__(self):
        return len(self.colors)

    def __getitem__(self, idx):
        return self.coords[idx], self.colors[idx]


if __name__ == "__main__":
    from parse import parser
    config = parser.parse_args()
    ImageDataset(config)

