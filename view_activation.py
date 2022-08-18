import json
import os

import numpy as np
from PIL import Image
from data import ImageDataset, pixel_to_continuous_coordinate, positional_encoding
from parse import parser
import torch
import matplotlib.pyplot as plt
import matplotlib
import utils

def get_coords(config, grid_sz):
    img = Image.open(config.imagepath)
    img = np.array(img)
    if config.drop_alpha:
        img = img[:, :, :-1]
    height = img.shape[0]
    width = img.shape[1]
    num_grids = int(np.ceil(float(height)/grid_sz) * np.ceil(float(width)/grid_sz))
    grids = []
    grid_shapes = []

    for i in range(int(np.ceil(height/grid_sz))):
        for j in range(int(np.ceil(width/grid_sz))):
            shape = [0, 0]
            if (i+1) * grid_sz <= height:
                shape[0] = grid_sz
            else:
                shape[0] = height % grid_sz
            if (j+1) * grid_sz <= width:
                shape[1] = grid_sz
            else:
                shape[1] = width % grid_sz
            grid_shapes.append(shape)

    for i in range(num_grids):
        grids.append([])

    for i in range(height):
        for j in range(width):
            grid_idx = int(int(j/grid_sz) + np.ceil(width/grid_sz)*(int(i/grid_sz)))
            grids[grid_idx].append(positional_encoding(pixel_to_continuous_coordinate(img.shape[:2], i, j), config))
    return grids, num_grids, grid_shapes


def get_activations(net_in, net_act, name):
    def hook(model, input, output):
        net_in[name] += np.sum(output.detach().cpu().numpy(), axis=0)
        net_act[name] += np.sum(torch.nn.functional.relu(output.detach()).cpu().numpy(), axis=0)
    return hook


if __name__ == "__main__":
    config = parser.parse_args()
    model, _, _, _ = utils.setup(config)

    grid_sz = 160
    grids, num_grids, grid_shapes = get_coords(config, grid_sz)

    hidden_input = {}
    hidden_act = {}
    top_activations = {}

    for j in range(config.n_layers-2):
        hidden_input[j + 1] = np.zeros(shape=config.hidden_size)
        hidden_act[j + 1] = np.zeros(shape=config.hidden_size)
        model.base.layers[j+1].register_forward_hook(get_activations(hidden_input, hidden_act, j + 1))

    model.eval()

    if not os.path.exists(os.path.join(config.expdir, "weight_heatmap_" + str(grid_sz))):
        os.mkdir(os.path.join(config.expdir, "weight_heatmap_" + str(grid_sz)))

    for n, grid in enumerate(grids):
        top_activations[n] = []
        num_batches = int(np.ceil(len(grid)/config.batchsize))
        outputs = []
        for i in range(0, len(grid), config.batchsize):
            batch = torch.tensor(grid[i:min(i+config.batchsize, len(grid))], dtype=torch.float32).to(config.device)
            outputs.append(model(batch))
        outputs = torch.vstack(outputs).detach().cpu().numpy()
        outputs = outputs.reshape((grid_shapes[n][0], grid_shapes[n][1], 3))
        outputs = np.clip(outputs, 0., 1.)
        pred_img = (outputs*255).astype(np.uint8)
        pred_img = Image.fromarray(pred_img)
        pred_img.save(os.path.join(config.expdir, "weight_heatmap_" + str(grid_sz), f"grid_{n}_img.png"))

        fig, axs = plt.subplots(1, config.n_layers-2)
        fig.set_size_inches((config.n_layers-2) * 5, 8)
        fig.suptitle("Grid " + str(n))

        for j in range(config.n_layers-2):
            hidden_act[j+1] = hidden_act[j+1]/num_batches
            axs[j].imshow(np.expand_dims(hidden_act[j + 1], 1), aspect=0.3)
            top_idx = np.argsort(-hidden_act[j+1])[:5]
            top_activations[n].append(top_idx.tolist())

            axs[j].set_title("Layer " + str(j+1))
            plt.colorbar(
                mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min(hidden_act[j + 1]), vmax=max(hidden_act[j + 1])),
                                                    cmap='viridis'), ax=axs[j])
            hidden_act[j + 1] = np.zeros(shape=config.hidden_size)
            hidden_input[j + 1] = np.zeros(shape=config.hidden_size)

        plt.savefig(os.path.join(config.expdir, "weight_heatmap_" + str(grid_sz), f"grid_{n}_weight.png"))
        plt.close()

    with open(os.path.join(config.expdir, "weight_heatmap_" + str(grid_sz), "top_activations.txt"), 'w') as f:
        f.write(json.dumps(top_activations))



