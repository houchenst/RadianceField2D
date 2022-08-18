from turtle import position
import torch
import numpy as np
import os, sys
import glob
import matplotlib.pyplot as plt
import matplotlib
from network import RadianceField2D
from parse import parser
from data import positional_encoding, continuous_to_pixel_coordinate
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(config):
    all_checkpoints = glob.glob(os.path.join(".", "experiments", config.expname, "checkpoints", "*"))
    if len(all_checkpoints) == 0:
        print(f"Could not find any checkpoints for experiment '{config.expname}'")
        exit(1)
    last_checkpoint = all_checkpoints[-1]
    print(f"Loading checkpoint '{last_checkpoint}'")
    model_weights = torch.load(last_checkpoint, map_location=device)["model_state_dict"]
    return model_weights

def show_coordinate(config):
    '''
    Show the location of the input coordinate
    '''
    img = np.array(Image.open(config.imagepath))
    y_idx, x_idx = continuous_to_pixel_coordinate(img.shape[:2], config.y, config.x)
    cross_width = 10
    for i in range(max(0, y_idx-cross_width), min(img.shape[0], y_idx+cross_width)):
        img[i,x_idx,:3] = np.array([255.,0.,0.])
    for i in range(max(0, x_idx-cross_width), min(img.shape[1], x_idx+cross_width)):
        img[y_idx,i,:3] = np.array([255.,0.,0.])
    ax = plt.subplot()
    ax.imshow(img)
    plt.show()
    


def show_gradients(config):
    '''
    Compares the learned weights between models from 2 different experiments
    '''
    # exp1 = "hand_frame1"
    # exp2 = "hand_frame2"

    model_weights = load_weights(config)
    model = RadianceField2D(config).to(device)
    model.load_state_dict(model_weights)

    show_coordinate(config)
    input = [config.y, config.x]
    input = positional_encoding(input, config)
    input = torch.Tensor(input)[None,:].to(device)

    loss_fn = torch.nn.MSELoss()
    model.train()
    output = model(input)
    # delta = torch.zeros((1,3)) + 0.001
    # output_detached = torch.clone(output)
    # output_detached = output_detached.detach() 
    # output_detached += delta
    # loss = loss_fn(output, output_detached)
    # loss = torch.mean(output)
    loss = output[0,2]
    loss.backward()

    weight_grads = []
    bias_grads = []
    weight_names = []
    bias_names = []
    for name, param in model.base.named_parameters():
        if "weight" in name:
            weight_grads.append(param.grad.cpu().numpy())
            weight_names.append(name)
        elif "bias" in name:
            bias_grads.append(param.grad.cpu().numpy())
            bias_names.append(name)

    all_grads = weight_grads + bias_grads
    all_names = weight_names + bias_names
    n_params = len(all_names)
    fig, axs = plt.subplots(1, n_params)
    fig.set_size_inches(n_params*2.0, 2.5)

    for ax, g, name in zip(axs, all_grads, all_names):
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        aspect = 1.
        if len(g.shape) == 1:
            g = g[:,None]
            aspect = 1./(g.shape[0]/4.)
        elif g.shape[0] < 10:
            aspect = g.shape[1]/10.
        ax.imshow(g, aspect=aspect)
        ax.set_xlabel(g.shape)
        ax.set_title(name.split(".")[2] + " " + name.split(".")[1])
    # plt.colorbar(mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_diff, vmax=max_diff),cmap='viridis'), ax=axs[-1])
    plt.show()

# TODO:
# 1) Save comparison to file
# 2) Add SSIM metric
# 2) Show diff and standard deviation (somehow normalize weights?)
# 3) Try using layernorm between layers


if __name__ == "__main__":
    parser.add_argument("-y", type=float, default=0.0, help="y coordinate to show neuron gradients for")
    parser.add_argument("-x", type=float, default=-0.2, help="x coordinate to show neuron gradients for")
    config = parser.parse_args()

    expdir = os.path.join(".", "experiments", config.expname)
    config.expdir = expdir

    show_gradients(config)
    # compare_weights(sys.argv[1], sys.argv[2])