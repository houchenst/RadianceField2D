import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import json
import math
from PIL import Image
import glob

from data import positional_encoding, pixel_to_continuous_coordinate
from network import RadianceField2D, RadianceField2DPalette
from data import ImageDataset


def load_history(config, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'r') as f:
        loss_history = json.loads(f.read())
    return loss_history


def save_history(config, history, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'w') as f:
        f.write(json.dumps(history))

def plot_loss_curve(config, loss_dict):
    '''
    Plots the loss curve and writes to file
    '''
    # loss_history = load_history(config, "loss")
    # loss_history['train_loss'] += loss_dict['train_loss']
    # loss_history['val_loss'] += loss_dict['val_loss']
    # save_history(config, loss_history, "loss")
    loss_history = loss_dict
    f, ax = plt.subplots()
    ax.plot([x for x in range(len(loss_history['train_loss']))], loss_history['train_loss'], color='b', label='train')
    ax.plot([x for x in range(len(loss_history['val_loss']))], loss_history['val_loss'], color='b', label='val')
    f.suptitle(config.expname + " Loss")
    plt.savefig(os.path.join(config.expdir, "loss_curves.png"))
    plt.close()

def plot_metrics_curve(config, dict, type):
    f, ax = plt.subplots()
    ax.plot([x for x in range(len(dict['train_' + type]))], dict['train_' + type], color='b', label='train')
    ax.plot([x for x in range(len(dict['val_' + type]))], dict['val_' + type], color='r', label='val')
    f.suptitle(config.expname + " " + type)
    plt.savefig(os.path.join(config.expdir, type + "_curves.png"))
    plt.close()

def render_image(model, config):
    '''
    Renders the image from the radiance field
    '''
    model.eval()
    coords = []
    for i in range(config.ydim):
        for j in range(config.xdim):
            coords.append(torch.tensor(positional_encoding(pixel_to_continuous_coordinate((config.ydim, config.xdim), i, j, x_offset=0.5, y_offset=0.5), config)))
    outputs = []
    for i in range(math.ceil(len(coords)/config.batchsize)):
        batch_input = torch.stack(coords[i*config.batchsize:(i+1)*config.batchsize],dim=0).to(config.device)
        if not config.palette:
            outputs.append(model(batch_input))
        else:
            _, batch_output = model(batch_input)
            outputs.append(batch_output)
    outputs = torch.vstack(outputs).detach().cpu().numpy()
    outputs = outputs.reshape((config.ydim, config.xdim, 3))
    outputs = np.clip(outputs, 0., 1.)
    return outputs

def show_palette(model, config, onscreen=False):
    '''
    Displays and save the color palette of the model
    '''

    def is_square(i):
        root = int(math.sqrt(i))
        if root*root != i:
            return None
        return root

    root = is_square(config.palette_size)
    if root is not None:
        palette = model.palette.palette.detach().cpu().numpy()
        palette = palette.reshape((root,root,3))
        if not onscreen:
            palette = Image.fromarray((palette*255).astype(np.uint8))
            palette = palette.resize((256,256), resample=Image.NEAREST)
            palette.save(os.path.join(config.expdir, f"{config.expname}_palette.png"))
        else:
            ax = plt.subplot()
            ax.imshow(palette)
            plt.show()

def make_video(config):
    '''
    Takes the frames that have been rendered during different epochs
    and combines them into a video
    '''
    pass
    #TODO: combine current images into video and write to file

def setup_dual(config):
    '''
    Sets up experiment directories for dual training mode
    '''
    

def setup(config):
    '''
    Sets up experiment directories
    Creates and loads model/optimizer
    Creates dataloader
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device: {device}")
    config.device = device

    # setup experiment folder
    expdir = os.path.join(".", "experiments", config.expname)
    config.expdir = expdir
    if not os.path.exists(config.expdir):
        os.mkdir(config.expdir)
    if not os.path.exists(os.path.join(config.expdir, "reconstructions")):
        os.mkdir(os.path.join(config.expdir, "reconstructions"))
    if not os.path.exists(os.path.join(config.expdir, "checkpoints")):
        os.mkdir(os.path.join(config.expdir, "checkpoints"))

    # TODO: save config file?

    # make model and optimizer
    last_checkpoint = None
    all_checkpoints = glob.glob(os.path.join(config.expdir, "checkpoints", "*"))
    if len(all_checkpoints) > 0:
        last_checkpoint = all_checkpoints[-1]    

    # Create Model
    if config.palette:
        model = RadianceField2DPalette(config).to(device)
    else:
        model = RadianceField2D(config).to(device)

    if last_checkpoint is not None:
        model.load_state_dict(torch.load(last_checkpoint, map_location=device)["model_state_dict"])
        print(f"Loading model from checkpoint: '{last_checkpoint}'")

    # Create Optimizers
    # If using palette model 2 optimizers are needed 
    optimizer = torch.optim.Adam(model.base.parameters(), lr=config.lr)
    palette_optimizer = None
    if config.palette:
        palette_optimizer = torch.optim.Adam(model.palette.parameters(), lr=config.lr)

    if last_checkpoint is not None:
        optimizer.load_state_dict(torch.load(last_checkpoint, map_location=device)["optimizer_state_dict"])
        if config.palette:
            palette_optimizer.load_state_dict(torch.load(last_checkpoint, map_location=device)["palette_optimizer_state_dict"])

    if not os.path.exists(os.path.join(config.expdir, "loss_history.txt")):
        loss_history = {}
        loss_history["train_loss"] = []
        loss_history["val_loss"] = []
        with open(os.path.join(config.expdir, "loss_history.txt"), 'w') as f:
            f.write(json.dumps(loss_history))

    if not os.path.exists(os.path.join(config.expdir, "mse_history.txt")):
        mse_history = {}
        mse_history["train_mse"] = []
        mse_history["val_mse"] = []
        with open(os.path.join(config.expdir, "mse_history.txt"), 'w') as f:
            f.write(json.dumps(mse_history))

    if not os.path.exists(os.path.join(config.expdir, "ssim_history.txt")):
        ssim_history = {}
        ssim_history["train_ssim"] = []
        ssim_history["val_ssim"] = []
        with open(os.path.join(config.expdir, "ssim_history.txt"), 'w') as f:
            f.write(json.dumps(ssim_history))

    #Make data loader
    dataset = ImageDataset(config)
    loader = DataLoader(dataset, batch_size=config.batchsize, shuffle=True, drop_last=True)
    return model, loader, optimizer, palette_optimizer