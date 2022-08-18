import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import os
import glob
import json
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from PIL import Image

from network import RadianceField2D
from data import ImageDataset, pixel_to_continuous_coordinate, positional_encoding
from parse import parser
from losses import ColorPickerLoss
import utils


def train(config, model, dataloader, optimizer, palette_optimizer=None):
    '''
    Trains model
    '''
    for name, w in model.named_parameters():
        print(name)

    #TODO training loop
    loss_fn = torch.nn.MSELoss()
    palette_loss_fn = ColorPickerLoss()
    model = model.to(config.device)

    if len(config.tunable_layers) >0:
        for name, param in model.named_parameters():
            if name not in config.tunable_layers:
                param.requires_grad = False

    img = Image.open(config.imagepath)
    img = np.array(img)
    if config.drop_alpha:
        img = img[:, :, :-1]

    mse_dict = load_history(config, "mse")
    loss_dict = load_history(config, "loss")
    ssim_dict = load_history(config, "ssim")

    for e in range(config.epochs):
        model.train()
        print(f"----------  EPOCH {e + 1}/{config.epochs}  ----------")
        train_losses = []
        i = 0
        for batch in tqdm(dataloader):
            i+=1
            coords, radiance = batch
            coords = coords.to(config.device)
            radiance = radiance.to(config.device)
            if not config.palette:
                optimizer.zero_grad()
                pred_radiance = model(coords)
                loss = loss_fn(pred_radiance, radiance)
                loss.backward()
                optimizer.step()
            else:
                color_weights, _ = model(coords)
                weights_loss = palette_loss_fn(radiance, color_weights, model, dim=1)
                optimizer.zero_grad()

                palette_loss = palette_loss_fn(radiance, color_weights, model, dim=0)
                palette_optimizer.zero_grad()

                weights_loss.backward(retain_graph=True)
                palette_loss.backward()

                optimizer.step()
                palette_optimizer.step()
                loss = weights_loss

            train_losses.append(loss.detach().cpu().numpy())
            loss_dict["train_loss"].append(float(np.mean(train_losses)))

        save(model, optimizer, e, loss_dict, mse_dict, ssim_dict, config, img, palette_optimizer)


def load_history(config, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'r') as f:
        loss_history = json.loads(f.read())
    return loss_history


def save_history(config, history, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'w') as f:
        f.write(json.dumps(history))


def save(model, optimizer, epoch, loss_dict, mse_dict, ssim_dict, config, img, palette_optimizer=None):
    '''
    Saves loss curve, checkpoint of the model, and the current image reconstruction
    Only saves checkpoint and images during certain epochs
    '''

    print(f"Epoch training loss: {loss_dict['train_loss'][-1]:.4f}")
    utils.plot_loss_curve(config, loss_dict)

    if epoch % config.save_frequency == 0:
        # save checkpoint
        save_dict = {}
        save_dict['model_state_dict'] = model.state_dict()
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if config.palette:
            save_dict['palette_optimizer_state_dict'] = palette_optimizer.state_dict()
        torch.save(save_dict, os.path.join(config.expdir, "checkpoints", f"{config.expname}_{epoch:06}"))

        #save image
        learned_img = utils.render_image(model, config)
        learned_img = Image.fromarray((learned_img*255).astype(np.uint8))
        # save image
        # TODO: inference, save image
        learned_img = (render_image(model, config)*255).astype(np.uint8)

        mse_dict["train_mse"].append(float(mse(learned_img, img)))
        ssim_dict["train_ssim"].append(float(ssim(learned_img, img, gaussian_weights=True, channel_axis=2, data_range=255)))

        learned_img = Image.fromarray(learned_img)
        learned_img.save(os.path.join(config.expdir, "reconstructions", f"{config.expname}_e{epoch:06}.png"))

        if config.palette:
            utils.show_palette(model, config)
        plot_metrics_curve(config, mse_dict, "mse")
        plot_metrics_curve(config, ssim_dict, "ssim")
        save_history(config, mse_dict, "mse")
        save_history(config, ssim_dict, "ssim")
        save_history(config, loss_dict, "loss")



def plot_metrics_curve(config, dict, type):


    f, ax = plt.subplots()
    ax.plot([x for x in range(len(dict['train_' + type]))], dict['train_' + type], color='b', label='train')
    ax.plot([x for x in range(len(dict['val_' + type]))], dict['val_' + type], color='r', label='val')
    f.suptitle(config.expname + " " + type)
    plt.savefig(os.path.join(config.expdir, type + "_curves.png"))
    plt.close()


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


def make_video(config):
    '''
    Takes the frames that have been rendered during different epochs
    and combines them into a video
    '''
    pass
    # TODO: combine current images into video and write to file


def setup(config):
    '''
    Sets up experiment directions
    Creates and loads model/optimizer
    Creates dataloader
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model = RadianceField2D(config).to(device)
    if last_checkpoint is not None:
        model.load_state_dict(torch.load(last_checkpoint, map_location=device)["model_state_dict"])
        print(f"Loading model from checkpoint: '{last_checkpoint}'")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    if last_checkpoint is not None:
        optimizer.load_state_dict(torch.load(last_checkpoint, map_location=device)["optimizer_state_dict"])

    # Initialize loss history
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

    # Make data loader
    dataset = ImageDataset(config)
    loader = DataLoader(dataset, batch_size=config.batchsize, shuffle=True, drop_last=True)
    return model, optimizer, loader


if __name__ == "__main__":
    config = parser.parse_args()
    if config.layer_norm:
        print("Training with layer norm")
    if config.batch_norm:
        print("Training with batch norm")

    model, loader, optimizer, palette_optimizer = utils.setup(config)
    # show_palette(model, config)
    train(config, model, loader, optimizer, palette_optimizer)

