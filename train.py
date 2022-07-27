import numpy as np
import torch
from torch.utils.data import DataLoader
from torchgeometry.losses import SSIM
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


def train(model, dataloader, optimizer, config):
    '''
    Trains model
    '''
    # TODO training loop
    loss_fn = torch.nn.MSELoss()
    model = model.to(config.device)

    mse_dict = {"train_mse": [], "val_mse": []}
    ssim_dict = {"train_ssim": [], "val_ssim": []}
    img = Image.open(config.imagepath)
    img = np.array(img)
    if config.drop_alpha:
        img = img[:, :, :-1]
    img = torch.tensor(np.expand_dims(img, 0)).to(torch.float32)

    for e in range(config.epochs):
        model.train()
        print(f"----------  EPOCH {e + 1}/{config.epochs}  ----------")
        train_losses = []
        for batch in tqdm(dataloader):
            coords, radiance = batch
            coords = coords.to(config.device)
            radiance = radiance.to(config.device)
            optimizer.zero_grad()
            pred_radiance = model(coords)
            loss = loss_fn(pred_radiance, radiance)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().numpy())
        loss_dict = {}
        loss_dict["train_loss"] = [float(np.mean(train_losses))]
        loss_dict["val_loss"] = []

        save(model, optimizer, e, loss_dict, mse_dict, ssim_dict, config, img)


def load_history(config, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'r') as f:
        loss_history = json.loads(f.read())
    return loss_history


def save_history(config, history, type):
    with open(os.path.join(config.expdir, type + "_history.txt"), 'w') as f:
        f.write(json.dumps(history))


def save(model, optimizer, epoch, loss_dict, mse_dict, ssim_dict, config, img):
    '''
    Saves loss curve, checkpoint of the model, and the current image reconstruction
    Only saves checkpoint and images during certain epochs
    '''

    print(f"Epoch training loss: {loss_dict['train_loss'][0]:.4f}")
    plot_loss_curve(config, loss_dict)

    if epoch % config.save_frequency == 0:
        # save checkpoint
        save_dict = {}
        save_dict['model_state_dict'] = model.state_dict()
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(save_dict, os.path.join(config.expdir, "checkpoints", f"{config.expname}_{epoch:06}"))

        # save image
        # TODO: inference, save image
        learned_img = (render_image(model, config)*255).astype(np.uint8)

        input_img = torch.tensor(np.expand_dims(learned_img, 0)).to(torch.float32)
        mse = torch.nn.MSELoss(reduction="mean")
        mse_dict["train_mse"].append(float(mse(input_img, img)))
        ssim = SSIM(11, reduction="mean")
        ssim_dict["train_ssim"].append(float(ssim(input_img, img)))

        learned_img = Image.fromarray(learned_img)
        learned_img.save(os.path.join(config.expdir, "reconstructions", f"{config.expname}_e{epoch:06}.png"))
        # np.save(os.path.join(config.expdir, "reconstructions", f"{config.expname}_e{epoch:06}.npy"), learned_img)

        plot_metrics_curve(config, mse_dict, "mse")
        plot_metrics_curve(config, ssim_dict, "ssim")
        save_history(config, mse_dict, "mse")
        save_history(config, ssim_dict, "ssim")


def render_image(model, config):
    '''
    Renders the image from the radiance field
    '''
    model.eval()
    coords = []
    for i in range(config.ydim):
        for j in range(config.xdim):
            coords.append(torch.tensor(positional_encoding(
                pixel_to_continuous_coordinate((config.ydim, config.xdim), i, j, x_offset=0.5, y_offset=0.5), config)))
    outputs = []
    for i in range(math.ceil(len(coords) / config.batchsize)):
        batch_input = torch.stack(coords[i * config.batchsize:(i + 1) * config.batchsize], dim=0).to(config.device)
        outputs.append(model(batch_input))
    outputs = torch.vstack(outputs).detach().cpu().numpy()
    outputs = outputs.reshape((config.ydim, config.xdim, 3))
    outputs = np.clip(outputs, 0., 1.)
    return outputs


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
    loss_history = load_history(config, "loss")
    loss_history['train_loss'] += loss_dict['train_loss']
    loss_history['val_loss'] += loss_dict['val_loss']
    save_history(config, loss_history, "loss")
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
    model, optimizer, loader = setup(config)
    train(model, loader, optimizer, config)
