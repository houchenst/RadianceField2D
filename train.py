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

    mse_dict = utils.load_history(config, "mse")
    loss_dict = utils.load_history(config, "loss")
    ssim_dict = utils.load_history(config, "ssim")

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

        # save image
        # TODO: inference, save image
        learned_img = (utils.render_image(model, config)*255).astype(np.uint8)

        mse_dict["train_mse"].append(float(mse(learned_img, img)))
        ssim_dict["train_ssim"].append(float(ssim(learned_img, img, gaussian_weights=True, channel_axis=2, data_range=255)))

        learned_img = Image.fromarray(learned_img)
        learned_img.save(os.path.join(config.expdir, "reconstructions", f"{config.expname}_e{epoch:06}.png"))

        if config.palette:
            utils.show_palette(model, config)
        utils.plot_metrics_curve(config, mse_dict, "mse")
        utils.plot_metrics_curve(config, ssim_dict, "ssim")
        utils.save_history(config, mse_dict, "mse")
        utils.save_history(config, ssim_dict, "ssim")
        utils.save_history(config, loss_dict, "loss")

if __name__ == "__main__":
    config = parser.parse_args()
    if config.layer_norm:
        print("Training with layer norm")
    if config.batch_norm:
        print("Training with batch norm")

    model, loader, optimizer, palette_optimizer = utils.setup(config)
    # show_palette(model, config)
    train(config, model, loader, optimizer, palette_optimizer)

