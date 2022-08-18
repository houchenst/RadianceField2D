import numpy as np
import torch
import os

from tqdm import tqdm
from PIL import Image

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

    # if config.n_tuning_layers >=0:
    #     for i, param in enumerate(model.parameters()):
    #         2*config.n_tuning_layers
    #         param.requires_grad=False
    if len(config.tunable_layers) >0:
        for name, param in model.named_parameters():
            if name not in config.tunable_layers:
                param.requires_grad = False

    for e in range(config.epochs):
        model.train()
        print(f"----------  EPOCH {e+1}/{config.epochs}  ----------")
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
        loss_dict = {}
        loss_dict["train_loss"] = [float(np.mean(train_losses))]
        loss_dict["val_loss"] = []
        save(model, optimizer, e, loss_dict, config, palette_optimizer=palette_optimizer)

        


def save(model, optimizer, epoch, loss_dict, config, palette_optimizer=None):
    '''
    Saves loss curve, checkpoint of the model, and the current image reconstruction
    Only saves checkpoint and images during certain epochs
    '''

    print(f"Epoch training loss: {loss_dict['train_loss'][0]:.4f}")
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
        learned_img.save(os.path.join(config.expdir, "reconstructions", f"{config.expname}_e{epoch:06}.png"))

        if config.palette:
            utils.show_palette(model, config)


    

if __name__ == "__main__":
    config = parser.parse_args()
    model, loader, optimizer, palette_optimizer = utils.setup(config)
    # show_palette(model, config)
    train(config, model, loader, optimizer, palette_optimizer)
    
