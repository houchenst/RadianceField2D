import torch
import torch.nn as nn

class ColorPickerLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.color_SM = nn.Softmax(dim=1)
        self.instance_SM = nn.Softmax(dim=0)
        self.SMs = [self.instance_SM, self.color_SM]

    def forward(self, targets, color_picks, model, dim=0):
        '''
        targets     - Radiance targets
        color_picks - The softmax values of the color selection
        model       - The color palette model, the palette tensor from the model is used in the loss
        '''
        color_picks = self.SMs[dim](color_picks)
        # color_picks = self.color_SM(color_picks)
        b = targets.shape[0]
        p = color_picks.shape[-1]
        targets = torch.tile(targets.unsqueeze(1), (1,p,1))
        palette = torch.tile(model.palette.palette.unsqueeze(0), (b,1,1))
        color_squared_errors = torch.sum(torch.square(targets-palette), dim=2)
        # print("COLOR SQUARED ERRORS")
        # print(color_squared_errors)
        # print("\n\n\n")
        losses = torch.sum(color_picks * color_squared_errors, axis=1)
        mean_loss = torch.mean(losses)
        return mean_loss
