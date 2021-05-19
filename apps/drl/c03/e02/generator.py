#
import torch
import torch.nn as nn

class Generator(nn.Module):
    LATENT_VECTOR_SIZE = 100
    GENER_FILTERS = 64

    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=Generator.LATENT_VECTOR_SIZE, out_channels=Generator.GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(Generator.GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=Generator.GENER_FILTERS * 8, out_channels=Generator.GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(Generator.GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=Generator.GENER_FILTERS * 4, out_channels=Generator.GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(Generator.GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=Generator.GENER_FILTERS * 2, out_channels=Generator.GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(Generator.GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=Generator.GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)