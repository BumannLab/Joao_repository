import torch
from torch import nn

# Important: 
# Default Unet uses padding = "valid", however to be able to use this padding size only images of size
# superior to 256 can run through the network.
# For now to allow images of 128x128 and 64x64 to train, padding = "same" will be used. Also because this 
# allows for the result of the unet be the same size as the input different from the Unet defined in the first article
# TODO: Check if all the core blocks are right, yes for now
# TODO: Check if the end of unet should be sigmoid or softmax? softmax
# TODO: in_channels -> number of channels in image

class ConvolutionBlock(nn.Module):
    # TODO: Add batch normalization?
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.convolution_operation = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                      kernel_size = 3, padding = "same"),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels,
                      kernel_size = 3, padding = "same"),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.convolution_operation(x)
        return x


class Encoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels) 

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = super().forward(x)
        x_down = self.max_pool(x)
        return x_down, x


class Decoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(in_channels, out_channels)

        self.t_convolution_operation = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2,
                                                           stride = 2, padding = 0)

    def forward(self, x, skip): 
        x = self.t_convolution_operation(x)
        x_plus_skip = torch.cat([x, skip], dim=1)
        x_plus_skip = super().forward(x_plus_skip)
        return x_plus_skip