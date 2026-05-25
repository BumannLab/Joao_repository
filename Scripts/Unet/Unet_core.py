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
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm = False):
        super().__init__()

        bias = True if not use_batchnorm else False
        
        self.convolution_operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convolution_operation(x)
        return x


class Encoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm = False):
        super().__init__(in_channels, out_channels, use_batchnorm) 

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        x = super().forward(x)
        x_down = self.max_pool(x)
        return x_down, x


class Decoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm = False):
        super().__init__(in_channels, out_channels, use_batchnorm)

        bias = True if not use_batchnorm else False

        self.t_convolution_operation = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2,
                                                           stride = 2, padding = 0, bias=bias)

    def forward(self, x, skip): 
        dim = 1 if x.ndim == 4 else 0
        x = self.t_convolution_operation(x)
        x = torch.cat([x, skip], dim=dim)
        x = super().forward(x)
        return x
    

if __name__ == "__main__":
    print(ConvolutionBlock(1, 2, use_batchnorm=True))
    print(Encoder(1, 2, use_batchnorm=False))
    print(Decoder(1, 2, use_batchnorm=False))