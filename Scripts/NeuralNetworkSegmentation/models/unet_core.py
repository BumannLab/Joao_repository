import torch
from torch import nn

# NOTE: Default Unet uses padding = "valid", and at the end the resulting image is not the same size as the input image,
# to handle this they increase the image size through padding and the artificial border added has the mirror mode 

# For now to allow images of 128x128 and 64x64 to train, padding = "same" will be used. Also because this 
# allows for the result of the unet be the same size as the input different from the Unet defined in the first article

class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout_rate:float, in_3d: bool):
        super().__init__()

        bias = True if not use_batchnorm else False

        conv = nn.Conv3d if in_3d else nn.Conv2d
        bach_norm = nn.BatchNorm3d if in_3d else nn.BatchNorm2d
        dropout = nn.Dropout3d if in_3d else nn.Dropout2d
        
        self.convolution_operation = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3, padding="same", bias=bias),
            bach_norm(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),

            conv(out_channels, out_channels, kernel_size=3, padding="same", bias=bias),
            bach_norm(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
            dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity(),
        )

    def forward(self, x):
        x = self.convolution_operation(x)
        return x


class Encoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout_rate: float, in_3d: bool):
        super().__init__(in_channels, out_channels, use_batchnorm, dropout_rate, in_3d) 

        max_pool = nn.MaxPool3d if in_3d else nn.MaxPool2d
        
        self.max_pool = max_pool(kernel_size = 2, stride = 2)


    def forward(self, x):
        x = super().forward(x)
        x_down = self.max_pool(x)
        return x_down, x


class Decoder(ConvolutionBlock):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool, dropout_rate: float, in_3d: bool):
        super().__init__(in_channels, out_channels, use_batchnorm, dropout_rate, in_3d)

        bias = True if not use_batchnorm else False

        conv_transp = nn.ConvTranspose3d if in_3d else nn.ConvTranspose2d
        
        self.in_3d = in_3d
        self.t_convolution_operation = conv_transp(in_channels = in_channels, out_channels = out_channels, kernel_size = 2,
                                                           stride = 2, padding = 0, bias=bias)
    
    def forward(self, x, skip): 
        
        if self.in_3d:
            dim = 1 if x.ndim == 5 else 0
        else:
            dim = 1 if x.ndim == 4 else 0
        
        x = self.t_convolution_operation(x)

        x = torch.cat([x, skip], dim=dim)
        x = super().forward(x)
        return x
    

if __name__ == "__main__":
    print(ConvolutionBlock(1, 2, use_batchnorm=True))

    print(Encoder(1, 2, use_batchnorm=False))
    print(Decoder(1, 2, use_batchnorm=False))