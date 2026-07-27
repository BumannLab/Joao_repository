from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import Convolution
from monai.networks.layers.factories import Pool, Conv

# NOTE: Default Unet uses padding = "valid", and at the end the resulting image is not the same size as the input image,
# to handle this they increase the image size through padding and the artificial border added has the mirror mode 


class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NAD",
        act: tuple | str | None = "RELU",
        norm: tuple | str | None = "BATCH",
        dropout: tuple | str | float | None = None,
        bias: bool = True,
        padding: Sequence[int] | int | None = None,
    ):
        super().__init__()

        self.convblock = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=spatial_dims, # Perhaps add this as input var
                dilation=1,
                groups=1,
                bias=bias,
                padding=padding,
                output_padding=None,
            ),
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=kernel_size,
                adn_ordering=adn_ordering,
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=spatial_dims, # Perhaps add this as input var
                dilation=1,
                groups=1,
                bias=bias,
                padding=padding,
                output_padding=None,
            ),
        )

    def forward(self, x):
        x = self.convblock(x)
        return x


class Encoder(ConvBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NAD",
        act: tuple | str | None = "RELU",
        norm: tuple | str | None = "BATCH",
        bias = True,
        dropout: tuple | str | float | None = None,
        padding: Sequence[int] | int | None = None,
    ):
        
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            adn_ordering,
            act,
            norm,
            dropout,
            bias,
            padding,
        ) 

        max_pool = Pool[Pool.MAX, spatial_dims]
        
        self.max_pool = max_pool(kernel_size = 2, stride = 2)

    def forward(self, x):
        x_skip = super().forward(x)
        x = self.max_pool(x_skip)
        return x, x_skip


class Decoder(ConvBlock):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Sequence[int] | int = 1,
        kernel_size: Sequence[int] | int = 3,
        adn_ordering: str = "NAD",
        act: tuple | str | None = "RELU",
        norm: tuple | str | None = "BATCH",
        bias = True,
        dropout: tuple | str | float | None = None,
        padding: Sequence[int] | int | None = None,
    ):
        
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            strides,
            kernel_size,
            adn_ordering,
            act,
            norm,
            dropout,
            bias,
            padding,
        )
        
        conv_trans = Conv[Conv.CONVTRANS, spatial_dims]
        
        self.conv_trans = conv_trans(in_channels = in_channels, 
                                    out_channels = out_channels,
                                    kernel_size = 2,
                                   stride = 2,
                                   padding = 0, 
                                   bias=bias)
    
    def forward(self, x, skip): 
                
        x = self.conv_trans(x)
        x = torch.cat([skip, x], dim=1)
        x = super().forward(x)
        
        return x
    

if __name__ ==  "__main__":
    print(Encoder(3, 1, 2, bias=True, dropout=0.5))
    
    print(Decoder(3, 1, 2, norm = "BATCH", dropout=0.5))
