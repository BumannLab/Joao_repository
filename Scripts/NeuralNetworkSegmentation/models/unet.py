import warnings
from collections.abc import Sequence


from torch import nn
from torchinfo import summary
from monai.networks.layers.factories import Conv


from Scripts.NeuralNetworkSegmentation.utils import get_class
from Scripts.NeuralNetworkSegmentation.models.unet_core import ConvBlock, Encoder, Decoder


class AbstractUnet(nn.Module): 
    def __init__(
        self,
        in_channels,
        out_channels,
        f_maps = [32, 64, 128, 256],
        conv_kernel_size: Sequence[int] | int = 3,
        conv_strides: Sequence[int] | int = 1,
        layer_order: str = "NAD",
        act: tuple | str | None = "RELU",
        norm: tuple | str | None = "BATCH",
        dropout: tuple | str | float | None = None,
        conv_padding: Sequence[int] | int | None = None,
        is_3d = False,
    ):
        if norm is not None:
            bias = False
        else:
            bias = True

        spatial_dims = 3 if is_3d else 2

        super().__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        current_in_channels = in_channels
        for encoder_out_channels in f_maps:
            self.encoders.append(Encoder(
                spatial_dims=spatial_dims,
                in_channels=current_in_channels,
                out_channels=encoder_out_channels,
                strides=conv_strides,
                kernel_size=conv_kernel_size,
                adn_ordering=layer_order,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                padding=conv_padding,
                ))
            
            current_in_channels = encoder_out_channels

        bottleneck_channels = current_in_channels * 2
        self.conv = ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=current_in_channels,
                out_channels=bottleneck_channels,
                strides=conv_strides,
                kernel_size=conv_kernel_size,
                adn_ordering=layer_order,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                padding=conv_padding,
                )

        current_decoder_in_channels = bottleneck_channels
        for decoder_out_channels in reversed(f_maps):
            self.decoders.append(Decoder(
                spatial_dims=spatial_dims,
                in_channels=current_decoder_in_channels,
                out_channels=decoder_out_channels,
                strides=conv_strides,
                kernel_size=conv_kernel_size,
                adn_ordering=layer_order,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
                padding=conv_padding,
                ))
            
            current_decoder_in_channels = decoder_out_channels

        conv = Conv[Conv.CONV, spatial_dims]

        self.final_conv = conv(current_decoder_in_channels, out_channels, kernel_size=1)
        self.final_layer = nn.Sigmoid()
        self.is_3d = is_3d
        
    def _forward(self, x):
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.conv(x)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        x = self.final_conv(x)

        out = self.final_layer(x)
        
        return out, x # probabilities, logits
    
    def forward(self, x, return_logits = False):

        out, logits = self._forward(x)

        if return_logits:
            return out, logits
        
        return out


class UNet2D(AbstractUnet):

    def __init__(
        self,
        in_channels = 1,
        out_channels = 2,
        f_maps = [32, 64, 128, 256],
        conv_kernel_size = 3,
        conv_strides = 1,
        layer_order = "NAD",
        act = "RELU",
        norm = "BATCH",
        dropout = None,
        conv_padding = None,
        **kwargs
    ):
        
        super().__init__(
            in_channels,
            out_channels,
            f_maps,
            conv_kernel_size,
            conv_strides,
            layer_order,
            act,
            norm,
            dropout,
            conv_padding,
            is_3d = False,

        )


class UNet3D(AbstractUnet):

    def __init__(
        self,
        in_channels = 1,
        out_channels = 2,
        f_maps = [32, 64, 128, 256],
        conv_kernel_size = 3,
        conv_strides = 1,
        layer_order = "NAD",
        act = "RELU",
        norm = "BATCH",
        dropout = None,
        conv_padding = None,
        **kwargs
    ):
        
        super().__init__(
            in_channels,
            out_channels,
            f_maps,
            conv_kernel_size,
            conv_strides,
            layer_order,
            act,
            norm,
            dropout,
            conv_padding,
            is_3d = True,
        )

def get_model(model_config):
    model_class = get_class(model_config["name"], modules=["Scripts.NeuralNetworkSegmentation.models.unet"])
    return model_class(**model_config)


def is_model_2d(model):
    
    if model.is_3d == False:
        return True
    
    return False
        

if __name__ == "__main__":

    model = UNet3D(
        in_channels = 1,
        out_channels = 2,
        f_maps = [32, 64, 128],
        conv_kernel_size = 3,
        conv_strides = 1,
        layer_order = "NAD",
        act = "RELU",
        norm = "BATCH",
        dropout = None,
        conv_padding = None,
    )
    
    summary(model, (1, 1, 24, 128, 128), depth=20)
