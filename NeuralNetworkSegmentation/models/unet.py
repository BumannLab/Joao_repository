from torch import nn

from utils.utils import get_class
from NeuralNetworkSegmentation.models.unet_core import ConvolutionBlock, Encoder, Decoder

#TODO Add dropout
#TODO Replace channels by feature map (variable name)

class UNet(nn.Module): 
    def __init__(self, 
                 in_channels=1,
                 out_channels=2,
                 channels=64,
                 num_blocks=4,
                 use_batchnorm=False,
                 is_3d=True):
        
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        encoder_channels = [channels * (2 ** i) for i in range(num_blocks)]

        current_in_channels = in_channels
        for encoder_out_channels in encoder_channels:
            self.encoders.append(Encoder(current_in_channels, encoder_out_channels, use_batchnorm, is_3d))
            current_in_channels = encoder_out_channels

        bottleneck_channels = channels * (2 ** num_blocks)
        self.conv = ConvolutionBlock(current_in_channels, bottleneck_channels, use_batchnorm, is_3d)

        current_decoder_in_channels = bottleneck_channels
        for decoder_out_channels in reversed(encoder_channels):
            self.decoders.append(Decoder(current_decoder_in_channels, decoder_out_channels, use_batchnorm, is_3d))
            current_decoder_in_channels = decoder_out_channels

        conv = nn.Conv3d if is_3d else nn.Conv2d

        self.final_conv = conv(channels, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.conv(x)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        x = self.final_conv(x)

        return x


class UNet2D(UNet):
    def __init__(self, in_channels = 1, out_channels = 2, channels = 64, num_blocks = 4, use_batchnorm = False, **kwargs):
        super().__init__(in_channels, out_channels, channels, num_blocks, use_batchnorm, is_3d = False)


class UNet3D(UNet):
    def __init__(self, in_channels = 1, out_channels = 2, channels = 64, num_blocks = 4, use_batchnorm = False, **kwargs):
        super().__init__(in_channels, out_channels, channels, num_blocks, use_batchnorm, is_3d = True)


def get_model(model_config):
    model_class = get_class(model_config["name"], modules=["NeuralNetworkSegmentation.models.unet"])
    return model_class(**model_config)