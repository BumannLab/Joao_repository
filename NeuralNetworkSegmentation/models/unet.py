from torch import nn

from NeuralNetworkSegmentation.models.unet_core import ConvolutionBlock, Encoder, Decoder

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels=64, num_blocks=4, use_batchnorm=False, is_3d=True):
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

