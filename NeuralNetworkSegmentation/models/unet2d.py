import torch
from torch import nn
from torchinfo import summary

from NeuralNetworkSegmentation.models.unet_core import ConvolutionBlock, Encoder, Decoder

class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, channels=64, num_blocks=4, use_batchnorm=False):
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        encoder_channels = [channels * (2 ** i) for i in range(num_blocks)]

        current_in_channels = in_channels
        for encoder_out_channels in encoder_channels:
            self.encoders.append(Encoder(current_in_channels, encoder_out_channels, use_batchnorm))
            current_in_channels = encoder_out_channels

        bottleneck_channels = channels * (2 ** num_blocks)
        self.conv = ConvolutionBlock(current_in_channels, bottleneck_channels, use_batchnorm)

        current_decoder_in_channels = bottleneck_channels
        for decoder_out_channels in reversed(encoder_channels):
            self.decoders.append(Decoder(current_decoder_in_channels, decoder_out_channels, use_batchnorm))
            current_decoder_in_channels = decoder_out_channels

        self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

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


def _run_model_checks():
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if device.type == "cuda":
    #     torch.cuda.empty_cache()
    #     torch.cuda.ipc_collect()

    input_shape = (16, 2, 64, 64)

    test_cases = [
        ("UNet2D", UNet2D(in_channels=2, out_channels=2, channels=64, num_blocks=3, use_batchnorm=True).to(device)),
    ]

    for model_name, model in test_cases:
        model.eval()
        print(f"=== {model_name} ===")
        print(model)
        print(summary(model, input_size=input_shape, device=device, verbose=0))


if __name__ == "__main__":
    _run_model_checks()