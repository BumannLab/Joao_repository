import torch
from torch import nn
from torchinfo import summary
from Scripts.Unet.Unet_core import Encoder, Decoder, ConvolutionBlock

# class UNet(nn.Module):
#     def __init__(self, in_channels, out_channels, channels = 64, use_batchnorm = False):
#         super().__init__()

#         self.enc_1 = Encoder(in_channels, channels, use_batchnorm)
#         self.enc_2 = Encoder(channels, channels*2, use_batchnorm)
#         self.enc_3 = Encoder(channels*2, channels*4, use_batchnorm)
#         self.enc_4 = Encoder(channels*4, channels*8, use_batchnorm)
        
#         self.conv = ConvolutionBlock(channels*8, channels*16, use_batchnorm)

#         self.dec_4 = Decoder(channels*16, channels*8, use_batchnorm)
#         self.dec_3 = Decoder(channels*8, channels*4, use_batchnorm)
#         self.dec_2 = Decoder(channels*4, channels*2, use_batchnorm)
#         self.dec_1 = Decoder(channels*2, channels, use_batchnorm)
        
#         self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

#     def forward(self, x):
#         x, skip_1 = self.enc_1(x)
#         x, skip_2 = self.enc_2(x)
#         x, skip_3 = self.enc_3(x)
#         x, skip_4 = self.enc_4(x)

#         x = self.conv(x)

#         x = self.dec_4(x, skip_4)
#         x = self.dec_3(x, skip_3)
#         x = self.dec_2(x, skip_2)
#         x = self.dec_1(x, skip_1)

#         x = self.final_conv(x)

#         return x
    
# class UNet2D_3blocks(nn.Module):
#     def __init__(self, in_channels, out_channels, channels = 64, use_batchnorm = False):
#         super().__init__()

#         self.enc_1 = Encoder(in_channels, channels, use_batchnorm)
#         self.enc_2 = Encoder(channels, channels*2, use_batchnorm)
#         self.enc_3 = Encoder(channels*2, channels*4, use_batchnorm)
        
#         self.conv = ConvolutionBlock(channels*4, channels*8, use_batchnorm)

#         self.dec_3 = Decoder(channels*8, channels*4, use_batchnorm)
#         self.dec_2 = Decoder(channels*4, channels*2, use_batchnorm)
#         self.dec_1 = Decoder(channels*2, channels, use_batchnorm)
        
#         self.final_conv = nn.Conv2d(channels, out_channels, kernel_size=1)

#     def forward(self, x):

#         x, skip_1 = self.enc_1(x)
#         x, skip_2 = self.enc_2(x)
#         x, skip_3 = self.enc_3(x)

#         x = self.conv(x)

#         x = self.dec_3(x, skip_3)
#         x = self.dec_2(x, skip_2)
#         x = self.dec_1(x, skip_1)

#         x = self.final_conv(x)

#         return x


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, channels = 64, num_blocks = 4, use_batchnorm = False):
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    input_shape = (64, 4, 128, 128)

    test_cases = [
        ("UNet2D", UNet2D(in_channels=4, out_channels=2, channels=64, num_blocks=4, use_batchnorm=True).to(device)),
        # ("UNet2D_3blocks", UNet2D_3blocks(in_channels=4, out_channels=2, channels=64, use_batchnorm=True).to(device)),
        # ("UNet_variable_blocks", UNet_variable_blocks(in_channels=4, out_channels=2, channels=64, num_blocks=4, use_batchnorm=True).to(device)),
    ]

    for model_name, model in test_cases:
        model.eval()
        print(f"=== {model_name} ===")
        print(model)
        print(summary(model, input_size=input_shape, device=device, verbose=0))


if __name__ == "__main__":
    _run_model_checks()
