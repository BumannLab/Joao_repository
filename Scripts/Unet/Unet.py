from torch import nn
from Unet_core import Encoder, Decoder, ConvolutionBlock

class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.enc_1 = Encoder(in_channels, 64)
        self.enc_2 = Encoder(64, 64*2)
        self.enc_3 = Encoder(64*2, 64*4)
        self.enc_4 = Encoder(64*4, 64*8)
        
        self.conv = ConvolutionBlock(64*8, 64*16)

        self.dec_4 = Decoder(64*16, 64*8)
        self.dec_3 = Decoder(64*8, 64*4)
        self.dec_2 = Decoder(64*4, 64*2)
        self.dec_1 = Decoder(64*2, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, skip_1 = self.enc_1(x)
        x, skip_2 = self.enc_2(x)
        x, skip_3 = self.enc_3(x)
        x, skip_4 = self.enc_4(x)

        x = self.conv(x)

        x = self.dec_4(x, skip_4)
        x = self.dec_3(x, skip_3)
        x = self.dec_2(x, skip_2)
        x = self.dec_1(x, skip_1)

        x = self.final_conv(x)

        return x