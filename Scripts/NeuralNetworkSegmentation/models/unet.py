from torch import nn

from torchinfo import summary


from Scripts.NeuralNetworkSegmentation.utils import get_class
from Scripts.NeuralNetworkSegmentation.models.unet_core import ConvolutionBlock, Encoder, Decoder

#TODO Add dropout

class UNet(nn.Module): 
    def __init__(self, 
                 in_channels=1,
                 out_channels=2,
                 features=64,
                 num_blocks=4,
                 use_batchnorm=False,
                 dropout_rate=0.0,
                 final_layer_type = "sigmoid",
                 is_3d=True):
        
        super().__init__()

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        encoder_channels = [features * (2 ** i) for i in range(num_blocks)]

        current_in_channels = in_channels
        for encoder_out_channels in encoder_channels:
            self.encoders.append(Encoder(current_in_channels, encoder_out_channels, use_batchnorm, is_3d))
            current_in_channels = encoder_out_channels

        bottleneck_channels = features * (2 ** num_blocks)
        self.conv = ConvolutionBlock(current_in_channels, bottleneck_channels, use_batchnorm, is_3d)

        current_decoder_in_channels = bottleneck_channels
        for decoder_out_channels in reversed(encoder_channels):
            self.decoders.append(Decoder(current_decoder_in_channels, decoder_out_channels, use_batchnorm, is_3d))
            current_decoder_in_channels = decoder_out_channels

        conv = nn.Conv3d if is_3d else nn.Conv2d

        self.final_conv = conv(features, out_channels, kernel_size=1)

        self.final_layer_type = nn.Softmax(dim = 1) if final_layer_type == "softmax" else nn.Sigmoid()

        self.is_3d = is_3d
        

    def forward(self, x):
        skip_connections = []

        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        x = self.conv(x)

        for decoder, skip in zip(self.decoders, reversed(skip_connections)):
            x = decoder(x, skip)

        x = self.final_conv(x)

        out = self.final_layer_type(x)

        return out, x # probabilities, logits


class UNet2D(UNet):
    def __init__(self, in_channels = 1, out_channels = 2, features = 64, num_blocks = 4, use_batchnorm = False, **kwargs):
        super().__init__(in_channels, out_channels, features, num_blocks, use_batchnorm, is_3d = False)


class UNet3D(UNet):
    def __init__(self, in_channels = 1, out_channels = 2, features = 64, num_blocks = 4, use_batchnorm = False, **kwargs):
        super().__init__(in_channels, out_channels, features, num_blocks, use_batchnorm, is_3d = True)


def get_model(model_config):
    model_class = get_class(model_config["name"], modules=["Scripts.NeuralNetworkSegmentation.models.unet"])
    return model_class(**model_config)


def is_model_2d(model):
    
    if model.is_3d == False:
        return True
    
    return False
        

if __name__ == "__main__":

    model = UNet(in_channels=1, out_channels=2, features=32, num_blocks=3, 
                 use_batchnorm=True, final_layer_type="softmax", is_3d=True)
    
    summary(model, (1, 1, 24, 128, 128))
