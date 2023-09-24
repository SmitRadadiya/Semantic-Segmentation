import torch
import torch.nn as nn
import numpy as np
import pickle
import torchvision.transforms.functional as TF


class Conv2Dblock(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super(Conv2Dblock, self).__init__()

        self.con2d1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.relu  = nn.ReLU()
        self.con2d2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.batchnorm = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        x = self.con2d1(x)
        # x = self.batchnorm(x)
        x = self.relu(x)
        x = self.con2d2(x)
        # x = self.batchnorm(x)
        x = self.relu(x)

        return x
    

class EncoderBlock(nn.Module):

    def __init__(self, in_channels) -> None:
        super(EncoderBlock, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.channels = np.array([64, 128, 256, 512])
        self.encoder = nn.ModuleList()

        for out_channel in self.channels:
            self.encoder.append(Conv2Dblock(in_channels, out_channel))
            in_channels = out_channel


    def forward(self, x):
        resudials = []
        # resudials = np.zeros((3,3))

        for encod in self.encoder:
            x = encod(x)
            resudials.append(x)
            x = self.pool(x)

        return x, resudials
    


class DecoderBlock(nn.Module):

    def __init__(self, out_channel) -> None:
        super(DecoderBlock, self).__init__()

        self.out_channel = out_channel
        self.channels = np.array([512, 256, 128, 64])  #fix chnnels as given in paper.
        self.decoder = nn.ModuleList()

        for channel in self.channels:
            self.decoder.append(nn.ConvTranspose2d(channel*2, channel, 2, 2))
            self.decoder.append(Conv2Dblock(channel*2, channel))

        self.out_put = nn.Conv2d(self.channels[-1], self.out_channel, 1)
        
    def forward(self, x, resudials):
        resudials = resudials[::-1]  #reverse skip connections

        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_conn = resudials[i//2]

            if x.shape != skip_conn.shape:
                x = TF.resize(x, skip_conn.shape[2:])

            concat_skip = torch.cat((skip_conn, x), dim=1)
            x = self.decoder[i+1](concat_skip)

        x = self.out_put(x)

        return x
    

class Unet(nn.Module):

    def __init__(self, in_challen, out_channel) -> None:
        super(Unet, self).__init__()

        self.encoder = EncoderBlock(in_challen)
        self.connection = Conv2Dblock(512, 1024)
        self.decoder = DecoderBlock(out_channel)


    def forward(self, x):
        
        x, resudials = self.encoder(x)
        x = self.connection(x)
        x = self.decoder(x, resudials)
        return x
        

    # def save(self, path):
    #     """Save the model to the given path.

    #     Args:
    #         path (str): Location to save the model.
    #     """
    #     with open(path, 'wb') as f:
    #         pickle.dump((self.L, self.Z0), f)
    
    # def load(self, path):
    #     """Load the model from the given path.

    #     Args:
    #         path (str): Location to load the model.
    #     """
    #     with open(path, 'rb') as f:
    #          = pickle.load(f)