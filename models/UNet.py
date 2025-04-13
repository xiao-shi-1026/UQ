import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3):
        super(UNet, self).__init__()
        self.enc1 = self. basicblock(in_channels, 64)
        self.enc2 = self. basicblock(64, 128)
        self.enc3 = self. basicblock(128, 256)
        self.enc4 = self. basicblock(256, 512)

        self.bottleneck = self.basicblock(512, 1024)

        self.upconv1 = self.upsample(1024, 512)
        self.dec1 = self. basicblock(1024, 512)
        self.upconv2 = self.upsample(512, 256)
        self.dec2 = self. basicblock(512, 256)
        self.upconv3 = self.upsample(256, 128)
        self.dec3 = self. basicblock(256, 128)
        self.upconv4 = self.upsample(128, 64)
        self.dec4 = self. basicblock(128, 64)
        
        self.outlayer = nn.Conv2d(64, out_channels, kernel_size=1)

    def basicblock(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block
    
    def upsample(self, in_channels, out_channels):
        block = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        return block

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = F.max_pool2d(enc1, kernel_size=2)

        enc2 = self.enc2(enc1_pool)
        enc2_pool = F.max_pool2d(enc2, kernel_size=2)

        enc3 = self.enc3(enc2_pool)
        enc3_pool = F.max_pool2d(enc3, kernel_size=2)

        enc4 = self.enc4(enc3_pool)
        enc4_pool = F.max_pool2d(enc4, kernel_size=2)

        bottleneck = self.bottleneck(enc4_pool)

        dec1_up = self.upconv1(bottleneck)
        dec1 = self.dec1(torch.cat((dec1_up, enc4), dim = 1))

        dec2_up = self.upconv2(dec1)
        dec2 = self.dec2(torch.cat((dec2_up, enc3), dim = 1))

        dec3_up = self.upconv3(dec2)
        dec3 = self.dec3(torch.cat((dec3_up, enc2), dim = 1))

        dec4_up = self.upconv4(dec3)
        dec4 = self.dec4(torch.cat((dec4_up, enc1), dim = 1))

        out = self.outlayer(dec4)

        return out

if __name__ == "__main__":

    model = UNet(in_channels=3, out_channels=3)
    print(model)

    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print("Output shape:", output.shape)  