import torch, pdb
import torch.nn as nn
class CFD_CNN(nn.Module):
    def __init__(self):
        super(CFD_CNN, self).__init__()
        
        ### ENCODER
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=64,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_2 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_3 = nn.Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_4 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_5 = nn.Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_6 = nn.Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=2,
                                stride=2,
                                padding=0)
        
        self.conv_7 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=2,
                                stride=2,
                                padding=0)   
        
        self.conv_8 = nn.Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=2,
                                stride=2,
                                padding=0) 
        
        ## DECODER
        self.deconv_1 = nn.ConvTranspose2d(in_channels=512,
                                           out_channels=512,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_2 = nn.ConvTranspose2d(in_channels=512,
                                           out_channels=512,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_3 = nn.ConvTranspose2d(in_channels=512,
                                           out_channels=256,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_4 = nn.ConvTranspose2d(in_channels=256,
                                           out_channels=256,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_5 = nn.ConvTranspose2d(in_channels=256,
                                           out_channels=256,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_6 = nn.ConvTranspose2d(in_channels=256,
                                           out_channels=128,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_7 = nn.ConvTranspose2d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)
        
        self.deconv_8 = nn.ConvTranspose2d(in_channels=64,
                                           out_channels=6,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0)

        
        self.activation = nn.ReLU()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

    def forward(self, x):   ##256x256,1
        
        ## ENCODER
        c1 = self.conv_1(x) 
        c1 = self.bn1(c1)
        c1 = self.activation(c1)    ## 128x128,64
        
        c2 = self.conv_2(c1)
        c2 = self.bn2(c2)
        c2 = self.activation(c2)    ## 64x64, 128
        
        c3 = self.conv_3(c2)
        c3 = self.bn3(c3)
        a3 = self.activation(c3)    ## 32x32, 256
        
        c4 = self.conv_4(a3)
        c4 = self.bn3(c4)
        a4 = self.activation(c4)    ## 16x16, 256
        
        c5 = self.conv_5(a4)
        c5 = self.bn3(c5)
        a5 = self.activation(c5)    ## 8x8, 256
        
        c6 = self.conv_6(a5)
        c6 = self.bn4(c6)
        a6 = self.activation(c6)    ## 4x4, 512
        
        c7 = self.conv_7(a6)
        c7 = self.bn4(c7)
        a7 = self.activation(c7)    ## 2x2, 512
        
        c8 = self.conv_8(a7)        ## 1x1, 512     

        ## DECODER
        d1 = self.deconv_1(c8)  ## 2x2, 512
        d1 = self.activation(d1)
        
        d2 = self.deconv_2(d1)  ## 4x4, 512
        d2 = self.activation(d2)
        
        d3 = self.deconv_3(d2)  ## 8x8, 256
        d3 = self.activation(d3)
        
        d4 = self.deconv_4(d3)  ## 16x16, 256
        d4 = self.activation(d4)
        
        d5 = self.deconv_5(d4)  ## 32x32, 256
        d5 = self.activation(d5)
        
        d6 = self.deconv_6(d5)  ## 64x64, 128
        d6 = self.activation(d6)
        
        d7 = self.deconv_7(d6)  ## 128x128, 64
        d7 = self.activation(d7)
        
        d8 = self.deconv_8(d7)  ## 256x256, NC
        
        return d8

#device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
#net = CFD_CNN().to(device)
#summary(net)
