import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()

        self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class unet(nn.Module):
    def __init__(self, Cin, Cout):
        super(unet, self).__init__()
        self.inc = inconv(Cin, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, Cout)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class unet_FS2LF_v1(nn.Module):
    def __init__(self,nF=None,nu=None,nv=None):
        super(unet_FS2LF_v1,self).__init__()
        self.nu,self.nv = nu,nv
        self.unet = unet(nF,nu*nv)
    def forward(self,FS_rgb):
        """
        Foward passing each color channel of the FS to get reconLF using single color Unet, treating the nF dimension of FS as the color channel in the unet.
        Input:
        FS_rgb: FS of shape B,C,nF,H,W
        Output；
        LF: B,C,nv,nu,H,W
        """
        B,C,_,H,W = FS_rgb.shape
        FS_r,FS_g,FS_b = FS_rgb[:,0],FS_rgb[:,1],FS_rgb[:,2]
        LF_r,LF_g,LF_b = self.unet(FS_r),self.unet(FS_g),self.unet(FS_b) # LF_r has shape B,nu*nv,H,W
        LF_rgb = torch.stack([torch.unsqueeze(LF_r, 1),torch.unsqueeze(LF_g, 1),torch.unsqueeze(LF_b, 1)],dim = 1) #LF_rgb: B,C,nu*nv,H,W
        LF_rgb = LF_rgb.view(B,C,self.nv,self.nu,H,W) #LF_rgb: B,C,nv,nu,H,W
        return LF_rgb
    
class unet_FS2LF_v2(nn.Module):
    def __init__(self,nF=None,nu=None,nv=None):
        super(unet_FS2LF_v2,self).__init__()
        self.nu,self.nv = nu,nv
        self.unet_r = unet(nF,nu*nv)
        self.unet_g = unet(nF,nu*nv)
        self.unet_b = unet(nF,nu*nv)
    def forward(self,FS_rgb):
        """
        Foward passing each color channel of the FS to get reconLF using three separate single color Unet, treating the nF dimension of FS as the color channel in the unet.
        Input:
        FS_rgb: FS of shape B,C,nF,H,W
        Output；
        LF: B,C,nv,nu,H,W
        """
        B,C,_,H,W = FS_rgb.shape
        FS_r,FS_g,FS_b = FS_rgb[:,0],FS_rgb[:,1],FS_rgb[:,2]
        LF_r,LF_g,LF_b = self.unet_r(FS_r),self.unet_g(FS_g),self.unet_b(FS_b) # LF_r has shape B,nu*nv,H,W
        LF_rgb = torch.stack([torch.unsqueeze(LF_r, 1),torch.unsqueeze(LF_g, 1),torch.unsqueeze(LF_b, 1)],dim = 1) #LF_rgb: B,C,nu*nv,H,W
        LF_rgb = LF_rgb.view(B,C,self.nv,self.nu,H,W) #LF_rgb: B,C,nv,nu,H,W
        return LF_rgb
    
class unet_FS2LF_v3(nn.Module):
    def __init__(self,nF=None,nu=None,nv=None,C=3):
        super(unet_FS2LF_v3,self).__init__()
        self.nu,self.nv = nu,nv
        self.unet = unet(C*nF,C*nu*nv)

    def forward(self,FS_rgb):
        """
        Foward passing  FS to get reconLF using Unet, by merging the nF dimension of FS and the color dimension into the color channel in the unet.
        Input:
        FS_rgb: FS of shape B,C,nF,H,W
        Output；
        LF: B,C,nv,nu,H,W
        """
        B,C,nF,H,W = FS_rgb.shape
        FS_rgb = FS_rgb.view(B,C*nF,H,W)
        LF_rgb = self.unet(FS_rgb) #LF_rgb: B,C*nu*nv,H,W
        LF_rgb = LF_rgb.view(B,C,self.nv,self.nu,H,W) #LF_rgb: B,C,nv,nu,H,W
        return LF_rgb        
    
# case: 3D conv case 

#to do https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        
        
    