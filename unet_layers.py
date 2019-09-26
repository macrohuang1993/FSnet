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
    def __init__(self,nF=None,nu=None,nv=None, box_constraint = None):
        """
        box_constraint:whether constrain the output value to [0,1],  using either 'tanh' or 'sigmoid'. If box_constraint = None, nothing done. 
        """
        super(unet_FS2LF_v2,self).__init__()
        self.nu,self.nv = nu,nv
        self. box_constraint = box_constraint
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
        if self.box_constraint == None:
            pass
        elif self.box_constraint == 'tanh':
            LF_rgb = (F.tanh(LF_rgb) + torch.tensor(1).to(LF_rgb))/2
        elif self.box_constraint == 'sigmoid':
            LF_rgb = F.sigmoid(LF_rgb)
        else: 
            raise('Wrong box_constraint')
        LF_rgb = LF_rgb.view(B,C,self.nv,self.nu,H,W) #LF_rgb: B,C,nv,nu,H,W
        return LF_rgb
    
class unet_FS2LF_v3(nn.Module):
    def __init__(self,nF=None,nu=None,nv=None,C=3, box_constraint = None):
        super(unet_FS2LF_v3,self).__init__()
        self.nu,self.nv = nu,nv
        self. box_constraint = box_constraint
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
        if self.box_constraint == None:
            pass
        elif self.box_constraint == 'tanh':
            LF_rgb = (F.tanh(LF_rgb) + torch.tensor(1).to(LF_rgb))/2
        elif self.box_constraint == 'sigmoid':
            LF_rgb = F.sigmoid(LF_rgb)
        else: 
            raise('Wrong box_constraint')
        LF_rgb = LF_rgb.view(B,C,self.nv,self.nu,H,W) #LF_rgb: B,C,nv,nu,H,W
        return LF_rgb        
    
# case: 3D conv case 

#to do https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        
        
        
class unet_FS2SAI_v3(nn.Module):
    def __init__(self,nF=None,nu=None,nv=None,C=3, box_constraint = None):
        super(unet_FS2SAI_v3,self).__init__()
        self.nu,self.nv = nu,nv
        self. box_constraint = box_constraint
        self.unet = unet(C*nF,C)

    def forward(self,FS_rgb):
        """
        Foward passing  FS to get SAI (all in focus image) using Unet, by merging the nF dimension of FS and the color dimension into the color channel in the unet.
        This is called v3 since this has similar structure to unet FS2LF v3
        Input:
        FS_rgb: FS of shape B,C,nF,H,W
        Output:
        SAI: B,C,H,W
        """
        B,C,nF,H,W = FS_rgb.shape
        FS_rgb = FS_rgb.view(B,C*nF,H,W)
        SAI_rgb = self.unet(FS_rgb) #LF_rgb: B,C,H,W
        if self.box_constraint == None:
            pass
        elif self.box_constraint == 'tanh':
            SAI_rgb = (F.tanh(SAI_rgb) + torch.tensor(1).to(SAI_rgb))/2
        elif self.box_constraint == 'sigmoid':
            SAI_rgb = F.sigmoid(SAI_rgb)
        else: 
            raise('Wrong box_constraint')
        return SAI_rgb     
    
    


class depth_network_pt(nn.Module):
    """
    Network for disparity field estimation from FS (and optionally estimated All in focus image), using dialated Convolution.
    Input:
        x:FS of shape B,C,nF,H,W
        lfsize: lightfield size (H,W,nv,nu)
        disp_mult: uplimit of the abs of the disparity. To be constrained by tanh
        concat_SAI: whether concat SAI with FS along color channel for the depth estimation.
        SAI_only: whether only use SAI for depth estimation (without use FS). When this is true, concat_SAI 
        
    Output:
        disparity field of shape B,nv,nu,H,W
    """
    def __init__(self,nF, lfsize, disp_mult, concat_SAI = False, SAI_only = False):
        super(depth_network_pt,self).__init__()
        self.v_sz,self.u_sz = lfsize[2],lfsize[3]
        self.disp_mult = disp_mult
        self.concat_SAI = concat_SAI
        self.SAI_only = SAI_only
        if SAI_only:
            C = 3 
        elif concat_SAI:
            C = 3 * nF +3
        else:
            C = 3*nF
        self.cnn_layers = nn.Sequential(
            cnn_layer(C,16),#conv => BN => LeakyReLU
            cnn_layer(16,64),
            cnn_layer(64,128),
            cnn_layer(128,128,dilation_rate = 2),
            cnn_layer(128,128,dilation_rate = 4),
            cnn_layer(128,128,dilation_rate = 8),
            cnn_layer(128,128,dilation_rate = 16),
            cnn_layer(128,128),
            cnn_layer(128,self.v_sz*self.u_sz),
            cnn_layer_plain(self.v_sz*self.u_sz, self.v_sz*self.u_sz)
        )
        self.tanh_NL = nn.Tanh()
    def forward(self,x, *args):
        if self.SAI_only:
            B,C,H,W = x.shape #input x is the SAI
        else:
            B,C,nF,H,W = x.shape #input x is the FS
            x = x.reshape(B,C*nF,H,W)
        if len(args) == 0:
            pass
        else:
            assert len(args) == 1 and self.concat_SAI and len(args[0].shape) == 4 ## check it is one 4D tensor (All in focus image)
            SAI = args[0]
            x = torch.cat([x,SAI], dim = 1) #concatenate FS and SAI along color channel
        x = self.cnn_layers(x) # A series of convolution (some dilated)
        x = self.disp_mult * self.tanh_NL(x) #constrain the output range
        return x.reshape(B,self.v_sz,self.u_sz,H,W) #Estimated depth fields, B,v,u,H,W
            
        
    
class cnn_layer(nn.Module):
    '''((possibly dilated)conv => BN => LeakyReLU), following learning Local_light field synthesis paper, used in depth_network_pt'''
    def __init__(self, in_ch, out_ch,filter_size = 3, dilation_rate = 1):
        super(cnn_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(dilation_rate*(filter_size-1)//2),
            nn.Conv2d(in_ch, out_ch, filter_size, padding=0,dilation = dilation_rate),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class cnn_layer_plain(nn.Module):
    '''((possibly dilated)conv), following learning Local_light field synthesis paper, used in depth_network_pt'''

    def __init__(self, in_ch, out_ch,filter_size = 3, dilation_rate = 1):
        super(cnn_layer_plain, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(dilation_rate*(filter_size-1)//2),
            nn.Conv2d(in_ch, out_ch, filter_size, padding=0,dilation = dilation_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class refineNet(nn.Module):
    """
    Network for denoising DIBR lambertian LF, using residual network and  3D conv along H,W,uv dimension. 
    Input:
        x:DIBR Lambertian LF of shape B,C,v,u,H,W,
        ray_depths: of shape B,v,u,H,W

        
    Output:
        Denoised LF of shape B,C,v,u,H,W
    """
    def __init__(self,):
        super(refineNet,self).__init__()
        
        # Takes in B,3+1,vu,H,W and output B,3,vu,H,W
        self.cnn_layers = nn.Sequential(
            cnn_layer_3D(4,8),#conv => BN => LeakyReLU
            cnn_layer_3D(8,8),
            cnn_layer_3D(8,8),
            cnn_layer_3D(8,8),
            cnn_layer_plain_3D(8, 3)
        )
        self.tanh_NL = nn.Tanh()
    def forward(self,x,ray_depths):
        B,C,nv,nu,H,W = x.shape
        
        ray_depths = ray_depths.unsqueeze(1) #B,1,v,u,H,W
        x2 = torch.cat([x,ray_depths],dim = 1) #B,4,v,u,H,W
        x2 = x2.reshape(B,C+1,nv*nu,H,W) #B,4,v*u,H,W
        x2 = self.cnn_layers(x2) #B,3,v*u,H,W, after # A series of 3D convolution 
        x2 = x2.reshape(B,C,nv,nu,H,W)
        #x = (self.tanh_NL(x) + 1) / 2 #constrain the output range to [0,1], I dont think we need to do it. 
        x = x + x2 #residual node
        return x #Estimated LF, B,C,v,u,H,W
    
    
class cnn_layer_3D(nn.Module):
    '''((possibly dilated)conv => BN => LeakyReLU), following learning Local_light field synthesis paper, used in refineNet'''
    def __init__(self, in_ch, out_ch,filter_size = 3, dialation_rate = 1):
        super(cnn_layer_3D, self).__init__()
        self.conv = nn.Sequential(
            ReflectionPad3d(dialation_rate*(filter_size-1)//2),
            nn.Conv3d(in_ch, out_ch, filter_size, padding=0,dilation = dialation_rate),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class cnn_layer_plain_3D(nn.Module):
    '''((possibly dilated)conv), following learning Local_light field synthesis paper, used in refineNet'''

    def __init__(self, in_ch, out_ch,filter_size = 3, dialation_rate = 1):
        super(cnn_layer_plain_3D, self).__init__()
        self.conv = nn.Sequential(
            ReflectionPad3d(dialation_rate*(filter_size-1)//2),
            nn.Conv3d(in_ch, out_ch, filter_size, padding=0,dilation = dialation_rate)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class ReflectionPad3d(nn.Module):
    """
    My 3D padding, padding a tensor's last three dimension's each size by padding_size.
    """
    def __init__(self,padding_size):
        super(ReflectionPad3d,self).__init__()
        self.pad = (padding_size,padding_size,padding_size,padding_size,padding_size,padding_size)
    def forward(self,x):
        #return F.pad(x, self.pad, mode='reflect') this will raise none implement error
        return F.pad(x, self.pad, mode='replicate') 