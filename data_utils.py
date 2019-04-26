import torch
import numpy as np
import h5py
import os
from PIL import Image  
from torch.utils.data import Dataset as dset

import time
import pdb
def read_lytroLF_as5D(file_path,lfsize):
    """
    Read raw lytro format light field.png file and format it into shape of 3,H,W,nv,nu.
    Note here the lfsize nv and nu may specifys a subregion of the entire aperture
    """
    img = Image.open(file_path)
    #Following step is bottlenecking the training. The image.read is slow 
    LF = np.asarray(img)[:lfsize[0]*14,:lfsize[1]*14,:3] #H*nv,W*nu,3. AFter removing the 4th transparency channel, and possibly cropping the H,W
    LF = LF.reshape([lfsize[0],14,lfsize[1],14,3]).transpose([4,0,2,1,3]) #3,H,W,nv,nu
    LF = LF[:, :, :, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2), (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2)] #only around central region of entire aperture, since its blank around border of the aperture

    return LF

def read_lytroLF_as5D_fromh5(h5_path,LF_name,lfsize):
    """
    Read raw lytro format light field from .h5 data file and format it into shape of 3,H,W,nv,nu.
    Note here the lfsize nv and nu may specifys a subregion of the entire aperture
    """
    f =  h5py.File(h5_path,'r')
    #t1=time.time()
    #Following step is bottlenecking the training. The image.read is slow 
    LF = f[LF_name][:]
    f.close()
    #t2=time.time()
    #print(t2-t1)
    LF = LF[:lfsize[0]*14,:lfsize[1]*14,:3] #H*nv,W*nu,3. AFter removing the 4th transparency channel, and possibly cropping the H,W
    LF = LF.reshape([lfsize[0],14,lfsize[1],14,3]).transpose([4,0,2,1,3]) #3,H,W,nv,nu
    LF = LF[:, :, :, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2), (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2)] #only around central region of entire aperture, since its blank around border of the aperture

    return LF

    
class FSdataset(dset):
    def __init__(self,lfsize = None, FSdata_path=None,LFdata_folder=None,trainorval = 'train', transform = None):
        super(FSdataset,self).__init__()
        self.lfsize = lfsize
        self.FSdata_path = FSdata_path
        self.LFdata_folder = LFdata_folder
        self.trainorval = trainorval
        f =  h5py.File(self.FSdata_path,'r')
        f_FS = f[trainorval]
        self.namelist = list(f_FS.keys())
        f.close()
        self.transform = transform
    def __len__(self):
        return len(self.namelist)
    def __getitem__(self,index):
        
        f= h5py.File(self.FSdata_path,'r')
        f_FS = f[self.trainorval]
        FS = f_FS[self.namelist[index]][:]
        LF = read_lytroLF_as5D(os.path.join(self.LFdata_folder,self.namelist[index]),self.lfsize)
        #LF = read_lytroLF_as5D_fromh5(('LF_chunked.h5'),self.namelist[index],self.lfsize)  
        LF = LF.transpose([0,3,4,1,2]) #since output from read_lytroLF_as5D has dim C,H,W,nv,nu. The network expect C,nv,nu,H,W
        FS = FS.transpose([1,0,2,3])  #since refocused images in h5 file generated from generate_FS has shape nF,C,H,W, the network expect C,nF,H,W
        
        f.close()
        if self.transform is not None:
            # possible normalization and augmentation done here.
            FS, LF = self.transform(FS), self.transform(LF)

        return {'FS':FS,'LF':LF}
    
    
class my_RandomCrop(object):
    """Crop randomly a multidimensional image in last two dimensions.

    Args:
        output_size (tuple (Hout,Wout) or int (Hout=Wout)): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, nDImage):
        """
        nDImage should of size (..., H,W)
        return cropped nDImage, cropped in H,W dimension
        """
        #image, landmarks = sample['image'], sample['landmarks']

        h, w = nDImage.shape[-2:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        nDImage = nDImage[...,top: top + new_h,
                      left: left + new_w]
        return nDImage  #cropped in H,W dimension

    
class my_normalize(object):
    """Normalize a tensor image by value normf.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        normf :

    Returns:
        Tensor: Normalized Tensor .
    """
    def __init__(self, normf):
        self.normf = normf
    def __call__(self,nparray):
        tensor=torch.from_numpy(nparray).to(torch.float32)
        tensor = tensor.div(self.normf)
        return tensor
