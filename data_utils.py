import torch
import numpy as np
import h5py
import os
from PIL import Image  
from torch.utils.data import Dataset as dset
from numbers import Number
import time
import pdb
def read_lytroLF_as5D(file_path,lfsize):
    """
    Read raw lytro format light field.png file and format it into shape of 3,H,W,nv,nu.
    Note here the lfsize nv and nu may specifys a subregion of the entire aperture
    """
    with open(file_path, 'rb') as f:
        img = Image.open(f)
        #Following step is bottlenecking the training. The image.read is slow 
        LF = np.asarray(img)[:lfsize[0]*14,:lfsize[1]*14,:3] #H*nv,W*nu,3. AFter removing the 4th transparency channel, and possibly cropping the H,W
        LF = LF.reshape([lfsize[0],14,lfsize[1],14,3]).transpose([4,0,2,1,3]) #3,H,W,nv,nu
        LF = LF[:, :, :, (14//2)-(lfsize[2]//2):(14//2)+(lfsize[2]//2), (14//2)-(lfsize[3]//2):(14//2)+(lfsize[3]//2)] #only around central region of entire aperture, since its blank around border of the aperture
        return LF

def read_lytroLF_as5D_fromh5(h5_path,LF_name,lfsize):
    """
    Read raw lytro format light field from .h5 data file and then crop and format it into shape of 3,H,W,nv,nu.
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

def read_preprocessedLF_fromh5(h5_path,LF_name):
    """
    Read preprocessed LF generated in matlab by inversecrime.m from h5py file. 
    Each sample loaded has shape 3,H,W,nv,nu
    """
    f =  h5py.File(h5_path,'r')
    LF = f[LF_name][:]
    f.close()
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
        sample = {'FS':FS,'LF':LF}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample

class FSdataset_h5(dset):
    """
    Same as FSdataset, except the light field dataset is loaded from h5 file.
    Used for training in avoiding inverse crime cases, where LF is saved as h5 file.
    """
    def __init__(self, FSdata_path=None,LFdata_path=None,trainorval = 'train', transform = None):
        super(FSdataset_h5,self).__init__()
        self.FSdata_path = FSdata_path
        self.LFdata_path = LFdata_path
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
        LF = read_preprocessedLF_fromh5(self.LFdata_path,self.namelist[index])  
        LF = LF.transpose([0,3,4,1,2]) #since output from read_lytroLF_as5D has dim C,H,W,nv,nu. The network expect C,nv,nu,H,W
        FS = FS.transpose([1,0,2,3])  #since refocused images in h5 file generated from generate_FS has shape nF,C,H,W, the network expect C,nF,H,W
        
        f.close()
        sample = {'FS':FS,'LF':LF}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample
    
class FSdataset_withName(dset):
    """
    Same as FSdataset, except return name as well, used in generateSAI.ipynb without avoiding inverse crime.
    """
    def __init__(self,lfsize = None, FSdata_path=None,LFdata_folder=None,trainorval = 'train', transform = None):
        super(FSdataset_withName,self).__init__()
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
        sample = {'FS':FS,'LF':LF}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample,self.namelist[index]
class FSdataset_withName_h5(dset):
    """
    Same as FSdataset_withName, except reading LF from h5 file and hence lfsize argument is not needed and LFdata_folder arg is changed to LFdata_path.
    """
    def __init__(self, FSdata_path=None,LFdata_path=None,trainorval = 'train', transform = None):
        super(FSdataset_withName_h5,self).__init__()
        self.FSdata_path = FSdata_path
        self.LFdata_path = LFdata_path
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
        LF = read_preprocessedLF_fromh5(self.LFdata_path,self.namelist[index])  
        #LF = read_lytroLF_as5D_fromh5(('LF_chunked.h5'),self.namelist[index],self.lfsize)  
        LF = LF.transpose([0,3,4,1,2]) #since output from read_lytroLF_as5D has dim C,H,W,nv,nu. The network expect C,nv,nu,H,W
        FS = FS.transpose([1,0,2,3])  #since refocused images in h5 file generated from generate_FS has shape nF,C,H,W, the network expect C,nF,H,W
        
        f.close()
        sample = {'FS':FS,'LF':LF}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample,self.namelist[index]
    
class FSdataset_withSAI(dset):
    """
    FS dataset with estimated SAI image from ViewNet included
    """
    def __init__(self,lfsize = None, FSdata_path=None,LFdata_folder=None,SAIdata_folder = None, trainorval = 'train', transform = None):
        super(FSdataset_withSAI,self).__init__()
        self.lfsize = lfsize
        self.FSdata_path = FSdata_path
        self.LFdata_folder = LFdata_folder
        self.SAIdata_folder = SAIdata_folder
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
        f2 = h5py.File(self.SAIdata_folder,'r')
        f_FS = f[self.trainorval]
        f_SAI = f2[self.trainorval] #estimated SAI
        FS = f_FS[self.namelist[index]][:]
        SAI = f_SAI[self.namelist[index]][:] #C,H,W
        LF = read_lytroLF_as5D(os.path.join(self.LFdata_folder,self.namelist[index]),self.lfsize)
        #LF = read_lytroLF_as5D_fromh5(('LF_chunked.h5'),self.namelist[index],self.lfsize)  
        LF = LF.transpose([0,3,4,1,2]) #since output from read_lytroLF_as5D has dim C,H,W,nv,nu. The network expect C,nv,nu,H,W
        FS = FS.transpose([1,0,2,3])  #since refocused images in h5 file generated from generate_FS has shape nF,C,H,W, the network expect C,nF,H,W
        
        f.close()
        f2.close()
        sample = {'FS':FS,'LF':LF,'SAI':SAI}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample
    
class FSdataset_withSAI_h5(dset):
    """
    FS dataset with estimated SAI image from ViewNet included, with LF data loaded from h5 file. Used in DIBR_train_invC.ipynb.
    """
    def __init__(self, FSdata_path=None,LFdata_path=None,SAIdata_folder = None, trainorval = 'train', transform = None):
        super(FSdataset_withSAI_h5,self).__init__()
        self.FSdata_path = FSdata_path
        self.LFdata_path = LFdata_path
        self.SAIdata_folder = SAIdata_folder
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
        f2 = h5py.File(self.SAIdata_folder,'r')
        f_FS = f[self.trainorval]
        f_SAI = f2[self.trainorval] #estimated SAI
        FS = f_FS[self.namelist[index]][:]
        SAI = f_SAI[self.namelist[index]][:] #C,H,W
        LF = read_preprocessedLF_fromh5(self.LFdata_path,self.namelist[index])  
        #LF = read_lytroLF_as5D_fromh5(('LF_chunked.h5'),self.namelist[index],self.lfsize)  
        LF = LF.transpose([0,3,4,1,2]) #since output from read_lytroLF_as5D has dim C,H,W,nv,nu. The network expect C,nv,nu,H,W
        FS = FS.transpose([1,0,2,3])  #since refocused images in h5 file generated from generate_FS has shape nF,C,H,W, the network expect C,nF,H,W
        
        f.close()
        f2.close()
        sample = {'FS':FS,'LF':LF,'SAI':SAI}
        if self.transform is not None:
            # possible normalization and augmentation done here.
            sample = self.transform(sample)
        return sample
        
    
    
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
    
class my_gamma_correction(object):
    """
    Apply random gamma agumentation to the image. 
    Input: torch tensor normalized to range [0,1]
    """
    def __init__(self,gamma_min,gamma_max):
        self.gamma_range = [gamma_min,gamma_max]
    def __call__(self,tensor):
        gam = np.random.uniform(low=self.gamma_range[0],high = self.gamma_range[1])
        return torch.pow(tensor,gam)
    
    
class my_paired_RandomCrop(object):
    """Crop randomly multidimensional input and target in last two dimensions.

    Args:
        output_size: (tuple (Hout,Wout) or int (Hout=Wout)): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        sample: should be a dict of two item, of size (..., H,W)
        return cropped input_target_pair, cropped in H,W dimension
        """
        
        FS, LF = sample['FS'], sample['LF']
        h, w = FS.shape[-2:]
        h2,w2 = LF.shape[-2:]
        assert h == h2 and w == w2
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        FS = FS[...,top: top + new_h,
                      left: left + new_w]
        LF = LF[...,top: top + new_h,
                      left: left + new_w] 
        return {'FS':FS,'LF':LF}  #cropped in H,W dimension
    
class my_paired_normalize(object):
    """Normalize  input and target by value normf.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        normf : a scalar or a dict with {'FS':normf_FS,'LF':normf_LF}

    Returns:
        Tensor: Normalized Tensor .
    """
    def __init__(self, normf):
        if isinstance(normf,Number):
            self.normf_LF, self.normf_FS = normf,normf
        else:
            assert(isinstance(normf,dict))
            self.normf_LF, self.normf_FS = normf['LF'],normf['FS']
    def __call__(self,sample):
        
        FS, LF = sample['FS'], sample['LF'] 
        
        FS, LF = torch.from_numpy(FS).to(torch.float32),torch.from_numpy(LF).to(torch.float32),
        FS, LF = FS.div(self.normf_FS), LF.div(self.normf_LF)
        return {'FS':FS,'LF':LF}
    
class my_paired_gamma_correction(object):
    """
    Apply random gamma agumentation to the image. 
    Input: dict of input and target (torch tensor)  normalized to range [0,1]
    """
    def __init__(self,gamma_min,gamma_max):
        self.gamma_range = [gamma_min,gamma_max]
    def __call__(self,sample):
        FS, LF = sample['FS'], sample['LF'] 
        gam = np.random.uniform(low=self.gamma_range[0],high = self.gamma_range[1])
        FS, LF = torch.pow(FS,gam), torch.pow(LF,gam)
        return {'FS':FS,'LF':LF}
    
class my_triplet_normalize(object):
    """Normalize  FS and LF by value normf, the SAI is not normalized since it's already normalized in previous ViewNet training.
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
        if isinstance(normf,Number):
            self.normf_LF, self.normf_FS = normf,normf
        else:
            assert(isinstance(normf,dict))
            self.normf_LF, self.normf_FS = normf['LF'],normf['FS']
    def __call__(self,sample):
        
        FS, LF, SAI = sample['FS'], sample['LF'], sample['SAI'] 
        
        FS, LF, SAI= torch.from_numpy(FS).to(torch.float32),torch.from_numpy(LF).to(torch.float32),torch.from_numpy(SAI).to(torch.float32)
        FS, LF = FS.div(self.normf_FS), LF.div(self.normf_LF) #Only normalize LF and FS
        return {'FS':FS,'LF':LF, 'SAI':SAI}
    
class my_triplet_RandomCrop(object):
    """Crop randomly multidimensional FS and LF and SAI in last two dimensions.

    Args:
        output_size: (tuple (Hout,Wout) or int (Hout=Wout)): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        sample: should be a dict of two item, of size (..., H,W)
        return cropped input_target_pair, cropped in H,W dimension
        """
        
        FS, LF, SAI = sample['FS'], sample['LF'], sample['SAI'] 
        h, w = FS.shape[-2:]
        h2,w2 = LF.shape[-2:]
        h3,w3 = SAI.shape[-2:]
        assert h == h2 and w == w2 and h == h3 and w == w3
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        FS = FS[...,top: top + new_h,
                      left: left + new_w]
        LF = LF[...,top: top + new_h,
                      left: left + new_w] 
        SAI = SAI[...,top: top + new_h,
                      left: left + new_w] 
        return {'FS':FS,'LF':LF, 'SAI':SAI}  #cropped in H,W dimension
    
class my_triplet_CenterCrop(object):
    """Crop out central multidimensional FS and LF and SAI in last two dimensions.

    Args:
        output_size: (tuple (Hout,Wout) or int (Hout=Wout)): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        sample: should be a dict of two item, of size (..., H,W)
        return cropped input_target_pair, cropped in H,W dimension
        """
        
        FS, LF, SAI = sample['FS'], sample['LF'], sample['SAI'] 
        h, w = FS.shape[-2:]
        h2,w2 = LF.shape[-2:]
        h3,w3 = SAI.shape[-2:]
        assert h == h2 and w == w2 and h == h3 and w == w3
        new_h, new_w = self.output_size
        
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        FS = FS[...,top: top + new_h,
                      left: left + new_w]
        LF = LF[...,top: top + new_h,
                      left: left + new_w] 
        SAI = SAI[...,top: top + new_h,
                      left: left + new_w] 
        return {'FS':FS,'LF':LF, 'SAI':SAI}  #cropped in H,W dimension