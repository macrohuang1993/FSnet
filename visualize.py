import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
def show(img):
    plt.imshow(img.permute([1,2,0])) #plt.imshow assumes the dims H,W,C
    plt.show()
def show_FS(FS,isshow = True, **kwargs):
    """
    one sample of FS:C,nF,H,W or a batch of FS: B,C,nF,H,W
    """
    shape = FS.shape
    if len(shape) == 4: 
        img_grid = make_grid(FS.permute([1,0,2,3]),padding = 30,**kwargs)

    elif len(shape) == 5:
        B,C,nF,H,W = shape
        img_grid = make_grid(FS.permute([0,2,1,3,4]).reshape(B*nF,C,H,W),nrow = nF, padding = 30,**kwargs)
    if isshow:
        show(img_grid)
    return img_grid
def show_SAI(LF,vu_pair_list, isshow = True, **kwargs):
    """
    LF: C,nv,nu,H,W or B,C,nv,nu,H,W
    uv_pair_list: [(v_1,u_1),(v_2,u_2)....]
    
    show number of SAI with angular location specified by uv_pair_list.
    """
    shape = LF.shape
    if len(shape) == 5:
        LF = LF.permute([1,2,0,3,4])
        SAI_list = [LF[vu_pair[0],vu_pair[1]] for vu_pair in vu_pair_list]
        img_grid = make_grid(SAI_list,padding = 30, **kwargs)
    elif len(shape) == 6:   
        B,C,nv,nu,H,W = shape
        LF = LF.permute([0,2,3,1,4,5])
        SAI_list = [LF[:,vu_pair[0],vu_pair[1]] for vu_pair in vu_pair_list]
        grids = [make_grid(list,padding = 30, nrow = 1,**kwargs) for list in SAI_list]
        img_grid = torch.cat(grids,dim = 2)
    if isshow:
        show(torch.cat(grids,dim = 2))
    return img_grid
def show_EPI_xu(LF,yv_pair_list,isshow = True, **kwargs):
    """
    LF: C,nv,nu,H,W or B,C,nv,nu,H,W
    yv_list: [(y_1,v_1),(y_2,v_2)....]
    
    show horizontal EPIs with y, v locations specified by yv_pair_list.
    """
    shape = LF.shape
    if len(shape) == 5:
        EPI_list = [LF[:,yv_pair[1],:,yv_pair[0],:] for yv_pair in yv_pair_list]
        img_grid = make_grid(EPI_list,padding = 10,**kwargs)

    elif len(shape) == 6:
        B,C,nv,nu,H,W = shape
        EPI_list = [LF[:,:,yv_pair[1],:,yv_pair[0],:] for yv_pair in yv_pair_list]
        grids = [make_grid(list,padding = 10, nrow = 1,**kwargs) for list in EPI_list]
        img_grid = torch.cat(grids,dim = 2)
    if isshow:
        show(img_grid)
    return img_grid
        
def show_EPI_yv(LF,xu_pair_list, isshow = True, **kwargs):
    """
    LF: C,nv,nu,H,W or B,C,nv,nu,H,W
    yv_list: [(x_1,u_1),(x_2,u_2)....]
    
    show vertical EPIs with x, u locations specified by xu_pair_list.
    """
    shape = LF.shape
    if len(shape) == 5:
        EPI_list = [LF[:,:,xu_pair[1],:,xu_pair[0]] for xu_pair in xu_pair_list]
        img_grid = make_grid(EPI_list,padding = 10,**kwargs)

    elif len(shape) == 6:
        B,C,nv,nu,H,W = shape
        EPI_list = [LF[:,:,:,xu_pair[1],:,xu_pair[0]] for xu_pair in xu_pair_list]
        grids = [make_grid(list,padding = 10, nrow = 1,**kwargs) for list in EPI_list]
        img_grid = torch.cat(grids,dim = 2)
    if isshow:
        show(img_grid)
    return img_grid
    