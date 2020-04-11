import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import os
import imageio

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
def show_SAI(LF,vu_pair_list = None, isshow = True, **kwargs):
    """
    LF: C,nv,nu,H,W or B,C,nv,nu,H,W or just a batch of SAI: B,C,H,W
    uv_pair_list: [(v_1,u_1),(v_2,u_2)....], ignored in the case of a batch of SAI: B,C,H,W
    
    show number of SAI with angular location specified by uv_pair_list.
    """
    shape = LF.shape
    if len(shape) == 4:# case of a batch of SAI: 
        LF = torch.unsqueeze(torch.unsqueeze(LF,2),3) #expand B,C,H,W into B,C,1,1,H,W
        LF = LF.permute([0,2,3,1,4,5])
        SAI_list = [LF[:,0,0]]
        grids = [make_grid(list,padding = 30, nrow = 1,**kwargs) for list in SAI_list]
        img_grid = torch.cat(grids,dim = 2)
    elif len(shape) == 5:
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


def generateGIF(idx_v,idx_u,save_folder,reconLF_all,trueLF_all=None,raydepth_all=None,FS_all=None):
    """
    Generate GIF of reconLF, trueLF, raydepth along aperture path specified by idx_u,idx_v
    Input:
        reconLF_all: numpy array of N,3,7,7,H,W
        trueLF_all: numpy array of N,3,7,7,H,W, if None, skip plotting
        raydepth_all: numpy array of N,7,7,H,W, if None, skip plotting
        FS_all: numpy array of N,3,7,H,W, if None, skip plotting
        idx_u,idx_v: list of aperture coordinate index, e.g.:
            idx_v=[0,0,0,0,0,0,0,1,2,3,4,5,6,6,6,6,6,6,6,5,4,3,2,1,0]
            idx_u=[0,1,2,3,4,5,6,6,6,6,6,6,6,5,4,3,2,1,0,0,0,0,0,0,0]
        save_folder: path of folder to save all gif, e.g.：
            save_folder = 'paper result/More Data for presentation/'+ '_'.join(model_folder.split('/')[1:])
    """
    #Loop all samples
    sample_indices = range(reconLF_all.shape[0])
    #sample_indices = [0]
    for sample_idx in sample_indices:
        print('Processing {0:02d}/{1:02d} Samples'.format(sample_idx+1,len(sample_indices)))
        save_path = os.path.join(save_folder,'sample_idx_{:03d}'.format(sample_idx))
        filenames_LFrecon=[]
        filenames_trueLF=[]
        filenames_FS=[]
        filenames_depth=[]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(len(idx_u)):
            fig = plt.imshow(reconLF_all[sample_idx,:,idx_v[i],idx_u[i],:,:].transpose([1,2,0]))
            plt.title('u={0:d}, v={1:d}'.format(idx_u[i],idx_v[i]))
            filename = '{:d}_{:d}_recon.png'.format(idx_v[i],idx_u[i])
            filename = os.path.join(save_path,filename)
            plt.savefig(filename)
            filenames_LFrecon.append(filename)
        if trueLF_all is not None:
            for i in range(len(idx_u)):
                fig = plt.imshow(trueLF_all[sample_idx,:,idx_v[i],idx_u[i],:,:].transpose([1,2,0]))
                plt.title('u={0:d}, v={1:d}'.format(idx_u[i],idx_v[i]))
                filename = '{:d}_{:d}_true.png'.format(idx_v[i],idx_u[i])
                filename = os.path.join(save_path,filename)
                plt.savefig(filename)
                filenames_trueLF.append(filename)
        if raydepth_all is not None:
            for i in range(len(idx_u)):
                fig = plt.imshow(raydepth_all[sample_idx,idx_v[i],idx_u[i],:,:],vmin=raydepth_all[sample_idx].min(),vmax=raydepth_all[sample_idx].max(),cmap=plt.get_cmap('gray'))
                plt.title('u={0:d}, v={1:d}'.format(idx_u[i],idx_v[i]))
                filename = '{:d}_{:d}_depth.png'.format(idx_v[i],idx_u[i])
                filename = os.path.join(save_path,filename)
                plt.savefig(filename)
                filenames_depth.append(filename)
        
        #FS_all = np.zeros([len(ds_val),3,7,185,269])
        if FS_all is not None:
            for i in range(FS_all.shape[2]):
                fig = plt.imshow(FS_all[sample_idx,:,i,:,:].transpose([1,2,0]))
                plt.title('idx={:d}'.format(i))
                filename = 'FS_idx_{:d}.png'.format(i)
                filename = os.path.join(save_path,filename)
                plt.savefig(filename)
                filenames_FS.append(filename)
                
        plt.close()

        images = []
        for filename in filenames_LFrecon:
            images.append(imageio.imread(filename))
        imageio.mimsave(os.path.join(save_path,'LFrecon.gif'), images)
        
        if trueLF_all is not None:
            images = []
            for filename in filenames_trueLF:
                images.append(imageio.imread(filename))
            imageio.mimsave(os.path.join(save_path,'trueLF.gif'), images)
        if raydepth_all is not None:
            images = []
            for filename in filenames_depth:
                images.append(imageio.imread(filename))
            imageio.mimsave(os.path.join(save_path,'raydepth.gif'), images)
        if FS_all is not None:
            images = []
            for filename in filenames_FS:
                images.append(imageio.imread(filename))
            imageio.mimsave(os.path.join(save_path,'FS.gif'), images,duration=0.2) 
    
    

def generateSAI(save_folder,trueLF_all):
    N,C,nv,nu,H,W = trueLF_all.shape
    sample_indices = range(trueLF_all.shape[0])
    for sample_idx in sample_indices:
            print('Processing {0:02d}/{1:02d} Samples'.format(sample_idx+1,len(sample_indices)))
            save_path = os.path.join(save_folder,'sample_idx_{:03d}'.format(sample_idx))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            fig = plt.imshow(trueLF_all[sample_idx,:,nv//2,nu//2,:,:].transpose([1,2,0]))
            plt.axis('off')
            #plt.title('Centeral SAI'.format(idx_u[i],idx_v[i]))
            filename = 'Central SAI.png'
            filename = os.path.join(save_path,filename)
            plt.savefig(filename)
            plt.close()
