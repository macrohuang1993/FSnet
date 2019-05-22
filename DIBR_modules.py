import torch
import torch.nn.functional as F
import numpy as np
def depth_rendering_pt(central, ray_depths, lfsize,central_u,central_v):
    """
    Render lambertian lightfield by backward warping using central SAI and depth field
    
    Input: central:torch tensor of single color channel central SAI (B,H,W)
        ray_depths: torch tensor of  depth field (B,H,W,nv,nu)
        lfsize:tuple with elements (H,W,nv,nu)
    Output: torch tensor of rendered light field (B,H,W,nv,nu) 
    
    Passed testing against original tensorflow version.
    """
    b_sz = central.shape[0]
    y_sz = central.shape[1]
    x_sz = central.shape[2]
    v_sz = lfsize[2]
    u_sz = lfsize[3]

    central = torch.unsqueeze(torch.unsqueeze(central,3),4)

    #create and reparameterize light field grid
    b_vals = torch.from_numpy(np.arange(b_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)
    v_vals = torch.from_numpy(np.arange(v_sz).astype(np.float32) - central_v).to(dtype = ray_depths.dtype, device = ray_depths.device) # The value substract here has to be the u,v coordinate of central, because we care warping the central.
    u_vals = torch.from_numpy(np.arange(u_sz).astype(np.float32) - central_u).to(dtype = ray_depths.dtype, device = ray_depths.device)
    y_vals = torch.from_numpy(np.arange(y_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)
    x_vals = torch.from_numpy(np.arange(x_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)

    #b_vals = tf.to_float(tf.range(b_sz))
    #v_vals = tf.to_float(tf.range(v_sz)) - tf.to_float(v_sz)/2.0
    #u_vals = tf.to_float(tf.range(u_sz)) - tf.to_float(u_sz)/2.0
    #y_vals = tf.to_float(tf.range(y_sz))
    #x_vals = tf.to_float(tf.range(x_sz))

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals)
    #b, y, x, v, u = tf.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals, indexing='ij')

    #warp coordinates by ray depths
    y_t = y + v * ray_depths
    x_t = x + u * ray_depths

    v_r = torch.zeros_like(b)
    u_r = torch.zeros_like(b)

    #indices for linear interpolation
    b_1 = b.to(torch.int32)
    y_1 = torch.floor(y_t).to(torch.int32)
    y_2 = y_1 + 1
    x_1 = torch.floor(x_t).to(torch.int32)
    x_2 = x_1 + 1
    v_1 = v_r.to(torch.int32)
    u_1 = u_r.to(torch.int32)

    y_1 = torch.clamp(y_1, 0, y_sz-1)
    y_2 = torch.clamp(y_2, 0, y_sz-1)
    x_1 = torch.clamp(x_1, 0, x_sz-1)
    x_2 = torch.clamp(x_2, 0, x_sz-1)

    #assemble interpolation indices
    interp_pts_1 = torch.cat([b_1.unsqueeze(5), y_1.unsqueeze(5), x_1.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)],-1)
    interp_pts_2 = torch.cat([b_1.unsqueeze(5), y_2.unsqueeze(5), x_1.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)
    interp_pts_3 = torch.cat([b_1.unsqueeze(5), y_1.unsqueeze(5), x_2.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)
    interp_pts_4 = torch.cat([b_1.unsqueeze(5), y_2.unsqueeze(5), x_2.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)

    #gather light fields to be interpolated
    
    lf_1 = gather_nd_pt(central, interp_pts_1.to(torch.long))
    lf_2 = gather_nd_pt(central, interp_pts_2.to(torch.long))
    lf_3 = gather_nd_pt(central, interp_pts_3.to(torch.long))
    lf_4 = gather_nd_pt(central, interp_pts_4.to(torch.long))        

    #calculate interpolation weights        
    y_1_f = y_1.to(torch.float)
    x_1_f = x_1.to(torch.float)
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1

    w1 = d_y_1 * d_x_1
    w2 = d_y_2 * d_x_1
    w3 = d_y_1 * d_x_2
    w4 = d_y_2 * d_x_2

    lf = w1*lf_1 + w2*lf_2 + w3*lf_3 + w4*lf_4

    return lf


def gather_nd_pt(params,indices):
    # my pytorch version of gather_nd in tensorflow for following specific inputs
    # params:(d1,d2,d3,d4,d5)
    # indices:(D1,D2,D3,D4,D5,5) 
    # follow https://discuss.pytorch.org/t/how-to-use-tf-gather-nd-in-pytorch/28271/3
    # checkout numpy advanced indexing for better understanding 
    # think about generalization？
    # checkout https://github.com/ashawkey/hawtorch/blob/6694b8cbf1adcad801e45653781c3f01ea13a37a/hawtorch/nn/functional.py
    return params[indices[...,0],indices[...,1],indices[...,2],indices[...,3],indices[...,4]]

def transform_ray_depths_pt(ray_depths, u_step, v_step, lfsize,central_u,central_v):
    """
    resample ray depths for depth consistency regularization
    Tested against original tensorflow code
    Input： 
        ray_depths： B,H,W,v,u
    Output：
        lf（transformed ray depths)： B,H,W,v,u
    """
    b_sz = ray_depths.shape[0]
    y_sz = ray_depths.shape[1]
    x_sz = ray_depths.shape[2]
    v_sz = lfsize[2]
    u_sz = lfsize[3]

    #create and reparameterize light field grid
    b_vals = torch.from_numpy(np.arange(b_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)
    v_vals = torch.from_numpy(np.arange(v_sz).astype(np.float32) - central_v).to(dtype = ray_depths.dtype, device = ray_depths.device)
    u_vals = torch.from_numpy(np.arange(u_sz).astype(np.float32) - central_u).to(dtype = ray_depths.dtype, device = ray_depths.device)
    y_vals = torch.from_numpy(np.arange(y_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)
    x_vals = torch.from_numpy(np.arange(x_sz).astype(np.float32)).to(dtype = ray_depths.dtype, device = ray_depths.device)

    b, y, x, v, u = torch.meshgrid(b_vals, y_vals, x_vals, v_vals, u_vals)

    #warp coordinates by ray depths
    
    y_t = y + v_step * ray_depths
    x_t = x + u_step * ray_depths
    v_t = v - v_step + central_v
    u_t = u - u_step + central_u #Modified by me
                
    #v_t = v - v_step + float(v_sz)/2.0
    #u_t = u - u_step + float(u_sz)/2.0
    

                              
    #v_t = v - v_step + tf.to_float(v_sz)/2.0
    #u_t = u - u_step + tf.to_float(u_sz)/2.0

    #indices for linear interpolation
    b_1 = b.to(torch.int32)
    y_1 = torch.floor(y_t).to(torch.int32)
    y_2 = y_1 + 1
    x_1 = torch.floor(x_t).to(torch.int32)
    x_2 = x_1 + 1
    v_1 = v_t.to(torch.int32)
    u_1 = u_t.to(torch.int32)
    
    y_1 = torch.clamp(y_1, 0, y_sz-1)
    y_2 = torch.clamp(y_2, 0, y_sz-1)
    x_1 = torch.clamp(x_1, 0, x_sz-1)
    x_2 = torch.clamp(x_2, 0, x_sz-1)
    v_1 = torch.clamp(v_1, 0, v_sz-1)
    u_1 = torch.clamp(u_1, 0, u_sz-1)


    #assemble interpolation indices
    
    interp_pts_1 = torch.cat([b_1.unsqueeze(5), y_1.unsqueeze(5), x_1.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)],-1)
    interp_pts_2 = torch.cat([b_1.unsqueeze(5), y_2.unsqueeze(5), x_1.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)
    interp_pts_3 = torch.cat([b_1.unsqueeze(5), y_1.unsqueeze(5), x_2.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)
    interp_pts_4 = torch.cat([b_1.unsqueeze(5), y_2.unsqueeze(5), x_2.unsqueeze(5), v_1.unsqueeze(5), u_1.unsqueeze(5)], -1)


    #gather light fields to be interpolated
    
    lf_1 = gather_nd_pt(ray_depths, interp_pts_1.to(torch.long))
    lf_2 = gather_nd_pt(ray_depths, interp_pts_2.to(torch.long))
    lf_3 = gather_nd_pt(ray_depths, interp_pts_3.to(torch.long))
    lf_4 = gather_nd_pt(ray_depths, interp_pts_4.to(torch.long))   

    #calculate interpolation weights
    
    y_1_f = y_1.to(torch.float)
    x_1_f = x_1.to(torch.float)
    d_y_1 = 1.0 - (y_t - y_1_f)
    d_y_2 = 1.0 - d_y_1
    d_x_1 = 1.0 - (x_t - x_1_f)
    d_x_2 = 1.0 - d_x_1


    w1 = d_y_1 * d_x_1
    w2 = d_y_2 * d_x_1
    w3 = d_y_1 * d_x_2
    w4 = d_y_2 * d_x_2

    lf = w1*lf_1 + w2*lf_2 + w3*lf_3 + w4*lf_4
    return lf

#loss to encourage consistency of ray depths corresponding to same scene point

def depth_consistency_loss_pt(x, lfsize,central_u,central_v):
    """
    x: depth_fields of shape:B,H,W,v,u
    call signiture: depth_consistency_loss(ray_depths, lfsize)
    Tested against original tensorflow code
    """
    x_u = transform_ray_depths_pt(x, 1.0, 0.0, lfsize,central_u,central_v)
    x_v = transform_ray_depths_pt(x, 0.0, 1.0, lfsize,central_u,central_v)
    x_uv = transform_ray_depths_pt(x, 1.0, 1.0, lfsize,central_u,central_v)
    d1 = (x[:,:,:,1:,1:]-x_u[:,:,:,1:,1:])
    d2 = (x[:,:,:,1:,1:]-x_v[:,:,:,1:,1:])
    d3 = (x[:,:,:,1:,1:]-x_uv[:,:,:,1:,1:])
    l1 = (d1.abs()+d2.abs()+d3.abs()).mean()
    return l1

#spatial TV loss (l1 of spatial derivatives)
#Tested against original tensorflow codes.
def image_derivs_pt(x, nc):
    """
    """
    dy_filter = torch.unsqueeze(torch.unsqueeze(torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],dtype = x.dtype, device = x.device),0),0).repeat(nc,1,1,1)
    dx_filter = torch.unsqueeze(torch.unsqueeze(torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],dtype = x.dtype, device = x.device),0),0).repeat(nc,1,1,1)
    dy = F.conv2d(x, dy_filter, groups=nc)
    dx = F.conv2d(x, dx_filter, groups=nc)
    return dy, dx

def tv_loss_pt(x):
    b_sz, y_sz, x_sz, v_sz, u_sz = x.shape
    
    temp = torch.reshape(x, [b_sz, y_sz, x_sz, u_sz*v_sz])
    temp = temp.permute([0,3,1,2])
    dy, dx = image_derivs_pt(temp, u_sz*v_sz)
    l1 = (dy.abs() + dx.abs()).mean()
    return l1
