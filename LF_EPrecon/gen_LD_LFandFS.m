function [LF_LD, FS_LD] = gen_LD_LFandFS(A_HD,A_LD,img_path, cam_HD,cam_LD, data_format)
import functions.*
%Generating low resolution FS and LF from High resolution one.
% Input:   
%   A_HD, A_LD are fatrix for HD and LD cam, 
%   data_format: 0 ('reconLF' format, for matlab iterative recon) or 1 ('conventional' format, for network training)
% Output:
% If data_format == 1
%   LF_LD: Low resolution light field image in format of  H,W,nv,nu,C in range [0,1]
%   FS_LD: Low resolution FS in shape of  nF,C,H,W
% If data_format == 0
%   LF_LD: Low resolution light field image in format of  nx, ny, nu, nv,C in range [0,1]
%   FS_LD: Low resolution FS in shape of nx,ny,nF,C


lfsize=[372, 540, 7, 7];%H,W,nv,nu
LF_HD = imread(img_path);
LF_HD = LF_HD(1:lfsize(1)*14,1:lfsize(2)*14,:); %H*nv,W*nu,3. After cropping the H,W
LF_HD = permute(reshape(LF_HD,[14,lfsize(1),14,lfsize(2),3]),[5,1,3,2,4]); % C,nv,nu,H,W
LF_HD = LF_HD(:,5:11,5:11,:,:) ;% ,C,7,7,H,W,Selected angular region of size 7 by 7. Using u=4 will have 0 has 0 or 255 at top left corner of SAI. So use u starting from 5. 
LF_HD  = reshape(LF_HD,[1,size(LF_HD)]); % Adding singleton dimension, to 1,C,Nv,Nu,H,W
camLF_HD = processLF_rgbFlower(LF_HD,1, cam_HD);% camLF_HD: nx, ny, nu, nv,3 in reconLF format

camLF_HD = camLF_HD/255; %normalize to [0,1], of shape nx, ny, nu, nv,3

camLF_LD = zeros(cam_LD.arg.nx,cam_LD.arg.ny,cam_LD.arg.nu,cam_LD.arg.nv,3);%LD_LF in reconLF format,of shape nx, ny, nu, nv,3
FS_HD = zeros(cam_HD.arg.nx,cam_HD.arg.ny,numel(cam_HD.arg.focalStackDis),3) ; %nx,ny,nF,3
FS_LD = zeros(cam_LD.arg.nx,cam_LD.arg.ny,numel(cam_LD.arg.focalStackDis),3);%nx,ny,nF,3

% generate HD FS for each color channel
for ic = 1:3
    FS_HD(:,:,:,ic)=A_HD*squeeze(camLF_HD(:,:,:,:,ic));
end

%downsample HD FS to generate LD FS to simulate realistic case
for ic = 1:3
    for iF =1:numel(cam_HD.arg.focalStackDis)
        FS_LD(:,:,iF,ic)=downsample2(FS_HD(:,:,iF,ic),double(int32(cam_HD.arg.nx)/int32(cam_LD.arg.nx)),'warn',0);  %%downsample by 3 times by doing average over the window
        FS_LD(:,:,iF,ic)=FS_LD(:,:,iF,ic)*(cam_HD.arg.nx/cam_LD.arg.nx)*(cam_HD.arg.ny/cam_LD.arg.ny)*...
            (cam_HD.arg.nu/cam_LD.arg.nu)*(cam_HD.arg.nv/cam_LD.arg.nv); %the scaling is needed to have right magnitude
    end
end

%Downsample HD LF to generate LD LF 
for ic = 1:3
    for i = 1:cam_LD.arg.nu %nu/nv_HD and nu/nv_LD should be same for this code to work
        for j = 1:cam_LD.arg.nu
            camLF_LD(:,:,i,j,ic)=downsample2((camLF_HD(:,:,i,j,ic)),double(int32(cam_HD.arg.nx)/int32(cam_LD.arg.nx)),'warn',0);%sum them
        end
    end
end


%{ 
%Only for testing against FS_LD, they should be very similar
    %generate HD FS for each color channel
    %second way of generating lowres FS, this is 'cheating'. In reality will has model mismatch
FS_LD2 = zeros(cam_LD.arg.nx,cam_LD.arg.ny,numel(cam_LD.arg.focalStackDis),3) ; %nx,ny,nF,3
for ic = 1:3
    FS_LD2(:,:,:,ic)=A_LD*squeeze(camLF_LD(:,:,:,:,ic));
end
%}

%return LF_LD, FS_LD in reconLF format if data_format == 0,
%Otherwise else convert them before return.
if data_format == 0
    LF_LD = camLF_LD; %nx, ny, nu, nv,3
    return
end
    
%convert  LF and FS back to conventional orientation
LF_LD=zeros(cam_LD.arg.ny,cam_LD.arg.nx,cam_LD.arg.nv,cam_LD.arg.nu,3);
for ic = 1:3
    for u=1:cam_LD.arg.nu
        for v=1:cam_LD.arg.nv
            LF_LD(:,:,v,u,ic)=rot90(squeeze(camLF_LD(:,:,u,v,ic))',2);
        end
    end
end

FS_LD = flip(flip(permute(FS_LD,[2,1,3,4]),1),2); %H,W,nF,C, Same as for loop method below
FS_LD = permute(FS_LD,[3,4,1,2]); %nF,C,H,W

%{ 
%Method using for loop (slower), same result as above.

FS_LD2 = zeros(185,269,7,3);
for ic= 1:3
    for iF = 1:numel(cam_HD.arg.focalStackDis)
        FS_LD2(:,:,iF,ic) = rot90(FS_LD(:,:,iF,ic)',2); %H,W,nF,C
    end
end
%}


