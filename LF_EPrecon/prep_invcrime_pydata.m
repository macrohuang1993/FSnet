%Script for generating low resolution FS and LF h5  dataset for training
%neural networks considering inverse crime.
% To run: Check the dispRange, resolution, LF_filename and FS_filename.
% Check the lfsize in gen_LD_LFandFS.
clear 
clc
import classes.*
import functions.*
import utilities.*

LFimg_folder = '/home/zyhuang/WD/FSnet_flower/Flowers_8bit/';
LF_filename = 'LF_LDs.h5';
FS_filename = 'FS_LDs.h5';
f=100; %(mm) 
focusingDis=22*1000; %(mm) object side distance the LF is focusing at
refDis = 1/(1/f-1/focusingDis); % conjuagte of the focusingDis (image side)

dispRange = linspace(-1,0.3,7);
nxnynunv_HD=num2cell([539,371,7,7]);
dxdydudv_HD=num2cell([0.06836,0.06836,100,100]);
[nx_HD,ny_HD,nu_HD,nv_HD]=deal(nxnynunv_HD{:});
[dx_HD,dy_HD,du_HD,dv_HD]=deal(dxdydudv_HD{:});


nxnynunv_LD=num2cell([269,185,7,7]);
dxdydudv_LD=num2cell([0.06836*2,0.06836*2,100,100]);
[nx_LD,ny_LD,nu_LD,nv_LD]=deal(nxnynunv_LD{:});
[dx_LD,dy_LD,du_LD,dv_LD]=deal(dxdydudv_LD{:});

fsDis = setFocalStackDis([], [], 'disp_list', f,dx_HD,du_HD,focusingDis,refDis,dispRange);% fsDis: image side focal stack distance

%%%%ilyong (18-10-27): imaging system in high dimension
argCam_HD = { 'nx', nx_HD,  'ny', ny_HD,  'dx', dx_HD, 'dy', dy_HD, ...         % Sensor plane
    'nu', nu_HD,  'nv', nv_HD,   'du', du_HD, 'dv', dv_HD, ...              % Aperture plane
    'focalLen', f, 'apeSize', 500000, 'focalStackDis', fsDis, 'refDis', refDis};   % Camera settings   
cam_HD = Camera( argCam_HD{:} );

%%%%ilyong (18-10-27): imaging system in lower dimension (for recon.)
argCam_LD = { 'nx', nx_LD,  'ny', ny_LD,  'dx', dx_LD, 'dy', dy_LD, ...         % Sensor plane
    'nu', nu_LD,  'nv', nv_LD,   'du', du_LD, 'dv', dv_LD, ...              % Aperture plane
    'focalLen', f, 'apeSize', 500000, 'focalStackDis', fsDis, 'refDis', refDis};   % Camera settings   
cam_LD = Camera( argCam_LD{:} );  


A_HD = GfocalStack('camera', cam_HD);		%for high res
A_LD = GfocalStack('camera', cam_LD);	%for low res (recon.)


% list all LF names and create train/val name list
folder_struct2cell = struct2cell(dir(LFimg_folder));
namelist = folder_struct2cell(1,3:end);%excluding . and ..
rng(100) %deterministic name list
idx = randperm(numel(namelist));
train_namelist =  namelist(idx(1:(numel(namelist)-100)));
val_namelist = namelist(idx(end-99:end));


%Generate trainset,save LD_FS to h5f.[train] and save LF_LD to [lfname]
for i = 1:numel(train_namelist)
    LF_name = train_namelist{i};
    
[LF_LD, FS_LD] = gen_LD_LFandFS(A_HD,A_LD,...
    fullfile(LFimg_folder, LF_name),cam_HD,cam_LD,1);
%Compensating for ordering diff in python and matlab
LF_LD = permute(LF_LD,[4,3,2,1,5]);%nu,nv,W,H,3
FS_LD = permute(FS_LD,[4,3,2,1]); %W,H,C,nF

%write LF and FS data
h5create(LF_filename,fullfile('/',LF_name),size(LF_LD),'Datatype','single');
h5write(LF_filename,fullfile('/',LF_name),single(LF_LD));
h5create(FS_filename,fullfile('/','train',LF_name),size(FS_LD),'Datatype','single');
h5write(FS_filename,fullfile('/','train',LF_name),single(FS_LD));

disp(i)
end


%Generate valset
for i = 1:numel(val_namelist)
     LF_name = val_namelist{i};
    
[LF_LD, FS_LD] = gen_LD_LFandFS(A_HD,A_LD,...
    fullfile(LFimg_folder, LF_name),cam_HD,cam_LD,1);
%Compensating for ordering diff in python and matlab
LF_LD = permute(LF_LD,[4,3,2,1,5]);%nu,nv,W,H,3
FS_LD = permute(FS_LD,[4,3,2,1]); %W,H,C,nF

%write LF and FS data
h5create(LF_filename,fullfile('/',LF_name),size(LF_LD),'Datatype','single');
h5write(LF_filename,fullfile('/',LF_name),single(LF_LD));
h5create(FS_filename,fullfile('/','val',LF_name),size(FS_LD),'Datatype','single');
h5write(FS_filename,fullfile('/','val',LF_name),single(FS_LD));

disp(i)
end


