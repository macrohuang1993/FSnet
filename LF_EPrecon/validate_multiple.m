% for running main on multiple validation samples at once. 
%%settings below

import classes.*
import functions.*
import utilities.*

%%
for sample_idx = 1:5
% f = 50;
f=100;
dx =0.06836;
dy = 0.06836;
du = 100;
dv = 100;
nu = 7;
nv = 7;
nx = 539;
ny = 371;
focusingDis = 22000; % object side distance the lens is focusing at
refDis = 1/(1/f-1/focusingDis); % conjuagte of the focusingDis (image side)
dispRange = linspace(-1,0.3,7);
Alpha = 5e-6; %about the half the maximam alpha that diverge the cost
nIter = 100;
% workRange = [19.7158 25.5545]*100;
%workRange = [20 25]*1000;
%fsDis = setFocalStackDis(workRange, 5, 'even Fourier slice',f);
%fsDis = setFocalStackDis(workRange, 4, 'linear');
fsDis = setFocalStackDis([], [], 'disp_list', f,dx,du,focusingDis,refDis,dispRange);% fsDis: image side focal stack distance
argCam = { 'nx', nx,  'ny', ny,  'dx', dx, 'dy', dy, ...         % Sensor plane
    'nu', nu,   'nv',nv,   'du', du,  'dv', dv, ...              % Aperture plane
    'focalLen', f, 'apeSize', 160000, 'focalStackDis', fsDis, 'refDis', refDis};   % Camera settings   1/(1/f-1/2200)
cam = Camera( argCam{:} );


save_name=strcat('flowerLF_val_',num2str(sample_idx));
%trueLF = importLF('4D LF Benchmark',   importdata_name, cam);%% this is the LF that has been masked by the aperture
dataset = load("val_LF.mat");%load variable 'LF' and copy to variable dataset, LF has shape N,C,Nv,Nu,H,W
dataset = dataset.LF;
trueLF_rgb=processLF_rgbFlower(dataset,sample_idx, cam); %note imported LF is already normalized
%trueLF = trueLF/max(trueLF(:)); 
reconLF_rgb = zeros(size(trueLF_rgb));

%% Generate the forward model (fatrix A) and focal stack images
A = GfocalStack('camera', cam);

str = 'rgb';
%mkdir(fullfile('Result',save_name,str(1)))
%mkdir(fullfile('Result',save_name,str(2)))
%mkdir(fullfile('Result',save_name,str(3)))
for ic = 1:3
    
    printf(strcat('Processing Channel: ',str(ic)))
    trueLF = trueLF_rgb(:,:,:,:,ic);
    trueFS = A*trueLF;

    %save_name = strcat(Save_name,'_',str(ic));
    dirName = ['Result/',save_name,'/',str(ic),'/'];
    objTrueFS = FocalStack('camera', cam, 'focalStack', trueFS, 'focalStackDis', fsDis, 'name', 'True Focal Stack');
    fsFigs = objTrueFS.plotFS();
    saveFig(fsFigs, string(fsDis), dirName);
    
    
    [reconLF, errorplot] = reconGD2(trueFS, A, trueLF,Alpha,nIter);
    reconLF_rgb(:,:,:,:,ic) = reconLF;
    
    saveas(errorplot,fullfile('Result',save_name,str(ic),'error.fig'));
    saveFig(errorplot, "error", dirName);
    
    
    %% Plots
    i1_u=(size(reconLF,4)+1)/2;  %%it's the 4th dimension because reconLF has transposed u,v
    i1_v=1;
    i1 = sub2ind([size(reconLF,4) size(reconLF,3)], i1_v, i1_u); %% i1_v and i1_u are filiped in postion because reconLF has transposed u,v
    I1 = reconLF(:, :, i1);
    fig1=figure;
    imgDisplay(I1);
    title('reconView at i1');
    saveas(fig1,fullfile('Result',save_name,str(ic),'reconView at i1.jpg'));
    
    cv = cenView(reconLF);
    cv = cv/max(cv(:));
    fig2=figure; imgDisplay(cv);
    saveas(fig2,fullfile('Result',save_name,str(ic),'reconView at center.jpg'));
    
    i2_u=(size(reconLF,4)+1)/2;  %%it's the 4th dimension because reconLF has transposed u,v
    i2_v=size(reconLF,3);
    i2 = sub2ind([size(reconLF,4) size(reconLF,3)], i2_v, i2_u);  %% i1_v and i1_u are filiped in postion because reconLF has transposed u,v
    I2 = reconLF(:, :, i2);
    fig3=figure;
    imgDisplay(I2);
    title('reconView at i2');
    saveas(fig3,fullfile('Result',save_name,str(ic),'reconView at i2.jpg'));
    
    shift = 13.3;
    i2 = sub2ind([size(reconLF,4) size(reconLF,3)], i2_v, i2_u);
    img = reconLF(:, :, i2);
    [X, Y] = ndgrid(1:size(img, 1), 1:size(img, 2));
    I2 = interpn( X, Y, img, X-shift, Y, 'linear');

end

psnr = my_psnr(reconLF_rgb,trueLF_rgb,1);
err = reconLF_rgb - trueLF_rgb;
L1error = mean(abs(err(:)));
save(fullfile('Result',save_name,'result.mat'),'trueLF_rgb','reconLF_rgb','psnr','L1error','dispRange','nu','nv','nx','ny','Alpha','nIter');

clear
close all; pause(0);
clear vars;
clc;
end