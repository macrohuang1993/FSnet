%For reconstructign light field using edge preserving regualrization
%Hyperparameter looping version

clear all; close all; clc;
import classes.*
import functions.*
import utilities.*
%% Setting here
LF_datapath = 'LF_LDs_matrecon.h5';
FS_datapath = 'FS_LDs_matrecon.h5';
save_folder = 'EP_hyperparameter_search_newrun';
N_val = 100; % number of val sample to validate. (<= numel(name_list))
%% Camara setting and Generate model Matrix
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

%%%%ilyong (18-10-27): imaging system in lower dimension (for recon.)
argCam_LD = { 'nx', nx_LD,  'ny', ny_LD,  'dx', dx_LD, 'dy', dy_LD, ...         % Sensor plane
    'nu', nu_LD,  'nv', nv_LD,   'du', du_LD, 'dv', dv_LD, ...              % Aperture plane
    'focalLen', f, 'apeSize', 500000, 'focalStackDis', fsDis, 'refDis', refDis};   % Camera settings   
cam_LD = Camera( argCam_LD{:} );  
A = GfocalStack('camera', cam_LD);	%for low res (recon.)
%%
info_h5 = h5info(LF_datapath);
info_h5 = struct2cell(info_h5.Datasets);
name_list = info_h5(1,:);

%Parameterss
niterTV = 30;
%alphaTV = logspace(5,7,20);%1e4, 2.5e4, 5e4, 7.5e4, reg strength parameter.
%deltaTV = logspace(-2,1,20);%1e-2, 1e-3, 1e-4; sharpness of transition from hyperbola to L1loss, small means sharp.
%config=combvec(alphaTV,deltaTV);
config = [1.6e5;3.8e-1];%[1.6e5;3.8e-1];
[~,Nconfig]=size(config);

info_all = cell(3*N_val,Nconfig);% Each column is the info (of one config) for r,g,b channel of each sample, 
reconLF_all = zeros(nx_LD,ny_LD,nu_LD,nv_LD,3,N_val);
Final_PSNR_lists = zeros(2+3*N_val,Nconfig); % alpha,delta, [psnr(r), psnr(g), psnr(b) for each sample]
Final_PSNR_lists(1:2,:) = config;



for idx = 1:N_val
    LF_name = name_list{idx};
    [~,LF_name_noext,~]= fileparts(LF_name);
    mkdir(fullfile(save_folder,LF_name_noext))
    %% Load LF and FS and create fatrix A
    FS = h5read(FS_datapath,fullfile('/','val',LF_name));%nx,ny,nF,C
    LF = h5read(LF_datapath,fullfile('/',LF_name)); %nx,ny,nu,nv,C
    %Dimensions
    [Nx, Ny, Nu, Nv,~] = size(LF);
    %Initial lightfield
    LFrecon0 = A' * FS; % Operation is broadcasted across color channel.
    LFrecon0 = LFrecon0/max(LFrecon0(:));
    %{
% same as  result above.
LFrecon1 = zeros(size(LF));
for ic= 1:3
    LFrecon1(:,:,:,:,ic) = A'* FS(:,:,:,ic);
end
LFrecon1 = LFrecon1/max(LFrecon1(:));
    %}
    
    for i =1:Nconfig
        %Regularizer: First-order finite difference (hyper3)
        pot_arg = {'hyper3', config(2,i)};
        R = Reg1(true(Nx,Ny,Nu,Nv), 'beta', config(1,i), 'pot_arg', pot_arg, 'offsets', [1 Nx Nx*Ny Nx*Ny*Nu]);
        % loop color channels
        for ic = 1:3
            %TV recon
            display('TV Recon begins.');
            tic;
            [LFreconTV,info] = pwls_pcg1(vec(LFrecon0(:,:,:,:,ic)), A, 1, vec(FS(:,:,:,ic)), R, 'niter', niterTV, 'userfun', @mystatus, 'userarg', {A,vec(FS(:,:,:,ic)),LF(:,:,:,:,ic),1}, 'isave', 'last' );
            fig1=figure(1);
            plot(cell2mat(info(:,2)))
            title(sprintf('peakPSNR is %.2f',max(cell2mat(info(:,2)))))
            fig2=figure(2);
            plot(cell2mat(info(:,1)))
            title('FSerr')
            %LFreconTV = max(reshape(LFreconTV, size(LFrecon0)), 0);
            saveas(fig1,fullfile(save_folder, LF_name_noext, sprintf('%s_c_%d_PSNR_plot_alpha%.2e_delta %.2e.png',...
                LF_name_noext,ic,config(1,i),config(2,i))))
            saveas(fig2,fullfile(save_folder, LF_name_noext, sprintf('%s_c_%d_FS_err_alpha%.2e_delta %.2e.png',...
                LF_name_noext,ic,config(1,i),config(2,i))))
            saveas(figure(3),fullfile(save_folder, LF_name_noext, sprintf('%s_c_%d_reconsubaperture at (3,3)_alpha%.2e_delta %.2e.png',...
                LF_name_noext,ic,config(1,i),config(2,i))))
            
            Final_PSNR_lists(3*idx+ic-1,i) = info{end,2}; %final psnr 
            reconLF_all(:,:,:,:,ic,idx) = reshape(LFreconTV,size(LF(:,:,:,:,ic)));% use this only for single config
            info_all{3*(idx-1)+ic,i} = cell2mat(info);
        end
    end
    Final_PSNR_color(idx) = my_psnr(reconLF_all(:,:,:,:,:,idx),LF,1)% use this only for single config, otherwise only the last config PNSR is calculated
end
%generate PSNR (averaged over val samples) vs time
PSNRvsTime = genTimingdata(info_all,niterTV);


function psnrval = my_psnr(I, ref, peakval )
err = (norm(I(:)-ref(:),2).^2) / numel(I);
psnrval = 10*log10(peakval.^2/err);
psnrval = double(psnrval); %trick for convert info cell2mat
end
function info=mystatus(reconLF,iter,A,trueFS,trueLF,peakval)
psnrval = my_psnr(reconLF, trueLF, peakval );
FS_error=A*reconLF-trueFS;
FS_error=norm(FS_error);
time = toc;

fig3=figure(3);
reconLF=reshape(reconLF, size(A.imask));
imagesc(squeeze(reconLF(:,:,3,3)))
title('reconSubaperture\_img')
drawnow
sprintf('PSNR is:%.2f,FSerr is:%.1f', psnrval,FS_error)
info={FS_error,psnrval,time};
end
function info=mystatus2(reconLF,iter,A,trueFS,trueLF,peakvals)
time = toc;
FS_error = 0;
psnrval = 0;
info={FS_error,psnrval,time};
end

function [flattened]=vec(x)
flattened = x(:);
end 
function [PSNRvsTime] = genTimingdata(info_all,niterTV)
% Generate a Average PSNR versus time data. 
%Note the PSNR is averaged from 3 color channels at each iter, this is only
%an approx (qualitatively good enough, ~+-0.3dB), need to implement the accurate one if needed. 
PSNRvsTime = zeros(niterTV,2);
for iter = 1:niterTV
    t= 0;
    psnr = 0;
    for i = 1:numel(info_all)
        info = info_all{i};
        t = t + info(iter,3);
        psnr = psnr + info(iter,2);
    end
    t = t/(numel(info_all)/3); % time of each color channel is added, not averaged
    psnr = psnr/numel(info_all);
    PSNRvsTime(iter,1) = t;
    PSNRvsTime(iter,2) = psnr;
end

end