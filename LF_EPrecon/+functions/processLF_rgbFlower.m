function camLF= processLF_rgbFlower(dataset,sample_idx, cam)
%Select one sample of  LF in shape N,C,Nv,Nu,H,W and reformat to reconLF format of nx, ny, nu, nv,C
% dataset: the LF dataset exported from python, in the format of
% N,C,Nv,Nu,H,W
% sample_idx: which sample among dataset to select
% return camLF: nx, ny, nu, nv,3
import functions.*;
import utilities.*;


%sceneLF = LF.LF; ,%Nv,Nu,H,W,C
sceneLF = squeeze(dataset(sample_idx,:,:,:,:,:)); %C,Nv,Nu,H,W
sceneLF = permute(sceneLF,[2,3,4,5,1]); %Nv,Nu,H,W,C
nx = cam.arg.nx; ny = cam.arg.ny; nu = cam.arg.nu; nv = cam.arg.nv;
camLF = zeros(nx, ny, nu, nv,3);
apeMask = cam.arg.apeMask;

for ic = 1:3
    for iu = 1:nu
        for iv = 1:nv
            if apeMask(iu, iv)
                %img = rgb2gray(squeeze(sceneLF(iu, iv, :, :, :)));
                img = squeeze(sceneLF(iv,iu,:,:,ic));
                camLF(:, :, iu, iv,ic) = rot90(img(1:end-1, 1:end-1), 2).';

            end
        end
    end
end
end
