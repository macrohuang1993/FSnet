function camLF = importLF(dataset, scene, cam)
import functions.*;
import utilities.*;
printf('Import LF from dataset...');

libFolder = 'LF Dataset';

load(fullfile(libFolder,dataset, [scene '.mat']));
sceneLF = LF.LF;
%{
            figure; imagesc(LF.depth_lowres);
            min(min(LF.depth_lowres))
            max(max(LF.depth_lowres))
%}
nx = cam.arg.nx; ny = cam.arg.ny; nu = cam.arg.nu; nv = cam.arg.nv;
camLF = zeros(nx, ny, nu, nv);
apeMask = cam.arg.apeMask;

for iu = 1:nu
    for iv = 1:nv
        if apeMask(iu, iv)
            img = rgb2gray(squeeze(sceneLF(iu, iv, :, :, :)));
            %camLF(:, :, iv, iu) = rot90(img(1:2:end-2, 1:2:end-2), 2).';
            camLF(:, :, iv, iu) = rot90(img(1:end-1, 1:end-1), 2).';
            
        end
    end
end





end
