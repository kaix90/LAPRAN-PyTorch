function evaluation_bsd500
    psnr_val1 = [];
    ssim_val1 = [];
    psnr_val2 = [];
    ssim_val2 = [];
    l1_val = [];
    result = {};
    
%     poolobj = gcp('nocreate');
%     if isempty(poolobj)
%         parpool(10);
%     end
    
    model = 'reconnet';
    % model = 'adaptiveCS_resnet_wy_ifusion_ufirst'; % csgm, reconnet
    for set = [5, 14]
        for cr = [5, 10, 20, 30]
            image_dir = ['/home/user/kaixu/myGitHub/CSImageNet/matlab/Results/cr', num2str(cr), '/', model, '/set', num2str(set)];

            image_files = dir(fullfile(image_dir, '*.bmp'));

            for idx = 1:length(image_files)
                orig_fname = image_files(idx).name;
                if orig_fname(1:4) == 'orig'
                    image = imread(fullfile(image_dir, orig_fname));

                    recon_fname = strrep(orig_fname, 'orig', 'recon');
                    recon = imread(fullfile(image_dir, recon_fname)); 
                    tmpPSNR1 = calPSNR(image, recon);
                    tmpSSIM1 = calSSIM(image, recon);

                    tmpPSNR2 = getPSNR(image, recon);
                    tmpSSIM2 = getMSSIM(image, recon);

                    psnr_val1(idx) = tmpPSNR1;
                    ssim_val1(idx) = tmpSSIM1;

                    psnr_val2(idx) = tmpPSNR2;
                    ssim_val2(idx) = tmpSSIM2;

                    tmpL1 = sum(abs(image(:)-recon(:))) / (64*64*3);
                    l1_val(idx) = tmpL1;

                    disp(['processing image: ', num2str(idx)])
                    disp(['psnr1: ', num2str(tmpPSNR1)])
                    disp(['ssim1: ', num2str(tmpSSIM1)])
                    disp(['psnr2: ', num2str(tmpPSNR2)])
                    disp(['ssim2: ', num2str(tmpSSIM2)])
                    disp(['l1: ', num2str(tmpL1)])
                end
            end
            result.psnr_val1 = psnr_val1;
            result.ssim_val1 = ssim_val1;
            result.psnr_val2 = psnr_val2;
            result.ssim_val2 = ssim_val2;
            result.l1_val = l1_val;
            PSNR_avg1 = mean(result.psnr_val1);
            SSIM_avg1 = mean(result.ssim_val1);
            PSNR_avg2 = mean(result.psnr_val2);
            SSIM_avg2 = mean(result.ssim_val2);
            l1_avg = mean(result.l1_val);

            save_dir = fullfile('Results', ['cr', num2str(cr)], model, ['set', num2str(set)]);
            if exist(save_dir) ~= 7
               mkdir(save_dir) 
            end
            save([save_dir, '/result.mat'], 'PSNR_avg1', 'SSIM_avg1', 'PSNR_avg2', 'SSIM_avg2', 'l1_avg');
        end
    end
    %delete(gcp('nocreate'))
end


function [psnr_val] = calPSNR(ref, img)
    psnr_val =  psnr(rgb2gray(ref), rgb2gray(img));
end

function [ssim_val] = calSSIM(ref, img)
    ssim_val = ssim(rgb2gray(ref), rgb2gray(img));
end
