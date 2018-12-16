function evaluation(dataset)
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

%     model = 'adaptiveCS_resnet_wy_ifusion_ufirst';
%     model = 'csgm'; % reconnet ldamp
    model = 'adaptiveCS_resnet_wy_ifusion_ufirst';
    for cr = [5, 10, 20, 30]
        image_dir = fullfile('../results', dataset, ['cr', num2str(cr)], model, 'test');
        
        image_files = dir(fullfile(image_dir, '*.bmp'));
        
        parfor idx = 1:length(image_files)
            orig_fname = image_files(idx).name;
            if orig_fname(1:4) == 'orig'
                image = imread(fullfile(image_dir, orig_fname));
%                 image = image(3:66, 3:66, :);

                recon_fname = strrep(orig_fname, 'orig', 'recon');
                recon = imread(fullfile(image_dir, recon_fname)); 
%                 recon = recon(3:66, 3:66, :);

                if strcmp(dataset, 'mnist') == 1
%                     image = cat(3, image, image, image);
%                     recon = cat(3, recon, recon, recon);       
                    tmpPSNR1 = psnr(image, recon);
                    tmpSSIM1 = ssim(image, recon);

                    tmpPSNR2 = psnr(image, recon);
                    tmpSSIM2 = ssim(image, recon);
 
                else
                    tmpPSNR1 = calPSNR(image, recon);
                    tmpSSIM1 = calSSIM(image, recon);

                    tmpPSNR2 = getPSNR(image, recon);
                    tmpSSIM2 = getMSSIM(image, recon);
                end

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
        
        save_dir = fullfile('Results', dataset,  model, ['cr', num2str(cr)]);
        if exist(save_dir) ~= 7
               mkdir(save_dir) 
            end
        save(fullfile(save_dir, 'result.mat'), 'PSNR_avg1', 'SSIM_avg1', 'PSNR_avg2', 'SSIM_avg2', 'l1_avg');
    end    
%     delete(gcp('nocreate'))
end


function [psnr_val] = calPSNR(ref, img)
    psnr_val =  psnr(rgb2gray(ref), rgb2gray(img));
end

function [ssim_val] = calSSIM(ref, img)
    ssim_val = ssim(rgb2gray(ref), rgb2gray(img));
end
