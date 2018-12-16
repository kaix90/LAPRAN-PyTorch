height = [7, 8, 11, 4, 8, 4, 5, 3, 4, 7, 5, 4, 8, 8, 12, 8, 8,  3, 9];
width  = [7, 8, 9,  4, 8, 4  4, 5, 4, 5, 4, 4, 8, 8, 8,  8, 10, 5, 6];
set = [14, 5, 14, 5, 14, 5, 14, 14, 14, 14, 14, 5, 14, 14, 14, 14, 14, 5, 14];

model = 'reconnet'; % csgm, reconnet
dataset = 'bsd500_patch';

for cr = [5, 10, 20, 30]
    image_dir = ['../results/', dataset, '/cr', num2str(cr), '/', model, '/test/'];

%     image_files = dir(fullfile(image_dir, '*.bmp'));
    idx_orig = 0;
    idx_recon = 0;
    for i = 1:length(width)
        save_dir = ['Results/cr', num2str(cr), '/', model, '/set', num2str(set(i))];
         if exist(save_dir) ~= 7
           mkdir(save_dir)
         end
        %% orig
        tmpHeight = [];

        for x = 1:width(i)
            tmpWidth = [];
            for y = 1:height(i)
                    idx_orig = idx_orig + 1;
                    orig_fname = fullfile(image_dir, sprintf('orig_%03d.bmp', idx_orig-1));
                    image = imread(orig_fname);
%                     image = image(3:66, 3:66, :);
                    tmpWidth = [tmpWidth image];
            end
            tmpHeight = [tmpHeight; tmpWidth];
        end
%         maxd = double(max(tmpHeight(:)));
%         mind = double(min(tmpHeight(:)));
%         tmpHeight = uint8((double(tmpHeight) - mind)./(maxd-mind) .* 256);
        imwrite(tmpHeight, [save_dir, '/orig_', num2str(i), '.bmp']);
        tmpHeight = [];
        
        %% recon
        tmpHeight = [];
        for x = 1:width(i)
            tmpWidth = [];
            for y = 1:height(i)
                    idx_recon = idx_recon + 1;
                    orig_fname = fullfile(image_dir, sprintf('recon_%03d.bmp', idx_recon-1));
                    image = imread(orig_fname);
%                     image = image(3:66, 3:66, :);
                    tmpWidth = [tmpWidth image];
            end
            tmpHeight = [tmpHeight; tmpWidth];
        end
%         maxd = double(max(tmpHeight(:)));
%         mind = double(min(tmpHeight(:)));
%         tmpHeight = uint8((double(tmpHeight) - mind)./(maxd-mind) .* 256);
        imwrite(tmpHeight, [save_dir, '/recon_', num2str(i), '.bmp']);
        tmpHeight = [];
    end
end
