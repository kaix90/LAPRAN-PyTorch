
function generate_test(dataset)
    %% settings
    if strcmp(dataset, 'set5')
        folder = '/home/user/kaixu/myGitHub/datasets/SISR/Set5';
        savedir = '../data/test_set5_64x64/data';
    elseif strcmp(dataset, 'set14')
        folder = '/home/user/kaixu/myGitHub/datasets/SISR/Set14';
        savedir = '../data/test_set14_64x64/data';
    end
    
    if not(exist(savedir) == 7)
        mkdir(savedir)
    end
        
    size_input = 64;
    size_label = 64;
    stride = 64;

    %% initialization
    data = zeros(size_input, size_input, 1, 1);
    label = zeros(size_label, size_label, 1, 1);
    padding = abs(size_input - size_label)/2;
    count = 0;

    %% generate data
    filepaths = dir(fullfile(folder,'*.bmp'));
    
    horizontal_idx = 0;
    vertical_idx = 0;
    for i = 1 : length(filepaths)
        image = imread(fullfile(folder,filepaths(i).name));
        [path, name, ext] = fileparts(filepaths(i).name);
    %     image = rgb2ycbcr(image);
    %     image = im2double(image(:, :, :));

    %     im_label = modcrop(image, scale);
        im_label = image;
        [hei,wid,~] = size(im_label);

        for x = 1 : stride : hei-size_label+1
            horizontal_idx = horizontal_idx + 1;
            for y = 1 :stride : wid-size_label+1
                vertical_idx = vertical_idx + 1;
                if x+size_label-1 <= hei && y+size_label-1 <= wid
                    subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                end
                
                imwrite(subim_label, fullfile(savedir, strcat(name, '_', num2str(horizontal_idx), ...
                    '_', num2str(vertical_idx), '.bmp')));

                count=count+1;
            end
        end
    end
end
