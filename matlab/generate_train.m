clear;close all;
%% settings
%folder = '/home/user/kaixu/myGitHub/CSImageNet/data/BSDS500/val/1';
folder = '/home/user/kaixu/myGitHub/datasets/SISR/Train_291';
size_input = 64;
size_label = 64;
stride = 32;

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label, 1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));

for i = 1 : length(filepaths)
    image = imread(fullfile(folder,filepaths(i).name));
    [path, name, ext] = fileparts(filepaths(i).name);
%     image = rgb2ycbcr(image);
%     image = im2double(image(:, :, :));

%     im_label = modcrop(image, scale);
    im_label = image;
    [hei,wid,~] = size(im_label);

    for x = 1 : stride : hei-size_label+1
        for y = 1 :stride : wid-size_label+1
            if x+size_label-1 <= hei && y+size_label-1 <= wid
                subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
            end
            imwrite(subim_label, fullfile('../data/train_64x64', 'data', strcat(name, '_', num2str(count), '.bmp')));

            count=count+1;
        end
    end
    count = 0;
end
