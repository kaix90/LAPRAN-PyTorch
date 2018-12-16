clear all;
close all;
folder = '../data/smallDataset/291';
savepath = '../data/smallDataset/trainImages/';

filepaths = [dir(fullfile(folder, '*.jpg'));dir(fullfile(folder, '*.bmp'))];
     
for i = 1 : length(filepaths)
  filename = filepaths(i).name;
	[pathstr,imName,ext] = fileparts(filename);
  image = imread(fullfile(folder, filename));
	image = im2double(image);
  
  for angle = 0 : 90 : 90	
		imRotate = imrotate(image, angle);
    imwrite(imRotate, [savepath imName, '_rot' num2str(angle) '.bmp']);
		imFlip = fliplr(imRotate);
		imwrite(imFlip, [savepath imName, '_flip' num2str(angle) '.bmp']);
	end
end
