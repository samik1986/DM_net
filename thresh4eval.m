clear all;

% colormap('jet');
folder = '../DM++/membrane/morseUpdate/results/';
% folder = '../DM++/membrane/morseUpdate/stp_data/test/seg/';
dirF = dir(fullfile(folder,'*.tif'));

imgFolder = '../DM++/membrane/morseUpdate/stp_data/test/img/';

for i = 1 : length(dirF)
    
    mask = imread(fullfile(folder,dirF(i).name));
    label = imbinarize(mask);
    img = imread(fullfile(imgFolder,dirF(i).name));
%     img = imread(fullfile(imgFolder,[dirF(i).name(1: end - 10) '_row.tif']));
    
    imgN = imadjust(img);
    if length(unique(label))>1
        B = labeloverlay(imgN, label,'Colormap','jet', 'Transparency',0.5);
    else
        B = imgN;
    end
    imwrite(B, ['results2ImgM/' dirF(i).name]);
    
end
