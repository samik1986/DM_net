clear all;

% colormap('jet');
% folder = '../DM++/membrane/morseUpdate/results/';
folder = '../DM++/membrane/morseUpdate/stp_data/test/seg/';
dirF = dir(fullfile(folder,'*.tif'));

imgFolder = '../DM++/membrane/morseUpdate/results/';

FN_sum = 0;
FP_sum = 0;
TP_sum = 0;
jacc_sum = 0;
count = 0;

for i = 1 : length(dirF)
    
    mask = imread(fullfile(folder,dirF(i).name))*255;
    gt = imbinarize(mask);
    
    gtIdx = find(gt);
    %     img = imread(fullfile(imgFolder,dirF(i).name));
    img = imread(fullfile(imgFolder,[dirF(i).name(1: end - 10) '_row.tif']));
    
    lbl = imbinarize(img);
    if (length(gtIdx)>1 && length(lblIdx)>1)
        jacc = jaccard(lbl,gt);
    else
        jacc = 1;
    end
        
    if jacc
        count = count +1;
    end
    
    lblIdx = find(lbl);
%     if (lengththgtIdx
    FN = setdiff(gtIdx,lblIdx);
    FP = setdiff(lblIdx,gtIdx);
    TP = intersect(lblIdx,gtIdx);
    
%     P = length(TP) / (length(TP) + length(FP));
%     R = length(TP) / (length(TP) + length(FN));
%     F1 = 2*length(TP) / (2*length(TP) + length(FP) + length(FN));
    %     if length(unique(label))>1
    %         B = labeloverlay(imgN, label,'Colormap','jet', 'Transparency',0.5);
    %     else
    %         B = imgN;
    %     end
    %     imwrite(B, ['results2ImgM/' dirF(i).name]);
    FN_sum = FN_sum + length(FN);
    FP_sum = FP_sum + length(FP);
    TP_sum = TP_sum + length(TP);
    jacc_sum = jacc_sum + jacc;
%     P_sum = P_sum + P;
%     R_sum = R_sum + R;
%     F1_sum = F1_sum + F1;
    
end

P = TP_sum/ (TP_sum + FP_sum);
R = TP_sum/ (TP_sum + FN_sum);
F1 = 2*TP_sum/ (2*TP_sum + FP_sum + FN_sum);
jacc_sim = jacc_sum / count;
