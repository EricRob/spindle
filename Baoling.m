clear;
list = dir('R:\jwanglab\jwanglabspace\Param\data\Baoling_Share_EEG_data\Animal_3\19*');
list = {list.name};
EEG_19 = [];
for file = 1:length(list)
    load(['R:\jwanglab\jwanglabspace\Param\data\Baoling_Share_EEG_data\Animal_3\' list{file}]);
    EEG_19 = [EEG_19; EEG];
    clear EEG;
end

%% 

EEG = EEG_19;

EEG = resample(EEG, 200, 1000);
EEG = bandpass(EEG,'bandpass', [2, 50]);

%%
output = zeros(size(EEG));

save(['data/pain_sleep/Baoling_Animal3.mat'], 'EEG');

h1 = fopen(['RatData/test_data/pain_sleep/Baoling_Animal3_bp2_50.txt'], 'wt');  
h2 = fopen(['RatData/test_data/pain_sleep/Baoling_Animal3_bp2_50_labels.txt'], 'wt');  
 
fprintf(h1, '%f\n', EEG);
fprintf(h2, '%d\n', output);

fclose all;