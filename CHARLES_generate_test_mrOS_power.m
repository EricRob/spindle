function [] = generate_test_mrOS_power(fileNum, c4_std)

addpath(genpath('chronux_2_11'));

[~, records] = quickRead(fileNum, 'mros');
fs = 256;
target_fs = 200; % Change target_fs to resample to a different rate.

% This function removes all non-stage 2 sleep from the data, as derived from the annotation files
stageTwoRecords = isolateStage(2, fileNum, records, fs);
stageTwoRecords=resample(stageTwoRecords' , target_fs, fs)';
%stageTwoRecords=stageTwoRecords(:, 1:240000); % Truncating records to create abridged file, COMMENT OUT LATER.
fs = target_fs;

format = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
     
disp("Writing files, this may take a while.")

%0018
%c3_std = 8.8075; % Standard deviation values for spindle amplitudes
%c4_std = 8.3054; 

%0003_C4: 21.7939
%c3_std = 21.7939;
%c4_std = 21.7939; 

%0006_C4: 14.1465
%c3_std = 14.1465;
%c4_std = 14.1465; 

%0009_C4: 16.3501
%0012_C4: 20.4116
%0013_C4: 21.7184
%0014_C4: 21.0747

c3_std = c4_std;

%%%%%%% First channel

database = cell(1,1);
database{1} = stageTwoRecords(4, :)'; % Convert to column vector

movingwin=[1 1/fs];
params.Fs=fs;
params.fpass=[0 25];
params.tapers=[3 5];
params.err=0;
params.pad=0;
ratio = zeros(size(database{1}));
power_BB = zeros(size(database{1}));
power_spindle = power_BB;
band_pass = power_BB;
up_env = power_BB;
power_feat = zeros(size(database{1}, 1), 19);
for k=1:length(database{1})-floor(fs)+1
    seq = database{1}(k:k+floor(fs)-1);
    seq = (seq-mean(seq))/c3_std;
    [SS,tt,ff]=mtspecgramc(seq,movingwin,params);
    energyDB = 10*log10(SS);
    %     DB_LF = mean(energyDB(:,3:8),2);
    DB_BB = mean(energyDB(:,4:11),2);
    DB_spindle = mean(energyDB(:,12:22),2);
    power_BB(k+floor(fs)/2)=DB_BB;
    power_spindle(k+floor(fs)/2)=DB_spindle;
    ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
    bdpass = bandpass(seq, 'bandpass',[9,16]);
    band_pass(k+floor(fs)/2)=bdpass(floor(fs)/2+1);
    [up,~]=envelope(bdpass);
    up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
    power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
end

database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
sizeOfData = size(band_pass);
sizeOfData = sizeOfData(1);
dummyLabels = zeros(sizeOfData, 1);

num_step = floor((250*0.001)/(1/fs));

outFile=fopen(['test_data/test_' fileNum '_C3_' num2str(num_step) '_100%_f' num2str(target_fs) '_mf.txt'] , 'wt');
fprintf(outFile,format,database{1}');

fclose(outFile);

% We need to have labels to run the NN, but don't have any, so here we create a file with the
% correct dimensions to feed to the algorithm.e
outFile=fopen(['test_data/test_' fileNum '_C3_' num2str(num_step) '_100%_f' num2str(target_fs) '_mf_labels.txt'] , 'wt');
fprintf(outFile, '%d\n' , dummyLabels);
fclose(outFile);

%%%%%%%%% Next Channel %%%%%%%%%
% 
% disp("First channel finished, writing second file...");
% 
% database = cell(1,1);
% database{1} = stageTwoRecords(5, :);
% 
% ratio = zeros(size(database{1}));
% power_BB = zeros(size(database{1}));
% power_spindle = power_BB;
% band_pass = power_BB;
% up_env = power_BB;
% power_feat = zeros(size(database{1}, 1), 19);
% for k=1:length(database{1})-floor(fs)+1
%     seq = database{1}(k:k+floor(fs)-1);
%     seq = (seq-mean(seq))/c3_std;
%     [SS,tt,ff]=mtspecgramc(seq,movingwin,params);
%     energyDB = 10*log10(SS);
%     %     DB_LF = mean(energyDB(:,3:8),2);
%     DB_BB = mean(energyDB(:,4:11),2);
%     DB_spindle = mean(energyDB(:,12:22),2);
%     power_BB(k+floor(fs)/2)=DB_BB;
%     power_spindle(k+floor(fs)/2)=DB_spindle;
%     ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
%     bdpass = bandpass(seq, 'bandpass',[9,16]);
%     band_pass(k+floor(fs)/2)=bdpass(floor(fs)/2+1);
%     [up,~]=envelope(bdpass);
%     up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
%     power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
% end
% database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
% 
% outFile=fopen(['test_data/visit1-aa' fileNum '_C4_mf.txt'] , 'wt');
% 
% fprintf(outFile,format,database{1}');
% disp("Finished writing.")
% fclose(outFile);



end % function