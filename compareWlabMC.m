% Written by Charles Zhang
% July 2018

% Given the variable component of the filenames specifying the trial/experiment to
% be analyzed, runs binaryKappa and getSpindleStats functions to save all relevant
% statistics to a .mat file.
% Compares Wlab results and McSleep results

% Input:
%   name -- Variable component of filename, e.g. "mros0003".
% Assumed Files:
%   /mcsleep/mcsleep_name.mat -- Containing binary output of mcsleep algorithm
%   probability_name.txt -- Containing probability output of wanglab nn, to be processed
%   /trimmedData/data_name.mat -- Containing the single channel of EDF NN was run on,
%       preprocessed to 200 Hz and only stage 2
% Output:
%   No variables returned, the following variables will be in the file labeled "name_stats.mat":
%   po, pe, k sek, ci, km, fourSquare, detectionMat -- See binaryKappa documentation
%   avgFreqs -- 2x2 matrix structured similarly to fourSquare with information about frequency
%       of spindles in each category
%   avgDur -- 2x2 matrix structured similarly to fourSquare with information about duration
%       of spindles in each category
%   avgPow -- 2x2 matrix structured similarly to fourSquare with information about power
%       of spindles in each category
%   freqMat/durMat/powMat + Agree/1/2 -- Individual scores for frequency, duration, power
%       corresponding to the spindles in each category.

addpath(genpath('chronux_2_11'));
%function [] = compareWlabMC(name, study, probThresh, timeThresh)
clearvars;

num_subject = {1,2,3,4,5,6,7,8};
session1 = {};
for i=1:length(num_subject)
    session1={session1{:},['excerpt' num2str(num_subject{i})]};
end
fs = 200;
%%
%name = 'Rat1_080618';
%study = '';



%load(strcat("./mcsleep/", study, name, ".mat")); % spindles
%f2=fopen(['RatData/test_data/Dino_' name '_mPFC_bp2_50_labels.txt']);\

% MC SLEEP DATA
% load(['/media/Share/jwanglab/jwanglabspace/Charles/Spindles/Matlab/mcsleep/' name '.mat']);
%     f2=fopen(['RatData/test_data/Earth10_71118_bp2_50.txt']);
%     spindles=fscanf(f2,'%f');
%     fclose(f2);
%wprob = load(strcat("probability_", study, name, "_C3.txt")); % wprob
%load(strcat("./trimmedData/data_", study, name, ".mat")); % data
%f1=fopen(['RatData/NN_output/probability_' name '.txt']);

% NEURAL NETWORK OUTPUT
% f1=fopen(['RatData/NN_output/probability_' name '.txt']);
% wprob=fscanf(f1,'%f');
% fclose(f1);
%load(['/media/share/jwanglab/jwanglabspace/David Rosenberg/Earth10/7_11_2018/LFP_1/Earth10_5-22-2018_250_WithLFP.mat']);

% RAT RECORDING DATA
% f2 = fopen(['RatData/test_data/' name '_bp2_50.txt']);
% EEG = fscanf(f2, '%f');
% fclose(f2);
% data = EEG;

for idx1 = 1:length(session1)  
    
    s1 = strsplit(session1{idx1}, '-');
    disp(session1{idx1});
    %load(strcat("./mcsleep/", study, name, ".mat")); % spindles
    load(['./results/mcsleep/', session1{idx1}, '.mat']); %spindles
    
    
    %wprob = load(strcat("probability_", study, name, "_C3.txt")); % wprob
    %f2 = fopen(strcat('./results/Cross_valid/probability(sub', num2str(str2num(s1{3})), ').txt')); %wprob
    %f2 = fopen(['./results/DREAMS/probability(1-6_', session1{idx1}(8), '_lr1e-5_drop.5).txt']); %wprob
    %wprob = fscanf(f2,'%f');
    %fclose(f2);
    
    %total_time = zeros(length(session), 1);
    %database = cell(1,1);
    %output = cell(1,3);   % 1-and / 2-or / 3-soft
    %Spindles = cell(2,2);
    %[Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = MASS_Reader(session{idx}, 0, 200);
    %[database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);           
    load(['./data/DREAMsub', session1{idx1}(8), 'Power.mat']);
        
    %load(strcat("./trimmedData/data_", study, name, ".mat")); % data
    %load(strcat("./data/MASSsub",  num2str(str2num(s1{3})), "_EEG.mat")); % data
    
    %f3 = fopen(['./data/test_data/test_', session1{idx1}, '_labels.txt']); %wprob
    %wangDet = fscanf(f3,'%f');
    %fclose(f3);
    
    data = EEG;
    probThresh = 0.1; % P
    timeThresh = 0.15; % seconds

    wangDet = makeBinary(wprob, probThresh, timeThresh*fs); % wangDet
    %wangDet = labels;
    %spindles = labels; %output{:, 2};

%     %start_time = 10*60*fs;
%     start_time = 75288-100;
%     end_time = start_time + 4*fs;
% 
%     figure;
%     x1 = subplot(3,1,1);
%     plot(bandpass(data(start_time:end_time),'bandpass',[3,30]));
%     %xlim([start_time end_time-100]);
%     end_interval = end_time - start_time;
%     xlim([0 end_interval]);
% 
%     x2 = subplot(3,1,2);
%     plot(wprob(start_time:end_time)); hold on;
%     plot(wangDet(start_time:end_time));
%     %xlim([start_time end_time-100]);
%     xlim([0 end_interval]);
% 
%     % x3 = subplot(4,1,3);
%     % plot(spindles(start_time:end_time), 'Color',[0 0 0], 'LineStyle', '-');
%     % %xlim([start_time end_time-100]);
%     % xlim([0 end_interval]);
% 
%     x4 = subplot(3,1,3);
%     movingwin=[1 1/fs];
%     params.Fs=fs;
%     params.fpass=[0 17];
%     params.tapers=[3 5];
%     params.err=0;
%     params.pad=0;
%     [SS,tt,ff]=mtspecgramc(data(start_time-fs:end_time+fs),movingwin,params);
%     colormap('jet');
%     plot_matrix(SS,tt*fs-fs,ff);
%     hold on
%     plot([0,end_interval + fs],[11 11],['w','-'],'linewidth',2);
%     plot([0,end_interval + fs],[16 16],['w','-'],'linewidth',2);
%     xlim([0 end_interval]);
% 
%     linkaxes([x1 x2 x4], 'x');

       
    % These two to be used in generating detections from probability files
    delay = .1;
    
    [po, pe, k, sek, ci, km, fourSquare, detectionMat]= binaryKappa(wangDet', spindles, fs, delay);
    
    avgFreqs = zeros(2);
    avgDurs = zeros(2);
    avgPows = zeros(2);
    
    freqMatAgree = zeros(fourSquare(1, 1), 1);
    freqMat1 = zeros(fourSquare(1, 2), 1);
    freqMat2 = zeros(fourSquare(2, 1), 1);
    
    durMatAgree = zeros(fourSquare(1, 1), 1);
    durMat1 = zeros(fourSquare(1, 2), 1);
    durMat2 = zeros(fourSquare(2, 1), 1);
    
    powMatAgree = zeros(fourSquare(1, 1), 1);
    powMat1 = zeros(fourSquare(1, 2), 1);
    powMat2 = zeros(fourSquare(2, 1), 1);
    
    dbMatAgree = zeros(fourSquare(1, 1), 1);
    dbMat1 = zeros(fourSquare(1, 2), 1);
    dbMat2 = zeros(fourSquare(2, 1), 1);
    
    [avgFreqs(1, 1), avgDurs(1, 1), avgPows(1, 1), freqMatAgree, durMatAgree, powMatAgree] = ...
        getSpindleStats(data, detectionMat(:, 1:2), fs);
    
    [avgFreqs(1, 2), avgDurs(1, 2), avgPows(1, 2), freqMat1, durMat1, powMat1] = ...
        getSpindleStats(data, detectionMat(:, 3:4), fs);

    [avgFreqs(2, 1), avgDurs(2, 1), avgPows(2, 1), freqMat2, durMat2, powMat2] = ...
        getSpindleStats(data, detectionMat(:, 5:6), fs);
    
    wfreq = avgFreqs.*fourSquare;
    s1freq = (wfreq(1,1) + wfreq(1,2))/(fourSquare(1,1) + fourSquare(1,2));
    s2freq = (wfreq(1,1) + wfreq(2,1))/(fourSquare(1,1) + fourSquare(2,1));
    
    wdur = avgDurs.*fourSquare;
    s1dur = (wdur(1,1) + wdur(1,2))/(fourSquare(1,1) + fourSquare(1,2));
    s2dur = (wdur(1,1) + wdur(2,1))/(fourSquare(1,1) + fourSquare(2,1));
    
    wpow = avgPows.*fourSquare;
    s1pow = (wpow(1,1) + wpow(1,2))/(fourSquare(1,1) + fourSquare(1,2));
    s2pow = (wpow(1,1) + wpow(2,1))/(fourSquare(1,1) + fourSquare(2,1));
    
    dbMatAgree = 10*log10(powMatAgree);
    dbMat1 = 10*log10(powMat1);
    dbMat2 = 10*log10(powMat2);
    
    avgDBs = 10*log10(avgPows);
    
    s1powDB = 10*log10(s1pow);
    s2powDB = 10*log10(s2pow);
    
    %save(strcat("./RatData/", study, name , "_stats.mat"));
    %save(strcat("./results/mcsleep_comparison/", session{idx}, "_stats.mat"));
end
%%
    %chats = {'10_071118','10_080318','11_080618'};
    chats = {'1_080618'};
    for subj = 1:length(chats)
        name1 = ['Rat' chats{1,subj}];
        %load(['/media/Share/jwanglab/jwanglabspace/Charles/Spindles/Matlab/full_stats/' name1 '_stats.mat']);
        load(['./RatData/' name1 '_stats.mat']);
        dbMatAgree = zeros(fourSquare(1, 1), 1);
        dbMat1 = zeros(fourSquare(1, 2), 1);
        dbMat2 = zeros(fourSquare(2, 1), 1);
        
        dbMatAgree = 10*log10(powMatAgree);
        dbMat1 = 10*log10(powMat1);
        dbMat2 = 10*log10(powMat2);
        
        freq_nn = zeros((length(freqMat1) + length(freqMatAgree)), 1);
        freq_nn(1:length(freqMat1),1) = freqMat1;
        freq_nn(length(freqMat1)+1:length(freq_nn)) = freqMatAgree;
        
        freq_mc = zeros((length(freqMat2) + length(freqMatAgree)), 1);
        freq_mc(1:length(freqMat2),1) = freqMat2;
        freq_mc(length(freqMat2)+1:length(freq_mc)) = freqMatAgree;
        
        dur_nn = zeros((length(durMat1) + length(durMatAgree)), 1);
        dur_nn(1:length(durMat1),1) = durMat1;
        dur_nn(length(durMat1)+1:length(dur_nn)) = durMatAgree;
        
        dur_mc = zeros((length(durMat2) + length(durMatAgree)), 1);
        dur_mc(1:length(durMat2),1) = durMat2;
        dur_mc(length(durMat2)+1:length(dur_mc)) = durMatAgree;
        
        db_nn = zeros((length(dbMat1) + length(dbMatAgree)), 1);
        db_nn(1:length(dbMat1),1) = dbMat1;
        db_nn(length(dbMat1)+1:length(db_nn)) = dbMatAgree;
        
        db_mc = zeros((length(dbMat2) + length(dbMatAgree)), 1);
        db_mc(1:length(dbMat2),1) = dbMat2;
        db_mc(length(dbMat2)+1:length(db_mc)) = dbMatAgree;
        
        len_nn = length(freq_nn);
        len_mc = length(freq_mc);
        
        total_sub = zeros(max(len_nn, len_mc), 8);
        cHeader = {'freq_nn', 'freq_mc', 'dur_nn', 'dur_mc', 'db_nn', 'db_mc', 'nn_density', 'mc_density'};
        
        total_sub(1:length(freq_nn),1) = freq_nn;
        total_sub(1:length(freq_mc),2) = freq_mc;
        
        total_sub(1:length(dur_nn),3) = dur_nn;
        total_sub(1:length(dur_mc),4) = dur_mc;
        
        total_sub(1:length(db_nn),5) = db_nn;
        total_sub(1:length(db_mc),6) = db_mc;

        
        nn_time = length(wangDet) / (60*fs);
        mc_time = length(spindles) / (60*fs);
        
        nn_density = length(db_nn) / nn_time;
        mc_density = length(db_mc) / mc_time;
        
        disp([name1 ':']);
        disp(['NN density: ' num2str(nn_density) ' spindles/min']);
        disp(['MC density: ' num2str(mc_density) ' spindles/min']);
        
        total_sub(1,7) = nn_density;
        total_sub(1,8) = mc_density;
        
        csvfile = ['results/mcsleep_comparison/' name1 '.csv'];
        
        commaHeader = [cHeader;repmat({','},1,numel(cHeader))]; %insert commaas
        commaHeader = commaHeader(:)';
        textHeader = cell2mat(commaHeader); %cHeader in text with commas
        textHeader = textHeader(1:length(textHeader)-1);


%         xlswrite(xlsfile, header);
%         xlswrite(xlsfile, total_sub);
%         
        fid = fopen(csvfile,'w');
        fprintf(fid,'%s\n',textHeader);
        fclose(fid);
        dlmwrite(csvfile,total_sub,'-append');

        %dlmwrite(['results/mcsleep_comparison/' name1 '.csv'], total_sub);
        
    end

