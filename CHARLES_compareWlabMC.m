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

%function [] = compareWlabMC(name, study, probThresh, timeThresh)
    clear;
   
    num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14}; %{13,14,15,16,17,18,19};
    session={};
    for i=1:length(num_subject)
        if num_subject{i}<10
            session={session{:},['01-02-000' num2str(num_subject{i})]};
        else
            session={session{:},['01-02-00' num2str(num_subject{i})]};
        end
    end
    %name = '300021';
    %study = 'chat';
    
    fs = 200;
%%    
for idx = 1:length(session)     
    s1 = strsplit(session{idx}, '-');
    disp(session{idx});
    %load(strcat("./mcsleep/", study, name, ".mat")); % spindles
    load(['./results/mcsleep/', session{idx}, '.mat']); %spindles
    
    %wprob = load(strcat("probability_", study, name, "_C3.txt")); % wprob
    f2 = fopen('./results/Cross_valid/probability(sub2).txt'); %wprob
    wprob = fscanf(f2,'%f');
    fclose(f2);
    
    %load(strcat("./trimmedData/data_", study, name, ".mat")); % data
    %load(strcat("./data/MASSsub",  num2str(str2num(s1{3})), "_EEG.mat")); % data
     load(['./data/MASSsub', num2str(str2num(s1{3})), 'Power200Hz.mat']);
    data = EEG;
    %%
    probThresh = 0.1; % P
    timeThresh = 0.5; % seconds

    wangDet = makeBinary(wprob, probThresh, timeThresh*fs); % wangDet  
    %%

    start_time = 20*60*fs;
    end_time = start_time + 20*fs;

%     figure;
%     subplot(3,1,1);
%     plot(data(start_time:end_time));
% 
%     subplot(3,1,2);
%     plot(wprob(start_time:end_time)); hold on;
%     plot(wangDet(start_time:end_time));
% 
%     subplot(3,1,3);
%     plot(spindles(start_time:end_time));

    
    %%    
    % These two to be used in generating detections from probability files
    
    [po, pe, k, sek, ci, km, fourSquare, detectionMat]= binaryKappa(wangDet', spindles, fs);
    
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
    
    avgDBs = 10*log10(avgPows);
    
    s1powDB = 10*log10(s1pow);
    s2powDB = 10*log10(s2pow);
    
    %save(strcat("./full_stats/", study, name , "_stats.mat"));
     save(strcat("./full_stats/", session{idx}, "_stats.mat"));
%end
end
