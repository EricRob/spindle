%% This script provides a demo for the McSleep spindle detection method
%
% Last EDIT: 4/25/17
% Ankit Parekh
% Perm. Contact: ankit.parekh@nyu.edu
%
% To run the spindle detection on your EDF, replace the filename below
% with your EDF file. 
% The sample EDF used in this script has only 3 channels. Modify the script
% accordingly for your EDF in case of more than 3 channels (Fp1-A1, Cz-A1, O1-A1).
%
% The script downloads the EDF files from the DREAMS database. Please cite
% the authors of the DREAMS database appropriately when using this code. 
% 
% The visual detection by experts are stored as vd1 and vd2. These are
% obtained from the DREAMS Database. 
%
% Please cite as: 
% Multichannel Sleep Spindle Detection using Sparse Low-Rank Optimization 
% A. Parekh, I. W. Selesnick, R. S. Osorio, A. W. Varga, D. M. Rapoport and I. Ayappa 
% bioRxiv Preprint, doi: https://doi.org/10.1101/104414
%
% Note: In the paper above, we discard epochs where body movement artifacts were visible. 
%       Since this script is for illustration purposes only, we do not reject any epochs here
%% MASS data:
clearvars;
addpath(genpath('chronux_2_11'));
clearvars;
addpath(genpath('chronux_2_11'));
num_subject = {1,2,3,4,5,6,7,8};
session = {};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end
%% Initialize
%clear; close all; clc;
warning('off','all')

% EDF file
%num = '11_080618'; % CHANGE
params.channels = [1 1]; % CHANGE

%[header, data] = edfread([params.filename, '.edf']);


%fs = header.frequency(9); 
target_f = 200;

%data = isolateStage(2, num, data, fs);
%data=resample(data' , target_fs, fs)';
%fs = target_fs;

%load(strcat("data_ecog", num, ".mat"));
%data = load(strcat("../Matlab/trimmedData/data_earth", num, ".mat"));
% name = 'Rat1_080618';

%f1 = fopen(['/media/share/jwanglab/jwanglabspace/Zhengdong Xiao/Sleep_Project/dataset/RatData/test_data/' name '_bp2_50.txt']);
%data = fscanf(f1, '%f');
%fclose(f1);

for idx=1:length(session)
     % reading data form .edf file
    %[Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs] = MASS_Reader(session{idx}, 0, target_fs);
    
    data_path = ['DatabaseSpindles/' session{idx} '.edf'];
    [hdr, record] = edfread(data_path);
    IndexC = strfind(hdr.label,'CZA1');
    Index = find(not(cellfun('isempty', IndexC)));
    if isempty(Index)
        IndexC = strfind(hdr.label,'C3A1');
        Index = find(not(cellfun('isempty', IndexC)));
    end
    EEG = record(Index,:)';
    sample_rate = hdr.frequency(Index);
    if sample_rate ~= target_f
        EEG = interp(EEG,target_f/sample_rate);
        sample_rate = target_f;
    end
    
    %data = data';
    data = EEG;
    % one row vector of probability file

    %N = length(data(4, :));
    N = length(data(:));
    n = 0:N-1;

    %n = 0:239999;

    

    stdVal = std(data);
    ratio = 29.9257246048693/stdVal; %29.925... is the std of dreams, which the algorithm is normalized for
    data = data*ratio;
    % for i = 1:length(data)
    %     data(:, i) = data(:, i) * ratio;
    % end

    %vd2 = obtainVisualRecord(visualScorer2,fs,N);  
    %% Select parameters for McSleep
    % Adjust parameters to improve performance 
    params.lam1 = 0.3;
    params.lam2 = 6.5;
    params.lam3 = 36;
    params.mu = 0.5;
    params.Nit = 80;
    params.K = 200;
    params.O = 100;

    % Bandpass filter & Teager operator parameters
    params.f1 = 11;
    params.f2 = 17;
    params.filtOrder = 4;
    params.Threshold = 0.5; 

    % Other function parameters
    % Don't calculate cost to save time 
    % In order to see cost function behavior, run demo.m
    params.calculateCost = 0;   
    %% Run parallel detection for transient separation
    % Start parallel pool. Adjust according to number of virtual
    % cores/processors. Starting the parallel pool for the first time may take
    % few seconds. 

    %if isempty(gcp) 
    %        p = parpool(8); 
    %end


    spindles = psd_preread(params, data', 200);
    %save(strcat("../Matlab/mcsleep/ecog", num, ".mat"), "spindles");
    %save(strcat("../Matlab/mcsleep/earth", num, ".mat"), "spindles");
    save(strcat("results/mcsleep/", session{idx}, ".mat"), "spindles");

end