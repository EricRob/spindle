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

%% Initialize
clear; close all; clc;
warning('off','all')


params.filename = 'mros-visit1-aa0018'; % CHANGE THIS
isolateFilenum = '0018'; % CHANGE THIS
probFile = fopen('probability_MrOS_18_C3.txt','rt'); % CHANGE THIS

%% EDF Reading
[header, data] = edfread([params.filename, '.edf']);

studyId = extractBefore(params.filename, "-");
studyId = lower(studyId);

switch studyId
    case "mros"
        params.channels = [4 5];
    case "chat"
        params.channels = [9 10];
end

fs = header.frequency(params.channels(1)); 
target_fs = 200;


data = isolateStage(2, isolateFilenum, data, fs);
data=resample(data' , target_fs, fs)';
%data = data(:, 1:360000);
fs = target_fs;

startTime = 1/fs; % When do we start?
graphTime = 1200; % How long do we go?

N = length(data(params.channels(1), :));

stdVal = std(data(params.channels(1), :));
ratio = 29.9257246048693/stdVal; %29.925... is the std of dreams, which the algorithm is normalized for
for i = 1:length(data(params.channels(1),:))
    data(:, i) = data(:, i) * ratio;
end

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

if isempty(gcp) 
        p = parpool(8); 
end

spindles = psd_preread(params, data, fs);
%% F1 Score calculation

% Change the filename and sampling frequency according to your visual 
% detection filenames
format long g
N = length(spindles);



probRecord = zeros(1200*fs, 1);
i = 1;
while true
    line = fgetl(probFile);
    if ~ischar(line)
      break; 
    end  %end of file

    probRecord(i) = str2double(line);
    i = i + 1;
end % while

% CHANGE THESE
prob_thresh = 0.6; % P
time_thresh = 0.5; % in Seconds

binProbs = makeBinary(probRecord, prob_thresh, time_thresh*fs);

% End of wanglab probabilities

%% Plot the results

n = 0:graphTime*fs-1;

figure(3), clf
gap = 120;

x = n/fs;
plot(x, data(params.channels(1),startTime*fs:startTime*fs + graphTime*fs - 1),...
    x, data(params.channels(2),startTime*fs:startTime*fs + graphTime*fs - 1)-gap, ...
    x, probRecord(startTime*fs:startTime*fs + graphTime*fs - 1)*30-3*gap);

hold on
plot(x, spindles(startTime*fs:startTime*fs + graphTime*fs - 1)*30-2*gap, 'k');
hold on
plot(x, binProbs(startTime*fs:startTime*fs + graphTime*fs - 1)*30-3*gap, 'm');


box off
xlabel('Time (s)')
ylabel('\mu V')
title('Comparison of McSleep to Wanglab')
%ylim([-6*gap gap])
set(gca,'YTick',[])
wangStart = 0; % When did wanglab spindle start?
mcStart = 0; % When did mcsleep spindle start?
spinMin = 0; % What was the lowest value the spindle had?
spinMax = 0; % What was the highest?
for i = startTime*fs:startTime*fs + graphTime*fs - 1
    if binProbs(i) ~= 0 % i.e. if wang spindle detection
        if wangStart == 0 % i.e. start of spindle
            wangStart = i;
        end
    else
        if wangStart ~= 0 % i.e. end of spindle
            hold on
            spinMin = min(data(params.channels(1), wangStart:i));
            spinMax = max(data(params.channels(1), wangStart:i));
            plot([(wangStart-startTime*fs)/fs (wangStart-startTime*fs)/fs], [spinMin spinMax], 'm');
            plot([(i-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMax], 'm');
            plot([(wangStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMin], 'm');
            plot([(wangStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMax spinMax], 'm');
            
            wangStart = 0;
        end
    end
    if spindles(i) ~= 0 % i.e. if mc spindle detection
        if mcStart == 0 % i.e. start of spindle
            mcStart = i;
        end
    else
        if mcStart ~= 0 % i.e. end of spindle
            hold on
            spinMin = min(data(params.channels(1), mcStart:i));          
            spinMax = max(data(params.channels(1), mcStart:i));
            plot([(mcStart-startTime*fs)/fs (mcStart-startTime*fs)/fs], [spinMin spinMax], 'k');
            plot([(i-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMax], 'k');
            plot([(mcStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMin], 'k');
            plot([(mcStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMax spinMax], 'k')
           
            spinMin = min(data(params.channels(2), mcStart:i)) - gap;          
            spinMax = max(data(params.channels(2), mcStart:i)) - gap;
            plot([(mcStart-startTime*fs)/fs (mcStart-startTime*fs)/fs], [spinMin spinMax], 'k');
            plot([(i-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMax], 'k');
            plot([(mcStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMin spinMin], 'k');
            plot([(mcStart-startTime*fs)/fs (i-startTime*fs)/fs], [spinMax spinMax], 'k')
            
            mcStart = 0;
        end  
    end
end
legend('Channel 1', 'Channel 2', 'Wanglab Probs', 'McSleep Detections', 'Wanglab Detections')
%xlim([1 n/fs])

fig = gcf;
fig.PaperUnits = 'points';
fig.PaperPosition = [0 0 1050*graphTime/60 1000]; % ~1050 pixels per minute, width wise
ax.XTickMode = 'manual'; % Ensures axes remain intact
xticks(0:10:graphTime);
print('BigFig','-dpng','-r0')
   
   