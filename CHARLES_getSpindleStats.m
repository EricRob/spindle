% Written by Charles Zhang
% July 2018

% Given a single row/column vector of data and a binary record of the same size annotating
% the locations of spindle detections, returns the average frequency, duration, and power
% of each of the spindles. (Also matrices containing individual data per spindle).

% Input:
%   record -- Vector containing EEG/LFP/etc. data.
%   detections -- Two column matrix annotating beginning and ending sample-frames of
%       detected spindles.
%   fs -- Sampling Rate (Hz), used to calculate frequency and duration.
% Output:
%   freqAvg -- Average frequency of detected spindles.
%   durAvg -- Average duration of detected spindles.
%   powAvg -- Average power of detected spindles in 10-16 Hz Frequency range.
%   freqMat -- Vector of individual frequencies, per spindle.
%   durMat -- Vector of individual durations, per spindle.
%   powMat -- Vector of individual powers, per spindle.

function [freqAvg, durAvg, powAvg, freqMat, durMat, powMat] = getSpindleStats(record, detections, fs)
    detections = detections(1:find(detections(:, 1),1,'last'), :); % Trim trailing zeros
    numSpins = length(detections);
    if numSpins == 2 % EDGE CASE
        [x, y] = size(detections);
        if x == 1 || y == 1
            numSpins = 1;
        end
    end % EDGE CASE
    
    % Duration
    durMat = zeros(numSpins , 1);
    for i = 1:numSpins
       durMat(i, 1) = (detections(i, 2) - detections(i, 1))/fs;
    end
    durAvg = mean(durMat);
    % \Duration
    
    freqMat = zeros(numSpins , 1);
    powMat = zeros(numSpins, 1);
    for i = 1:numSpins % For Each Detection
       % Load spectrum 
       [pspec, f] = pspectrum(record(detections(i,1):detections(i,2)), fs);
       lowerBound = find(f > 10, 1);
       upperBound = find(f > 16, 1) - 1;
       % lowerBound and upperBound now denote the boundaries of the frequency band for spindles
       
       [~, maxInd] = max(pspec(lowerBound : upperBound));
       freqMat(i, 1) = f(maxInd+lowerBound);
       powMat(i, 1) = mean(pspec(lowerBound : upperBound));
    end % For each detection
    freqAvg = mean(freqMat);
    powAvg = mean(powMat);
end% of function

