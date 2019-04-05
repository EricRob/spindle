% Written by Charles Zhang
% July 2018

% Given a binary vector (for instance, of spindle detections calculated from a probability vector
% of equivalent length), writes the indices corresponding to the beginnings and ends of detections to
% a two wide, n tall matrix, and saves that to a text file.

% Input:
%   binProbs -- Binary vector of detections
%   fs -- Sampling rate (Hz)
%   outFileName -- Name of file to save to.
% Output:
%   [detTimes] -- Two wide, n tall matrix of beginnings and ends of detections in binProbs
%   Saved to text file, also.

function [detTimes] = writeDetections(binProbs, fs, outFileName)

fid = fopen(strcat(outFileName, ".txt"), "w+");

detTimes = zeros(1);

currSpinds = 1;

startTime = -1;
for i=1:size(binProbs)
    if startTime == -1
        if binProbs(i) == 1
            startTime = i/fs;
        end
    else % we're in a spindle already
        if binProbs(i) == 0
            detTimes(currSpinds, 1) = startTime;
            detTimes(currSpinds, 2) = i/fs; % end of spindle time
            startTime = -1;
            currSpinds = currSpinds + 1;
        end
    end
end %for

for i = 1:size(detTimes, 1)
    fprintf(fid, "%3f %3f\n", detTimes(i, 1), detTimes(i, 2)-detTimes(i, 1));
end

