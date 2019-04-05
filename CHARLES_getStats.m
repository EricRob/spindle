% Written by Charles Zhang
% June 2018

% Given a binary vector representing activations on a neural network detecting spindles and a
% sampling rate fs, computes various statistics

% Inputs:
%   binProbs -- vector containing values 0 and 1 only, represents activations
%   fs -- sampling rate in Hz, used co calculate duration and density
% Outputs:
%   avgDuration -- the average duration of a spindle, in seconds
%   density -- average number of spindles per minute, a cross binProbs
%   numSpindles -- total number of  spindles in binProbs

function [avgDuration, density, numSpindles] = getStats(binProbs,fs)
    length = size(binProbs, 1);
    
    spindles = [];
    currStart = 0;
    currNum = 1;
    for i = 1:length % for each frame in binProbs
        if currStart == 0 % Not in a spindle yet
            if binProbs(i) == 1 % But this is the start of one
                currStart = i;
            end
        else % Already in a spindle
            if binProbs(i) == 0 % But it's over now
                spindles(1, currNum) = currStart;
                spindles(2, currNum) = i - 1;
                currNum = currNum + 1;
                currStart = 0;
            end
        end
    end
    if currStart ~= 0 % If we're still in a spindle at the very end of the file
        spindles(1, currNum) = currStart;
        spindles(2, currNum) = length;
        currNum = currNum + 1;
    end
    
    % We now have a matrix, spindles, of dimensions 2(rows)x n(columns) where n is the number of
    % detected spindles in our input.
    
    numSpindles = currNum - 1;
    time = length/fs; % seconds now
    time = time/60; % minutes now
    density = numSpindles/time; % Number of spindles per minute.
    
    %Calculate duration
    totalDur = 0;
    for i = 1:size(spindles, 2)
        totalDur = totalDur + spindles(2, i) - spindles(1, i);
    end
    avgDuration = totalDur/numSpindles;
    avgDuration = avgDuration/fs;

end

