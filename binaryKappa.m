% Written by Charles Zhang
% July 2018

% Kappa function here:
%    Cardillo G. (2007) Cohen's kappa: compute the Cohen's kappa ratio on a square matrix.   
%    http://www.mathworks.com/matlabcentral/fileexchange/15365

% Given a pair of binary sequences (for instance, two pairs of spindle detection outcomes, for
% which each sequence contains either a 0 or 1 for every sampling frame of an EEG, LFP, etc.),
% provide the kappa coefficient and other statistics pertaining to their agreement.

% Input:
%   seq1 -- First input sequence,  to be compared to seq2
%   seq2 -- Second input sequence, to be compared to seq1
%   fs -- Sampling rate of input sequences (Hz), important because baseline agreements are
%       counted only once per second of continuous agreement, roughly the length of
%       a spindle
%   delay -- Number of sampling frames allowed after the end of one detection
%       and before the next before they will be counted as seperate events
%       **OPTIONAL** (default is 0.1s as defined by fs)
% Output:
%   po -- Observed Agreement
%   pe -- Random Agreement   (po-pe = true agreement)
%   k -- Cohen's Kappa Coefficient
%   sek - Kappa Error
%   ci -- Kappa Confidence Interval
%   km -- Maximum Possible Kappa    (k/km expresses kappa as a percentage of the maximum)
%   fourSquare -- Agreements and disagreements, arranged in a grid as follows:
%          Agreed Detections  |  Sequence 1 Only
%         --------------------|--------------------
%            Sequence 2 Only  |  Agreed Baseline
%   detectionMat -- Six Column matrix denoting the beginnings and endings of spindles in each
%       category.
%           AgreeBeginning | AgreeEnd | SeqOneBeginning | SeqOneEnd | SeqTwoBeginning | SeqTwoEnd


function [po, pe, k, sek, ci, km, fourSquare, detectionMat]...
= binaryKappa(seq1, seq2, fs, delay)
    if nargin < 4
        delay = fs/10;
    end % no delay

    seqLength = length(seq1);
    if size(seq1) ~= size(seq2)
        disp("Error: sizes of seq1 and seq2 are not the same.");
    end % error catch
    
    detAgreements = 0; % Agreed Detections
    s1Det = 0; % Sequence one alone detections
    s2Det = 0; % Sequence two alone detections
    baseAgreements = 0; % Baseline agreements (neither detected)
    
    modeFlag = 0; % 0 for baseline, 1 for 'event'
    baseCount = 0; % For counting length of baseline before baseAgreements incremented
    whichSeq = 0; % For tracking if a detection is one sided or an agreement.
    delayCount = 0; % For counting frames before an event ends, counts up to delay
    
    spinStart = 0; % Beginning of spindle for detectionMat
    spinEnd = 0; % Ending of spindle for detectionMat
    
    detectionMat = zeros(floor(2* length(seq1)/(fs*60)), 6);
    
    for i = 1:seqLength
        if modeFlag == 0 % baseline
            if seq1(i) == 0 && seq2(i) == 0 % i.e. if still baseline
                baseCount = baseCount + 1;
                if baseCount > fs - 1
                    baseCount = 0;
                    baseAgreements = baseAgreements + 1;
                end
            else % detection on one channel
                modeFlag = 1;
                baseCount = 0;
                spinStart = i;
                if seq1(i) == 1 && seq2(i) == 1 % i.e. both on frame 1
                    whichSeq = 3;
                elseif seq1(i) == 1 % only seq1
                    whichSeq = 1;
                else % only seq2
                    whichSeq = 2;
                end % which sequence
            end % ifelse still baseline
        else % detection
            if seq1(i) == 1 || seq2(i) == 1 % ongoing
                delayCount = 0;
                if whichSeq == 1
                    if seq2(i) == 1 % both now
                        whichSeq = 3;
                    end
                elseif whichSeq == 2
                    if seq1(i) == 1 % both now
                        whichSeq = 3;
                    end
                end
            else % end of event?
                if delayCount == 0
                    spinEnd = i;
                end %dc=0
                delayCount = delayCount + 1;
                if delayCount > delay % delay fulfilled
                    modeFlag = 0; % end of event!
                    delayCount = 0;
                    if whichSeq == 1 % one only
                        s1Det = s1Det + 1;
                        detectionMat(s1Det, 3) = spinStart;
                        detectionMat(s1Det, 4) = spinEnd;
                    elseif whichSeq == 2 % two only
                        s2Det = s2Det + 1;
                        detectionMat(s2Det, 5) = spinStart;
                        detectionMat(s2Det, 6) = spinEnd;
                    else % both detect
                        detAgreements = detAgreements + 1;
                        detectionMat(detAgreements, 1) = spinStart;
                        detectionMat(detAgreements, 2) = spinEnd;
                    end % whichSeq
                end % real end of event
            end % ifelse ongoing event
        end % ifelse baseline or not
    end % end of iteration
    
    % At this point, detAgreements, s1Det, s2Det, and baseAgreements are initialized
    %   to their final values.
    
    fourSquare = [detAgreements, s1Det; s2Det, baseAgreements];
    [po, pe, k, sek, ci, km] = kappa(fourSquare);
end

