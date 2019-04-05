% Written by Charles Zhang
% June 2018

% Given a stage of sleep and a file number for the mrOS dataset, pares
% records down to only columns representing time spent in that stage

% Input:
%   stageNum -- integer representing the stage of sleep
%   fileNum -- actually a string, since mrOS numbers have leading zeros
%   records -- output from the edfread function written by Brett Shoelson
%   fs -- sampling rate, used to translate between seconds in annotation
%           and frames in edf
% Output:
%   newRecords -- records, except all non-stageNum sleep is removed

function [newRecords] = isolateStage(stageNum, fileNum, records, fs)

newRecords = records(:, 1);
res = [];

fid = fopen(['Annotations/simple-stages_trial-' fileNum '_stage-' int2str(stageNum) '.txt'],'rt');

i = 1;
while true
  line = fgetl(fid);
  if ~ischar(line)
      break; 
  end  %end of file
  
  tup = strsplit(line,{'(',', ', ')'});
  % tup(1) == ''
  % tup(2) == stageNum
  % tup(3) == startTime
  % tup(4) == endTime
  
  res(1, i) = str2double(tup(3));
  res(2, i) = str2double(tup(4));
  i = i + 1;
end % while
fclose(fid);

curr = 1;
next = -1;

% For each segment of stage-(stageNum) sleep indicated by res, transpose
% that segment from records into newRecords.
for i = 1:length(res)
    next = curr + (res(2, i) - res(1, i));
    newRecords(:, fs*curr:fs*next) = records(:,fs*res(1, i):fs*res(2, i));
    curr = next + 1;
end % for


end % function

