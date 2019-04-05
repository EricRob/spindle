% Read contents of a mrOS edf file numbered filenum into the output
% whichSet specifies which dataset to read from (mrOs or chat, currently).
function [hdr, records] = quickRead(filenum, whichSet)
    addpath(genpath('chronux_2_11'));
    
    if strcmpi(whichSet,'mros')
        filename = ['edfs/', strcat('mros-visit1-aa', filenum ,'.edf')];
    else % chat
        filename = ['edfs/', strcat('chat-baseline-', filenum, '.edf')];
    end
    [hdr, records] = edfread(filename);
    %disp(hdr)
    %disp(records)
end