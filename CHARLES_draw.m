hasProbs = 1;
whichData = 1;

whichChannel = 4; % MrOS -- 4 ;; Chat -- 9

endTime = startTime + window;
if hasProbs == 1
    makeGraphs(startTime, endTime, fs, records(whichChannel:whichChannel+1, :), probRecord, hasProbs, whichData);
else
    makeGraphs(startTime, endTime, fs, records(whichChannel:whichChannel+1, :)); 
end