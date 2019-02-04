clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,5,6,7,9,10,11,12,13,14,17,18,19};
num_subject = {1};
session={};
    for i=1:length(num_subject)
        if num_subject{i}<10
            session={session{:},['01-02-000' num2str(num_subject{i})]};
        else
            session={session{:},['01-02-00' num2str(num_subject{i})]};
        end
    end
load('./data/MASSsub1Power200Hz.mat');
idx = 1;
