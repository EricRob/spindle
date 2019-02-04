clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
%num_subject = {1,2,3,4};
session={};
session={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session={session{:},['01-02-000' num2str(num_subject{i})]};
    else
        session={session{:},['01-02-00' num2str(num_subject{i})]};
    end
end

total_time = zeros(length(session), 1);
database = cell(1,1);
output = cell(1,3);   % 1-and / 2-or / 3-soft
Spindles = cell(2,2);
upsampled = cell(1,1);

for idx=19:length(session)
    fs = 34;
    num_step_fs = floor((250*0.001)/(1/fs));
    total_time = zeros(length(session), 1);
    database = cell(1,1);
    output = cell(1,3);   % 1-and / 2-or / 3-soft
    Spindles = cell(2,2);
    upsampled = cell(1,1);
    load(['./data/MASSsub' num2str(num_subject{idx}) 'Power' num2str(floor(fs)) 'Hz.mat']);
    target_f = 200;
    num_step_target_f = floor((250*0.001)/(1/target_f));
    num_step_fs = floor((250*0.001)/(1/floor(fs)));
    open_string_data = ['SleepSpindleData4RNN/upsampled_test_' session{idx} '_' num2str(num_step_target_f)  '_100%_f'  num2str(target_f)  '_from_' num2str(floor(fs)) '_mf.txt']; 
    %open_string_label = ['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step_fs)  '_100%_f'  num2str(fs)  '_mf_labels.txt']; 
    h1 = fopen(open_string_data, 'wt');
    %h2 = fopen(open_string_label, 'wt');
    upsampled{1} = resample(database{1,1},target_f,floor(fs));
        %     test_data = zeros(num_step, length(EEG)-num_step+1,25);
    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    %       format1 = '%f %f\n';
    %format2 = '%d\n';
    fprintf(h1,format1,upsampled{1}');
    %fprintf(h2,format2, output{1,2});
    
    fclose(h1);
    %fclose(h2);
end

%%
clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
%num_subject = {1,2,3,4};
session={};
session={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session={session{:},['01-02-000' num2str(num_subject{i})]};
    else
        session={session{:},['01-02-00' num2str(num_subject{i})]};
    end
end

total_time = zeros(length(session), 1);
database = cell(1,1);
output = cell(1,3);   % 1-and / 2-or / 3-soft
Spindles = cell(2,2);
upsampled = cell(1,1);

for idx=5:length(session)
    fs = 50;
    num_step_fs = floor((250*0.001)/(1/fs));
    total_time = zeros(length(session), 1);
    database = cell(1,1);
    output = cell(1,3);   % 1-and / 2-or / 3-soft
    Spindles = cell(2,2);
    upsampled = cell(1,1);
    load(['./data/MASSsub' num2str(num_subject{idx}) 'Power' num2str(floor(fs)) 'Hz.mat']);
    target_f = 200;
    num_step_target_f = floor((250*0.001)/(1/target_f));
    num_step_fs = floor((250*0.001)/(1/floor(fs)));
    open_string_data = ['SleepSpindleData4RNN/upsampled_test_' session{idx} '_' num2str(num_step_target_f)  '_100%_f'  num2str(target_f)  '_from_' num2str(floor(fs)) '_mf.txt']; 
    %open_string_label = ['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step_fs)  '_100%_f'  num2str(fs)  '_mf_labels.txt']; 
    h1 = fopen(open_string_data, 'wt');
    %h2 = fopen(open_string_label, 'wt');
    upsampled{1} = resample(database{1,1},target_f,floor(fs));
        %     test_data = zeros(num_step, length(EEG)-num_step+1,25);
    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    %       format1 = '%f %f\n';
    %format2 = '%d\n';
    fprintf(h1,format1,upsampled{1}');
    %fprintf(h2,format2, output{1,2});
    
    fclose(h1);
    %fclose(h2);
end