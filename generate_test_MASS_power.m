clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
%num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
num_subject = {1};
%num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
session={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session={session{:},['01-02-000' num2str(num_subject{i})]};
    else
        session={session{:},['01-02-00' num2str(num_subject{i})]};
    end
end
%%

total_time = zeros(length(session), 1);
database = cell(1,1);
output = cell(1,3);   % 1-and / 2-or / 3-soft
Spindles = cell(2,2);
target_f = 100;
%%
for idx = 1:length(session)
     % reading data form .edf file
    [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs] = MASS_Reader(session{idx}, 0, target_f);
    % appending data 
    [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);
    
    % Start Added by Eric
    std_value = zeros(size(Spindles{2,1}, 1),1);
    for i=1:size(Spindles{2,1}, 1)
        s_data=database{1}(floor(Spindles{2,1}(i)*fs):floor((Spindles{2,1}(i)+Spindles{2,2}(i))*fs));
        s_data = detrend(s_data);
        std_value(i) = std(s_data);
    end
    pd = fitdist(std_value,'Normal');
    % End Added by Eric
    
 %%   compute power
    movingwin=[1 1/target_f];
    params.Fs=fs;
    params.fpass=[0 25];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    ratio = zeros(size(database{1}));
    power_BB = zeros(size(database{1}));
    power_spindle = power_BB;
    band_pass = power_BB;
    up_env = power_BB;
    power_feat = zeros(size(database{1}, 1), 19);
    
    for k=1:length(database{1})-floor(fs)+1
        
        seq = database{1}(k:k+floor(fs)-1);
        seq = (seq-mean(seq))/mean(pd); % Eric changed to mean(pd) from 13.8275
        
        tic
        [SS,tt,ff]=mtspecgramc(seq, movingwin, params);
        time_pow = toc
        
        energyDB = 10*log10(SS);
        %     DB_LF = mean(energyDB(:,3:8),2);
        DB_BB = mean(energyDB(:,4:11),2);
        DB_spindle = mean(energyDB(:,12:22),2);
        power_BB(k+floor(fs)/2)=DB_BB;
        power_spindle(k+floor(fs)/2)=DB_spindle;
        ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
        
        %tic
        bdpass = bandpass(seq, 'bandpass', [9,16], floor(fs)); % Eric added [9,16], floor(fs)
        band_pass(k+floor(fs)/2) = bdpass(floor(fs)/2+1);
        [up,~] = envelope(bdpass);
        up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
        %time_env = toc
        
        power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
        
    end
    database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
    %%
    % save MASSsub1Power;

     %database = cell(1,1);
     %output = cell(1,3);   % 1-and / 2-or / 3-soft
     %Spindles = cell(2,2);
     
     load(['./data/MASSsub' num2str(num_subject{idx}) 'Power' num2str(target_f) 'Hz.mat']);
     num_step = floor((250*0.001)/(1/fs));
     
     open_string_data = ['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt']; 
     open_string_label = ['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt']; 
     h1 = fopen(open_string_data, 'wt');
     h2 = fopen(open_string_label, 'wt');
     
     %h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt'] , 'wt');
     %h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt'] , 'wt');
     


%     test_data = zeros(num_step, length(EEG)-num_step+1,25);
      format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
%       format1 = '%f %f\n'; 
      format2 = '%d\n';
      if 0
          test_data = zeros(num_step,25);  %25
          test_label =zeros(num_step,1);
          labels = output{1,2};
          for i=1:floor((length(EEG)-num_step+1))
                test_data(:,1)=detrend(database{1}(i:i+num_step-1,1));
                test_data(:,2)=database{1}(i:i+num_step-1,2:end); %2:end
                test_label=labels(i:i+num_step-1);
                fprintf(h1,format1, test_data');
                fprintf(h2,format2, test_label);
          end
      else
          fprintf(h1,format1,database{1}');
          fprintf(h2,format2, output{1,2});
      end
%     test_data = reshape(test_data, [num_step*(length(EEG)-num_step+1), 25]);
%     test_label = reshape(test_label, [num_step*(length(EEG)-num_step+1), 1]);
%     fprintf(h1,format1, test_data(1:floor(size(test_data,1)/100)*10,:)');
%     fprintf(h2,format2, test_label(1:floor(size(test_data,1)/100)*10));

    fclose(h1);
    fclose(h2);
       
end
%%
num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
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
target_f = 200;
num_step = floor((250*0.001)/(1/target_f));
for idx=1:length(session)
    load(['./data/power_refine/MASSsub' num2str(idx) 'Power' num2str(target_f) 'Hz.mat']);
    
    open_string_data = ['./data/test_data_refine/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt']; %-02- or -20- ?
    open_string_label = ['./data/test_data_refine/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt']; %-02- or -20- ?
    h1 = fopen(open_string_data, 'wt');
    h2 = fopen(open_string_label, 'wt');
    
    %h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt'] , 'wt');
    %h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt'] , 'wt');
    
    
    
    %     test_data = zeros(num_step, length(EEG)-num_step+1,25);
    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    %       format1 = '%f %f\n';
    format2 = '%d\n';
    fprintf(h1,format1,database{1}');
    fprintf(h2,format2, output{1,2});
    
    fclose(h1);
    fclose(h2);
end
%%
num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
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
target_f = 34;
num_step = floor((250*0.001)/(1/target_f));
for idx=5:length(session)
    load(['./data/MASSsub' num2str(idx) 'Power' num2str(target_f) 'Hz.mat']);
    
    open_string_data = ['./SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt']; %-02- or -20- ?
    open_string_label = ['./SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt']; %-02- or -20- ?
    h1 = fopen(open_string_data, 'wt');
    h2 = fopen(open_string_label, 'wt');
    
    %h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt'] , 'wt');
    %h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt'] , 'wt');
    
    
    
    %     test_data = zeros(num_step, length(EEG)-num_step+1,25);
    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    %       format1 = '%f %f\n';
    format2 = '%d\n';
    fprintf(h1,format1,database{1}');
    fprintf(h2,format2, output{1,2});
    
    fclose(h1);
    fclose(h2);
end
%%
load(['./data/MASSsub4Power50Hz.mat']);
num_step = floor((250*0.001)/(1/fs));

open_string_data = './SleepSpindleData4RNN/test_01-02-0004_12_100%_f50_mf.txt'; %-02- or -20- ? 
open_string_label = './SleepSpindleData4RNN/test_01-02-0004_12_100%_f50_mf_labels.txt'; %-02- or -20- ? 
h1 = fopen(open_string_data, 'wt');
h2 = fopen(open_string_label, 'wt');

%h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf.txt'] , 'wt');
%h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f' num2str(target_f) '_mf_labels.txt'] , 'wt');



%     test_data = zeros(num_step, length(EEG)-num_step+1,25);
format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
%       format1 = '%f %f\n';
format2 = '%d\n';
fprintf(h1,format1,database{1}');
fprintf(h2,format2, output{1,2});

fclose(h1);
fclose(h2);
% std_value = zeros(size(Spindles{1}, 1),1);
% for i=1:size(Spindles{1}, 1)
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
%     s_data = detrend(s_data);
%     std_value(i) = std(s_data);
% end
% pd = fitdist(std_value,'Normal');
% disp(mean(pd));
% for i=121:140
%     figure;
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
%     s_data = detrend(s_data);
%     plot(s_data);
% end
%%
% clear;
% num_step = 50;
% format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
% format2 = '%d\n';
% for m=1:19
%     load(['data/MASSsub' num2str(m) 'Power.mat']);
%     h1=fopen(['data/test_data/test_' session{idx} '_' num2str(num_step) '_100%_f200_mf.txt'] , 'wt');
%     h2=fopen(['data/test_data/test_' session{idx} '_' num2str(num_step) '_100%_f200_mf_labels.txt'] , 'wt');
%     fprintf(h1,format1,database{1}');
%     fprintf(h2,format2, output{1,2});
%     fclose(h1);
%     fclose(h2);
% end
%           