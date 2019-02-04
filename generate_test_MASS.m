clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
num_subject = {1};
session={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session={session{:},['01-02-000' num2str(num_subject{i})]};
    else
        session={session{:},['01-02-00' num2str(num_subject{i})]};
    end
end
num_step = 50;
total_time = zeros(length(session), 1);
database = cell(1,1);
output = cell(1,3);   % 1-and / 2-or / 3-soft
Spindles = cell(2,2);
for idx=1:length(session)
    % reading data form .edf file
    [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs] = MASS_Reader(session{idx}, 0);
    % appending data 
    [database,output,Spindles,total_time] = collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);           

    
    h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f200.txt'] , 'wt');
    h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_' num2str(num_step) '_100%_f200_labels.txt'] , 'wt');
    
    if 0
        test_data = zeros(num_step, length(EEG)-num_step+1);
        test_label =test_data;
        labels = output{1,2};
        for i=1:length(EEG)-num_step+1
            test_data(:,i)=EEG(i:i+num_step-1);
            test_label(:,i)=labels(i:i+num_step-1);
        end
        test_data=detrend(test_data);
        test_data = reshape(test_data, [num_step*(length(EEG)-num_step+1), 1]);
        test_label = reshape(test_label, [num_step*(length(EEG)-num_step+1), 1]);
    else
        test_data = EEG;
        test_label = output{1,2};
    end
%     fprintf(h1,'%f\n', test_data(1:floor(size(test_data,1)/100)*10));
%     fprintf(h2,'%d\n', test_label(1:floor(size(test_data,1)/100)*10));
    fprintf(h1,'%f\n', test_data);
    fprintf(h2,'%d\n', test_label);
    fclose(h1);
    fclose(h2);
end

std_value = zeros(size(Spindles{1}, 1),1);
for i=1:size(Spindles{1}, 1)
    s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
    s_data = detrend(s_data);
    std_value(i) = std(s_data);
end
pd = fitdist(std_value,'Normal');
disp(mean(pd));
% for i=121:140
%     figure;
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
%     s_data = detrend(s_data);
%     plot(s_data);
% end

