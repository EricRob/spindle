clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
num_subject = {1};
load_data = 1; % 1 - load_data from .edf file   0 - load data from .mat file
if load_data
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
    for idx=1:length(session)
        % reading data form .edf file
        [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs] = MASS_Reader(session{idx}, 0);
        % appending data 
        [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);           
    end
else
    load('MassDataBasePart.mat');   
end
% save MassDataBase database output Spindles
std_value = zeros(size(Spindles{1,1}, 1),1);
for i=1:size(Spindles{1,1}, 1)
    s_data=database{1}(floor(Spindles{1,1}(i)*fs):floor((Spindles{1,1}(i)+Spindles{1,2}(i))*fs));
    s_data = detrend(s_data);
    std_value(i) = std(s_data);
end
pd = fitdist(std_value,'Normal');
disp(mean(pd));
% for i=2040:2070
%     figure;
%     s_data=database{1}((Spindles{1}(i)-0.5)*fs:floor((Spindles{1}(i)+Spindles{2}(i)+0.5)*fs));
% %     s_data = detrend(s_data);
%     plot([1:length(s_data)], s_data);
%     hold on 
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
% %     s_data = detrend(s_data);
%     plot([floor(0.5*fs)+1:length(s_data)+floor(0.5*fs)],s_data,'r');
% end

%% set subset fraction
subset = 1;

%  get baseline
num_steps = 50;   % sequency length of one example
baseline=database{1};
baseline(stages~=2)=[];
t_output = output{1,2};
t_output(stages~=2)=[];
baseline(t_output==1)=[];
baseline =  baseline(1:floor(length(baseline)/num_steps)*num_steps);
num_base = length(baseline)/num_steps;
baseline = reshape(baseline, [num_steps, num_base]);
baseline = detrend(baseline);
train_b = floor(num_base*0.7*subset);  % take 70% of the baseline as traning data
valid_b = floor(num_base*0.2*subset);  % take 20% of the baseline as validation data
test_b = floor(num_base*0.1*subset); % num_base-train_b-valid_b;
ind_b = randperm(num_base);
train_base = reshape(baseline(:,ind_b(1:train_b)), [num_steps*train_b , 1]);
valid_base = reshape(baseline(:,ind_b(train_b+1:train_b+valid_b)), [num_steps*valid_b , 1]);
test_base = reshape(baseline(:,ind_b(train_b+valid_b+1:train_b+valid_b+test_b)), [num_steps*test_b , 1]);

%% Remove spindles with power less than 4dB
movingwin=[1 1/target_f];
params.Fs=fs;
params.fpass=[0 25];
params.tapers=[3 5];
params.err=0;
params.pad=0;
[SS,tt,ff]=mtspecgramc(database{1},movingwin,params);
energyDB = 10*log10(SS);
%     DB_LF = mean(energyDB(:,3:8),2);
% DB_BB = mean(energyDB(:,4:11),2);
DB_spindle = mean(energyDB(:,12:22),2);
power_spindle =zeros(size(database{1}));
power_spindle(tt(1)*fs:tt(end)*fs)=DB_spindle;
save Augment_MASS_sub1
%% remove spindle duration less than 0.5s
S = [Spindles{2,1}, Spindles{2,2}];
pp=zeros(size(S,1),1);
for m=1:size(S,1)
    pp(m)=power_spindle(floor((S(m,1)+S(m,2)/2)*fs)) ;
end
% S(pp<4,:)=[];
% S(S(:,2)<0.5,:)=[];
%% Augment Spindles
num_spindles = size(S,1);
train_s = floor(num_spindles*0.7*subset);
valid_s = floor(num_spindles*0.2*subset);
test_s = floor(num_spindles*0.1*subset);%num_spindles-train_s-valid_s;
ind_s = randperm(num_spindles);
train_spindles = S(ind_s(1:train_s) , :);
valid_spindles =  S(ind_s(train_s+1:train_s+valid_s), :);
test_spindles = S(ind_s(train_s+valid_s+1:train_s+valid_s+test_s), :);

suffix = 'or';
h1=fopen(['SleepSpindleData4RNN/Augment_train_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '.txt'],'wt');
h2=fopen(['SleepSpindleData4RNN/Augment_train_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '_labels.txt'],'wt');
h3=fopen(['SleepSpindleData4RNN/Augment_valid_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '.txt'],'wt');
h4=fopen(['SleepSpindleData4RNN/Augment_valid_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '_labels.txt'],'wt');
h5=fopen(['SleepSpindleData4RNN/Augment_test_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '.txt'],'wt');
h6=fopen(['SleepSpindleData4RNN/Augment_test_MASS_sub18_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_' suffix '_labels.txt'],'wt');

format1 = '%f\n';
format2 = '%d\n';
augment_data(train_spindles, h1, h2, num_steps, database, output{1,2}, fs, 0, format1, format2);
augment_data(valid_spindles, h3, h4, num_steps, database, output{1,2}, fs, 0, format1, format2);
augment_data(test_spindles, h5, h6, num_steps, database, output{1,2}, fs, 1, format1, format2);

% add baseline to file
fprintf(h1,'%f\n',train_base);
fprintf(h2,'%d\n',zeros(size(train_base)));
fprintf(h3,'%f\n',valid_base);
fprintf(h4,'%d\n',zeros(size(valid_base)));
fprintf(h5,'%f\n',test_base);
fprintf(h6,'%d\n',zeros(size(test_base)));

fclose(h1);
fclose(h2);
fclose(h3);
fclose(h4);
fclose(h5);
fclose(h6);

