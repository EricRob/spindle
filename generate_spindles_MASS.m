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

baseline = database{1};
ind = [];
for i=1:length(Spindles{2,1})
    ind = [ind floor(fs*Spindles{2,1}(i)):floor(fs*(Spindles{2,1}(i)+Spindles{2,2}(i)))];
end
baseline(ind)=[];
stages(ind)=[];
baseline(stages~=2)=[];
%% 
movingwin=[1 .005];
params.Fs=fs;
params.fpass=[0 50];
params.tapers=[3 5];
params.err=0;
params.pad=0;
use_EMD = 0;
close all
visual = 0;
len = 6;
num_synth = floor(length(baseline)/len/fs);
synth_data=zeros(floor(len*fs), num_synth);
synth_label=synth_data;
synth_neg_data=zeros(floor(len*fs), num_synth);
synth_mix=synth_label;
synth_mix_label=synth_mix;
pos_power=zeros(num_synth,1);
neg_power=zeros(num_synth,1);
% rParabEmd = rParabEmd__L (baseline, 50, 50, 1);
%%
for k=1:num_synth
lambda=betarnd(0.2,0.2);
simulate=simulate_spindle(200,len);
%figure;plot(simulate);
ind=find(simulate>0.4 | simulate<-0.4);
s=floor(len*(k-1)*fs);
base=baseline(1+s:s+floor(len*fs));
if use_EMD
    rParabEmd = rParabEmd__L (base, 50, 50, 1);
    synth_ = zeros(length(base),1);
    %scale=max(max(rParabEmd));
    for i=1:size(rParabEmd,2)
        if i~=2
            synth = synth+rParabEmd(:,i);
        end
    end
else
    synth_ = bandpass(base, 'bandstop');
end
scale =  var(baseline) / var(simulate);
synth_neg = synth_+simulate*sqrt(scale*power(10,-0.5));%0.3*normrnd(scale,scale/10);     -5dB
synth = synth_+simulate*sqrt(scale*power(10,-1));%normrnd(scale,scale/10);   -10dB
synth_data(:,k)=synth;
synth_neg_data(:,k)=synth_neg;
synth_mix(:,k)=lambda*synth+(1-lambda)*synth_neg;
synth_label(ind(1):ind(end),k)=1;
synth_mix_label(ind(1):ind(end),k)=lambda;
[S,t,f]=mtspecgramc(synth,movingwin,params);
energyDB = 10*log10(S);
DB_spindle = mean(energyDB(:,12:21),2);
pos_power(k)=DB_spindle(200);
[S,t,f]=mtspecgramc(synth_neg,movingwin,params);
energyDB = 10*log10(S);
DB_spindle = mean(energyDB(:,12:21),2);
neg_power(k)=DB_spindle(200);
%
if visual==1 
    figure;
    subplot(611);
    plot([1/fs:1/fs:len],base);
    axis tight;
    [S,t,f]=mtspecgramc(base,movingwin,params);
    subplot(612);
    plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
    [S,t,f]=mtspecgramc(synth,movingwin,params);
    energyDB = 10*log10(S);
    DB_spindle = mean(energyDB(:,12:21),2);
    disp(DB_spindle(200));
    subplot(614);
    plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
    subplot(613);
    plot([1/fs:1/fs:len],synth);
    hold on;
%     plot([1/fs:1/fs:len],synth_neg,'r');
    plot(([ind(1),ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
    plot(([len*fs-ind(1),len*fs-ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
    xlim([min(t) max(t)]);
    axis tight;
    %ylim([-100 100]);
    subplot(615);
    plot([1/fs:1/fs:len],synth_neg);
    xlim([min(t) max(t)]);
    axis tight;
    %ylim([-100 100]);
    subplot(616);
    [S,t,f]=mtspecgramc(synth_neg,movingwin,params);
    energyDB = 10*log10(S);
    DB_spindle = mean(energyDB(:,12:21),2);
    disp(DB_spindle(200));
    plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
%     subplot(514);
%     plot([1/fs:1/fs:len],synth_mix(:,k));
%     hold on;
%     plot(([ind(1),ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
%     plot(([ind(end),ind(end)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
%     xlim([min(t) max(t)]);
%     ylim([-40 40]);
%     title(['Lambda: ' num2str(lambda)]);
%     subplot(515);
%     [S,t,f]=mtspecgramc(synth_mix(:,k),movingwin,params);
%     plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
end
end
%%
test_data=reshape(synth_neg_data,[size(synth_neg_data,1)*size(synth_neg_data,2), 1]);
test_label=reshape(synth_label,[size(synth_label,1)*size(synth_label,2), 1]);
num_step = 50;
h1=fopen(['SleepSpindleData4RNN/test_synthetic(-10dB).txt'] , 'wt');
h2=fopen(['SleepSpindleData4RNN/test_synthetic(-10dB)_labels.txt'] , 'wt');
for i=1:length(test_data)-num_step+1
    fprintf(h1,'%f\n', detrend(test_data(i:i+num_step-1)));
    fprintf(h2,'%d\n', test_label(i:i+num_step-1));
end
%%
figure;histogram(pos_power);
hold on;histogram(neg_power);
synth_database.pos=synth_data;
synth_database.neg=synth_neg_data;
synth_database.label=synth_label;
synth_database.mix=synth_mix;
synth_database.mix_label=synth_mix_label;
save synth_database2 synth_database
%%





