clearvars;
addpath(genpath('chronux_2_11'));

num_subject = {1}; 
session={};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end

Spindles=cell(1,2);
database = zeros(length(session)*360000,1);
for idx=1:length(session)
    data_path = ['DatabaseSpindles/' session{idx} '.edf'];
    [hdr,record]=edfread(data_path);
    IndexC=strfind(hdr.label,'CZA1');
    Index = find(not(cellfun('isempty', IndexC)));
    if isempty(Index)
        IndexC=strfind(hdr.label,'C3A1');
        Index = find(not(cellfun('isempty', IndexC)));
    end
    EEG=record(Index,:)';
    sample_rate=hdr.frequency(Index);
    if sample_rate~=200
        EEG=interp(EEG,200/sample_rate);
        sample_rate=200;
    end
%     EEG = -1 + 2*(EEG-min(EEG))/(max(EEG)-min(EEG)); 
    f=fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    total_time = 30*60;
    C=textscan(f,'%f %f');
    C{1}=C{1}+total_time*(idx-1);    
    database(1+sample_rate*total_time*(idx-1):sample_rate*total_time*idx)=EEG;
end
fs = sample_rate;

%% remove spindle band 
baseline = bandpass(database,'bandstop',[9,16]);
len = 6;
movingwin = [1 1/fs];
params.Fs = fs;
params.fpass = [0 25];
params.tapers = [3 5];
params.err = 0;
params.pad = 0;
num_spindle = 300;
generate = 0;
if generate
    simulate = zeros(len*fs,num_spindle);
    label = zeros(len*fs,num_spindle);
    for k=1:num_spindle
        simulate(:,k) = simulate_spindle(fs, len, cell2mat(num_subject));
        [S,t,f] = mtspecgramc(simulate(:,k), movingwin, params);
        energyDB = 10*log10(S);
        DB_spindle = mean(mean(energyDB(:,12:21),2));
        while DB_spindle>-50
            simulate(:,k) = simulate_spindle(fs, len, cell2mat(num_subject));
            [S,t,f] = mtspecgramc(simulate(:,k), movingwin, params);
            energyDB = 10*log10(S);
            DB_spindle = mean(mean(energyDB(:,12:21),2));
        end
        ind = find(simulate(:,k)>0.4 | simulate(:,k)<-0.4);
        label(ind(1):ind(end),k) = 1;
    end
end
%%
std_value = zeros(num_spindle,1);
for k=1:num_spindle
    label_ind = find(label(:,k)==1);
    std_value(k) = std(simulate(label_ind,k));
end
pd = fitdist(std_value,'Normal');
scale = 13.8275/mean(pd);
simulate = simulate*scale;
tmp = reshape(baseline,[12000,30]);  %% duplicate the first 1 min that has no artifacts
for m=2:30
    tmp(:,m) = tmp(:,1);
end
synthetic_data = reshape(tmp,size(baseline)) + reshape(simulate,[length(baseline),1]);
synthetic_label = reshape(label,[length(baseline),1]);
% plot(synthetic_data(1:12000));hold on;plot(-100+synthetic_label(1:12000)*50);
%%  add noise
noise = randn(size(synthetic_data));  %10.06-0dB  5.65-5dB  17.9- -5dB  31.8- -10dB
snr(synthetic_data,noise*31.8)
synthetic_data = synthetic_data+noise*31.8;
%% compute power feature
ratio = zeros(size(synthetic_data));
power_BB = zeros(size(synthetic_data));
power_spindle = power_BB;
band_pass = power_BB;
up_env = power_BB;
power_feat = zeros(size(synthetic_data, 1), 19);
for k=1:length(synthetic_data)-floor(fs)+1
    seq = synthetic_data(k:k+floor(fs)-1);
    seq = (seq-mean(seq))/mean(pd);
    [SS,tt,ff]=mtspecgramc(seq,movingwin,params);
    energyDB = 10*log10(SS);
    %     DB_LF = mean(energyDB(:,3:8),2);
    DB_BB = mean(energyDB(:,4:11),2);
    DB_spindle = mean(energyDB(:,12:22),2);
    power_BB(k+floor(fs)/2)=DB_BB;
    power_spindle(k+floor(fs)/2)=DB_spindle;
    ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
    bdpass = bandpass(seq, 'bandpass',[9,16]);
    band_pass(k+floor(fs)/2)=bdpass(floor(fs)/2+1);
    [up,~]=envelope(bdpass);
    up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
    power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
end
synthetic_mf = [synthetic_data,band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
%%
save(['./data/synthetic' num2str(num_subject{idx}) 'Power_snr(-10dB)']);
%% writing to the file
format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
format2 = '%d\n';
h1=fopen(['SleepSpindleData4RNN/test_synthetic_snr_inf_mf_' session{idx} '_snr_-10dB.txt'] , 'wt');
h2=fopen(['SleepSpindleData4RNN/test_synthetic_snr_inf_mf_' session{idx} '_snr_-10dB_labels.txt'] , 'wt');
fprintf(h1,format1,synthetic_mf');
fprintf(h2,format2,synthetic_label);
fclose(h1);
fclose(h2);

%% testing 
f2=fopen('results/Synthetic_SNR/probability(synthetic_snr_inf_excerpt1).txt'); 
prob=fscanf(f2,'%f');
fclose(f2);
num_steps = 50;
prob=prob(num_steps:end);
stages = ones(size(EEG))+1;
onset = (find(diff(synthetic_label)==1)+1)/fs;
duration = find(diff(synthetic_label)==-1)/fs - onset;
Annotation_or = [onset,duration];
%%
prob_th = 0.9; %0.85;
time_th = 20;
if 0     
    range = [0.02:0.05:0.95, 0.96:0.01:1];
    sensitivity = zeros(length(range),1);
    specificity = zeros(length(range),1);
    fdr = zeros(length(range),1);
    f1score = zeros(length(range),1);
    for i=1:length(range)
        prob_th=range(i);
        [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_steps, 0);
    end
    area=0;
    for k=1:length(sensitivity)
        if k==1
            area=area+(sensitivity(k))*(specificity(k));
        else
            area=area+(sensitivity(k-1)+sensitivity(k))*(specificity(k)-specificity(k-1))/2;
        end
    end
    disp(['AUROC Area:  '  num2str(area)]);
    figure; plot(1- specificity,sensitivity,'.-');ylabel('sensitivity');xlabel('1-specificity');xlim([0,1]);ylim([0,1]);
    [~, I]=min((1-sensitivity).^2+(1-specificity).^2);
    disp(range(I));
else
    [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_steps, 1);
end
%% visualization
t_s=[];
t_e=[];
start_time = 193;  % seconds
window = 4;
end_time = start_time+window;
datarange=(start_time+1/fs:1/fs:end_time);
ind=find(onset>datarange(1) & onset<datarange(end));
if ~isempty(ind)
    for i=1:length(ind)
        t_s(i)=onset(ind(i));
        t_e(i)=t_s(i)+duration(ind(i));
    end
end

figure; 
subplot(311);
plot(datarange,synthetic_data(floor(datarange*fs)),'Linewidth',2);
set(gca,'ylim',[-100,100]);
hold on;
if ~isempty(ind)
    for ii=1:length(t_s)
        plot([t_s(ii),t_s(ii)],[-100 100],['r','-'],'linewidth',2);
        plot([t_e(ii),t_e(ii)],[-100 100],['r','-'],'linewidth',2);
    end
end
% plot(datarange,bandpass(synthetic_data(floor(datarange*fs)),'bandpass',[0.5,2]),'Linewidth',2,'color','k');
% det = find(smooth1(floor(datarange*fs)-49)==1);
% tmp = synthetic_data(floor(datarange*fs)); 
% plot(datarange(det),tmp(det),'Linewidth',2,'color','r');
% [pxx,f]=pspectrum(bandpass(tmp(det),'bandpass',[9,16]),fs);[M,I]=max(pxx);title(['Spindle frequency: ' num2str(f(I))]);

xlim([min(datarange)+0.5, max(datarange)-0.5]);
ylabel('Amptitude (uV)');
set(gca,'fontsize',24);
grid on;
subplot(312)
[S,t,f]=mtspecgramc(synthetic_data(floor(datarange*fs)),movingwin,params);colormap('jet');     
plot_matrix(S,t,f);
% spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
hold on 
for i=1:length(t_s)
    plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
    plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
end
plot([0.5,window+.5],[11 11],['b','-'],'linewidth',2);
plot([0.5,window+.5],[16 16],['b','-'],'linewidth',2);
set(gca,'fontsize',24);
grid on;
subplot(313);
plot(datarange,smooth1(floor(datarange*fs)-49),'linewidth',2,'Color','r');
ylim([-1,4]);
xlim([min(datarange)+0.5, max(datarange)-0.5]);
ylabel('Predictions');
set(gca,'fontsize',24);
%     xlabel('Time (s)');
grid on;

hold on;
plot(datarange,prob(floor(datarange*fs)-49),'linewidth',2,'Color','b');   %prob
ylim([-1,2]);
ylabel('Probabilities');
xlabel('Time (s)');
set(gca,'fontsize',24);
xlim([min(datarange)+0.5, max(datarange)-0.5]);
legend('Prediction','Probability');
grid on;