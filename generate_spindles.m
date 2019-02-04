clearvars;
addpath(genpath('chronux_2_11'));

num_subject = {8}; 
session={};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end
%% Appending data from 'session' together

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
len = 6; % length of synthetic spindles in seconds (add spindle every 'len' seconds)
movingwin = [1 .005];
params.Fs = fs;
params.fpass = [0 50];
params.tapers = [3 5];
params.err = 0;
params.pad = 0;
visual = 0;

% %% get the spindle max amptitude
% amp=zeros(1,size(C{1}, 1));
% for i=1:size(C{1}, 1)
%     s_data=EEG(floor(C{1}(i)*fs):floor((C{1}(i)+C{2}(i))*fs));
%     out = bandpass(s_data);
%     amp(i) = max(out);
% end

% Replacing all spindles annotated by experts with NAN
baseline = database;
sp_ind = [];
real_label = zeros(size(database));
std_real = [];
re_pow = [];
re_range = [];
for i = 1:length(C{1})
    sp_start = floor(sample_rate*C{1}(i));
    sp_end = floor(sample_rate*(C{1}(i)+C{2}(i)));
    real_label(sp_start:sp_end) = 1;
    std_real = [std_real, std(detrend(database(sp_start:sp_end)))];
    
    re_sp = bandpass(database(sp_start:sp_end),'bandpass', [9,16]);
    re_range = [re_range, (max(re_sp)-min(re_sp))/2];
    
    
    baseline(sp_start:sp_end) =...
        bandpass(baseline(sp_start:sp_end),'bandstop', [9,16]);
    
    sp_ind = [sp_ind, floor((sp_start+sp_end)/2)];
end
std_real = mean(fitdist(std_real', 'Normal'));


for s = 1:length(sp_ind)
    if sp_ind(s) <= len*fs/2 % Can't deal with spindles within first len/2 seconds of the data
        continue;
    end
     
    [S,t,f] = mtspecgramc(baseline(floor(sp_ind(s)-len*fs/2):floor(sp_ind(s)+len*fs/2)-1), movingwin, params);
    energyDB = 10*log10(S);
    DB_spindle = mean(energyDB(:,12:21),2);
    re_pow = [re_pow, mean(DB_spindle)];
     
end

%Extend baseline by repeating it to add more synthetic spindles
% aug = 2;
% big_data = zeros(aug*length(baseline),1);
% big_ind = [];
% for i = 0:aug-1
%     big_data(1+i*length(baseline):i*length(baseline)+length(baseline)) = baseline;
%     big_ind = [big_ind, sp_ind + i*length(baseline)];
% end
% baseline = big_data;
% sp_ind = big_ind;
%% 
num_synth = length(sp_ind); %floor(length(baseline)/len/fs); % number of spindles to add
%synth_data = zeros(len*fs, num_synth); % "normal" spindles
%synth_label = synth_data;
synth_label = zeros(size(baseline));

% for generating mixed labels
%synth_mix=synth_label;
%synth_mix_label=synth_mix;

% for visualization of spindle power
pos_power = [];
% rParabEmd = rParabEmd__L (baseline, 50, 50, 1); %For removing spindle band using EMD
base_sp = baseline;
std_sp = [];

for k = 1:num_synth
    
    simulate = simulate_spindle(fs, len, cell2mat(num_subject));
    ind = find(simulate>0.4 | simulate<-0.4);
    %base = baseline(1+len*(k-1)*fs:(len*k)*fs);
    if sp_ind(k) <= len*fs/2 % Can't deal with spindles within first len/2 seconds of the data
        continue;
    end
    base = baseline(floor(sp_ind(k)-len*fs/2):floor(sp_ind(k)+len*fs/2)-1);
    %base(isnan(base)) = 0;
    
    %synth = zeros(fs*len, 1);
    
    %lambda=betarnd(0.2,0.2);
    %rParabEmd = rParabEmd__L (base, 50, 50, 1);
    %synth_neg = synth;
    scale = max(base); %max(max(rParabEmd));
    %sc2 = 0.1; %1; % 1.5
    sc2 = re_range(k);
    %for i=1:1 %size(rParabEmd,2)
        %if i==2
            synth = simulate*sc2; %*normrnd(scale,scale/10); 
            %synth = synth + sc_sp;
            %synth_neg = synth_neg+simulate*0.3*normrnd(scale,scale/10); %low power spindle
        %else
            %synth = synth+rParabEmd(:,i);
            %synth_neg = synth_neg+rParabEmd(:,i);
        %end
    %end
    
    [S,t,f] = mtspecgramc(synth, movingwin, params);
    energyDB = 10*log10(S);
    DB_spindle = mean(energyDB(:,12:21),2);
     
    if mean(DB_spindle) > -5
        while 1 
            simulate = simulate_spindle(fs, len, cell2mat(num_subject));
            ind = find(simulate>0.4 | simulate<-0.4);
            synth = simulate*sc2; %*normrnd(scale,scale/10); 
            [S,t,f] = mtspecgramc(synth, movingwin, params);
            energyDB = 10*log10(S);
            DB_spindle = mean(energyDB(:,12:21),2);
            
            if mean(DB_spindle) < -5
                break;
            else
                disp('Skipped');
                continue;
            end
        end
    end
    disp(mean(DB_spindle));
        
    base_sy = base + synth;
    std_sp_cur = std(detrend(base_sy(ind(1):ind(end))));
    std_ratio = 1; %std_real/std_sp_cur;
        
    base_sp(floor(sp_ind(k)-len*fs/2):floor(sp_ind(k)+len*fs/2)-1) =...
        std_ratio*(base_sp(floor(sp_ind(k)-len*fs/2):floor(sp_ind(k)+len*fs/2)-1) + synth);
    
    std_sp = [std_sp, std(detrend(base_sp(floor(sp_ind(k)-len*fs/2)+ind(1):floor(sp_ind(k)-len*fs/2)+ind(end))))];
    
    
    [S,t,f] = mtspecgramc(base_sp(floor(sp_ind(k)-len*fs/2):floor(sp_ind(k)+len*fs/2)-1), movingwin, params);
    energyDB = 10*log10(S);
    DB_spindle = mean(energyDB(:,12:21),2);
   
    pos_power = [pos_power, mean(DB_spindle)];
    
    %synth_data(:,k) = synth;
    %synth_neg_data(:,k)=synth_neg;
    %synth_mix(:,k)=lambda*synth+(1-lambda)*synth_neg;
    %synth_label(ind(1):ind(end), k) = 1;
    synth_label(floor(sp_ind(k)-len*fs/2)+ind(1):floor(sp_ind(k)-len*fs/2)+ind(end)) = 1;
    %synth_mix_label(ind(1):ind(end),k)=lambda;
    
    
%     [S,t,f]=mtspecgramc(synth_neg,movingwin,params);
%     energyDB = 10*log10(S);
%     DB_spindle = mean(energyDB(:,12:21),2);
%     neg_power(k)=DB_spindle(200);

    if visual == 1 && k == 20
       
        figure;
        [S,t,f]=mtspecgramc(base,movingwin,params);
        subplot(211);
        plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
        
        [S,t,f]=mtspecgramc(synth,movingwin,params);
        energyDB = 10*log10(S);
        DB_spindle = mean(energyDB(:,12:21),2);
        %disp(mean(DB_spindle));
        subplot(212);
        plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
        
        figure;
%         subplot(111);
        plot([1/fs:1/fs:len],base + synth);
        hold on;
%         plot([1/fs:1/fs:len],synth_neg,'r');
        plot(([ind(1),ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
        plot(([len*fs-ind(1),len*fs-ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
        xlim([min(t) max(t)]);
        ylim([-40 40]);
%         subplot(512);
%         plot([1/fs:1/fs:len],synth_neg);
%         xlim([min(t) max(t)]);
%         ylim([-40 40]);     
%         subplot(513);
%         [S,t,f]=mtspecgramc(synth_neg,movingwin,params);
%         energyDB = 10*log10(S);
%         DB_spindle = mean(energyDB(:,12:21),2);
%         disp(DB_spindle(200));
%         plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
        %subplot(514);
        %plot([1/fs:1/fs:len],synth_mix);
        %hold on;
        %plot(([ind(1),ind(1)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
        %plot(([ind(end),ind(end)])/fs,[min(synth) max(synth)],['r','-'],'linewidth',1);
        %xlim([min(t) max(t)]);
        %ylim([-40 40]);
        %title(['Lambda: ' num2str(lambda)]);
        %subplot(515);
        %[S,t,f]=mtspecgramc(synth_mix,movingwin,params);
        %plot_matrix(S,t,f);caxis([-40 15]);colormap('jet');
    end
end
% %%
% test_data=reshape(synth_data,[size(synth_data,1)*size(synth_data,2), 1]);
% test_label=reshape(synth_label,[size(synth_label,1)*size(synth_label,2), 1]);
% num_step = 50;
% h1=fopen(['SleepSpindleData4RNN/test_synthetic.txt'] , 'wt');
% h2=fopen(['SleepSpindleData4RNN/test_synthetic_labels.txt'] , 'wt');
% for i=1:length(test_data)-num_step+1
%     fprintf(h1,'%f\n', detrend(test_data(i:i+num_step-1)));vim
%     fprintf(h2,'%d\n', test_label(i:i+num_step-1));
% end
%%

% Real data + synthetic data
base_sp = [database; base_sp];
synth_label = [real_label; synth_label];
%base_sp = base_sp;
%synth_label = synth_label;

synth_database.std = mean(fitdist([std_real, std_sp]', 'Normal')); %mean(fitdist(std_sp', 'Normal'));
synth_database.label = synth_label;
synth_database.data = base_sp;
% synth_database.neg=synth_neg_data;
% synth_database.mix=synth_mix;
% synth_database.mix_label=synth_mix_label;
%%
%save('synth_DREAMS_1-2_5-8', 'synth_database');
save(['re_synth_DREAMS_' num2str(num_subject{1})], 'synth_database');