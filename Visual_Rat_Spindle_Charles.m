clear;
addpath(genpath('TStoolbox'));
addpath(genpath('chronux_2_11'));
session = 'Dino_072414_mPFC';
load([ 'RatData/' session '_BasicMetaData.mat']);
load([ 'RatData/' session '_Spindles.mat']);    

%% this loads the data to a  variable named EEG
%[EEG, ~]= LoadBinary_bw(['RatData/' session '.eeg'], bmd.goodeegchannel);
%[EEG2, ~]= LoadBinary_bw(['RatData/' session '.eeg'], bmd.goodeegchannel);
load([session '_EEG.mat']);
load([session '_EEG2.mat']);
%% Prev section ^ takes a long time
EEG = EEG * bmd.voltsperunit * 1e6;   %% convert to microvolt
fs=bmd.Par.lfpSampleRate;
% notch filter  remove power-line interfence 60Hz
wo = 60/(fs/2);  bw = wo/35;
[b,a] = iirnotch(wo,bw);
EEG= filtfilt(b,a,EEG);

target_fs = 200;
if fs~=200
    EEG=resample(EEG, target_fs, fs);
    fs=target_fs;
end
%% Compute average standard deviation for normalization
EEG_bp = bandpass(EEG,'bandpass', [2, 50]);
C=cell(1,2);
C{1}=SpindleData.normspindles(:,1);
C{2}=SpindleData.normspindles(:,3);
std_value = zeros(size(C{1}, 1),1);
for i=1:size(C{1}, 1)
    s_data=EEG_bp(floor(C{1}(i)*fs):floor(C{2}(i)*fs));
    s_data = detrend(s_data);
    std_value(i) = std(s_data);
end
pd = fitdist(std_value,'Normal');
disp(mean(pd));

%%
switch session
    case 'Dino_061814_mPFC'
        secA = [2000, 5000];
        secB = [14000, 17000];
    case 'Dino_062014_mPFC'
        secA = [4000, 9000];
        secB = [19000, 22000];
    case 'Dino_061914_mPFC'
        secA = [10000 13000];
        secB = [];
    case 'BWRat21_121813'
        secA = [3400 6600];
        secB = [10000 22000];
    case 'Dino_072314_mPFC'
        secA = [1600 6300];
        secB = [];
    case 'Dino_072414_mPFC'
        secA = [3300 4800];
        secB = [];
    case 'Bogey_012615'
        secA = [4300 7870];
        secB = [9700 23300];
end

EEG_bp = EEG_bp';

if ~isempty(secB)
    EEG_bp=[EEG_bp(secA(1)*fs:secA(2)*fs); EEG_bp(secB(1)*fs:secB(2)*fs)];  %ignore the first 4000 secs in EEG recording2
else
    EEG_bp=EEG_bp(secA(1)*fs:secA(2)*fs);
end
% fix the label
if ~isempty(secB)
    index = find(C{1}>secB(1), 1 );
    C{1}(1:index-1)=C{1}(1:index-1)-secA(1);
    C{2}(1:index-1)=C{2}(1:index-1)-secA(1);
    C{1}(index:end)=C{1}(index:end)-secB(1)+secA(2)-secA(1);
    C{2}(index:end)=C{2}(index:end)-secB(1)+secA(2)-secA(1);
else
    C{1}=C{1}-secA(1);
    C{2}=C{2}-secA(1);
end
%%
movingwin=[0.5 .005];
params.Fs=fs;
params.fpass=[0 50];
params.tapers=[3 5];
params.err=0;
params.pad=0;
[SS,tt,ff]=mtspecgramc(EEG_bp,movingwin,params);
energyDB = 10*log10(SS); 
DB_BB = mean(energyDB(:,4:11),2);
DB_spindle = mean(energyDB(:,12:21),2);
%% Writing bandpass data for no reason
% h1 = fopen(['RatData/test_data/' session 'hp.txt'], 'wt');  
% h2 = fopen(['RatData/test_data/' session '_hp_labels.txt'], 'wt');  
% % EEG_bp1 = bandpass(EEG_bp,'bandpass', [9,15]);
% num_step=50;
% labels = zeros(size(EEG_bp1));
% for k=1:length(C{1})
%     range = [floor(C{1}(k)*fs):floor(C{2}(k)*fs)];
%     labels(range)=1;
% end
% 
% fprintf(h1, '%f\n', EEG_bp1);
% fprintf(h2, '%d\n', labels);
% fclose(h1);
% fclose(h2);
%% Readiing the output of load_model.py
prob_th = 0.5;
time_th = 20;
f2=fopen(['results/Rats/probability_' session '_bp2_50.txt']);
prob=fscanf(f2,'%f');   
fclose(f2);
% predict=(prob>prob_th);
% ind = find(diff(predict)==1);
% ind = ind+1;
% ind1 = find(diff(predict)==-1);
% ind1 = ind1+1;
% smooth = predict;
% for i=1:length(ind)
%   if sum(predict(ind(i):ind1(i))) < thresh
%       smooth(ind(i):ind1(i)) = 0;
%   end
% end
start = C{1};
duration = C{2}-C{1};

[sens, spec, fdr, f1score, result, fp_result, smooth1,eval,probabilty]=eval_performance(prob, fs, tt, DB_spindle, DB_BB,  ones(size(prob))+1, [C{1},C{2}-C{1}], prob_th, time_th, 1, 1);
%
%% PLOTTING PT 1

start_time = 30;  % seconds
window = 1400; % 30 default for normal view
gap = 1000;

t_s=[];
t_e=[];
figure;
end_time = start_time + window;
datarange=(start_time+1/fs:1/fs:end_time);
params.fpass = [5 20];
[S,t,f]=mtspecgramc(EEG_bp(floor(datarange*fs)),movingwin,params);colormap('jet');
ind=find(start>datarange(1) & start<datarange(end));
if ~isempty(ind)
    for i=1:length(ind)
        t_s(i)=start(ind(i));
        t_e(i)=t_s(i)+duration(ind(i));
    end
end
%EEG_bp1 = bandpass(EEG_bp,'bandpass', [2, 99]);
EEG_bp1 = EEG_bp;
% set(gca,'ylim',[-1000,1000]);
%axis tight;
grid on;

ax2 = subplot(4, 1, 4);
plot_matrix(S,t,f); % SPECTROGRAM
colorbar('off');
hold on 
for i=1:length(t_s)
    plot([t_s(i),t_s(i)]-datarange(1),[0 50],['k','-'],'linewidth',1);
    plot([t_e(i),t_e(i)]-datarange(1),[0 50],['k','-'],'linewidth',1);
end
plot([0  30],[9 9],['b','-'],'linewidth',1);
plot([0 30],[15 15],['b','-'],'linewidth',1);
grid on;
xticks(start_time:10:window);

ax1 = subplot(4, 1, (1:3));

plot(datarange-datarange(1),EEG_bp1(floor(datarange*fs))); % RAW DATA

hold on

params.fpass = [0 50];
EEG_bandpassed = bandpass(EEG_bp1(floor(datarange*fs)),'bandstop', [9 16]);
plot(datarange-datarange(1), EEG_bandpassed - gap); % BANDPASSED

hold on

plot(datarange-datarange(1), smooth1(floor(datarange*fs)-49)*500 - 2*gap, 'm'); % wlab detections
plot(datarange-datarange(1), prob(floor(datarange*fs)-49)*500 - 2*gap);   %wlab probs

for i=1:length(t_s) % BOXING ANNOTATOR SPINDLES
    spinMin = min(EEG_bp1( floor(t_s(i)*fs) : floor(t_e(i)*fs) ));
    spinMax = max(EEG_bp1( floor(t_s(i)*fs) : floor(t_e(i)*fs) ));
    plot([t_s(i),t_s(i)]-datarange(1), [spinMin spinMax], 'k');
    plot([t_e(i),t_e(i)]-datarange(1), [spinMin spinMax], 'k');
    plot([t_s(i),t_e(i)]-datarange(1), [spinMin spinMin], 'k');
    plot([t_s(i),t_e(i)]-datarange(1), [spinMax spinMax], 'k');
    
    spinMin = min(EEG_bandpassed( (floor(t_s(i)*fs) : floor(t_e(i)*fs)) -datarange(1)*fs ));
    spinMax = max(EEG_bandpassed( (floor(t_s(i)*fs) : floor(t_e(i)*fs)) -datarange(1)*fs ));
    plot([t_s(i),t_s(i)]-datarange(1), [spinMin spinMax] - gap, 'k');
    plot([t_e(i),t_e(i)]-datarange(1), [spinMin spinMax] - gap, 'k');
    plot([t_s(i),t_e(i)]-datarange(1), [spinMin spinMin] - gap, 'k');
    plot([t_s(i),t_e(i)]-datarange(1), [spinMax spinMax] - gap, 'k');
end

wangStart = 0;
for i=datarange(1)*fs + 1:datarange(length(datarange))*fs % BOXING WANGLAB SPINDLES
    if smooth1(i) ~= 0 % i.e. if wang spindle detection
        if wangStart == 0 % i.e. start of spindle
            wangStart = i;
        end
    else
        if wangStart ~= 0 % i.e. end of spindle
            spinMin = min(EEG_bp1( wangStart:i ));
            spinMax = max(EEG_bp1( wangStart:i ));
            plot([(wangStart/fs)-datarange(1) (wangStart/fs)-datarange(1)], [spinMin spinMax], 'm');
            plot([(i/fs)-datarange(1) (i/fs)-datarange(1)], [spinMin spinMax], 'm');
            plot([(wangStart/fs)-datarange(1) (i/fs)-datarange(1)], [spinMin spinMin], 'm');
            plot([(wangStart/fs)-datarange(1) (i/fs)-datarange(1)], [spinMax spinMax], 'm');

            spinMin = min(EEG_bandpassed( (wangStart:i)-datarange(1)*fs ));
            spinMax = max(EEG_bandpassed( (wangStart:i)-datarange(1)*fs ));
            plot([(wangStart/fs)-datarange(1) (wangStart/fs)-datarange(1)], [spinMin spinMax] - gap, 'm');
            plot([(i/fs)-datarange(1) (i/fs)-datarange(1)], [spinMin spinMax] - gap, 'm');
            plot([(wangStart/fs)-datarange(1) (i/fs)-datarange(1)], [spinMin spinMin] - gap, 'm');
            plot([(wangStart/fs)-datarange(1) (i/fs)-datarange(1)], [spinMax spinMax] - gap, 'm');
            
            wangStart = 0;
        end
    end
    
end

legend({"Data", "Bandpass", "Wanglab Detections", "Wanglab Probabilities", "Annotator Detections"},... % Legend Names
        [10 , 250, 150, 75]  );

grid on;

xticks(start_time:10:window);
linkaxes([ax1, ax2], 'x');

%% SAVE TO FILE
fig = gcf;
fig.PaperUnits = 'points';
fig.PaperPosition = [0 0 1050*window/60 1000]; % ~1050 pixels per minute, width wise
%ax.XTickMode = 'manual'; % Ensures axes remain intact
print('BigFig','-dpng','-r0')

%% Comparison between of duration and power of FPs and GT


figure;subplot(211);
histogram(fp_result{3});
hold on;histogram(result{5});
legend('FP','GT');
%     temp1 = result{2};
%     temp2 = result{4};
%     temp2(temp1==999)=[];
%     temp1(temp1==999)=[];
%     histogram(DB_spindle(floor((temp1+temp2/2-tt(1))*fs)));
xlabel('Normalized Power (DB)');
ylabel('Number');
set(gca,'fontsize',32);

subplot(212);histogram(fp_result{2});
hold on;histogram(duration);
xlabel('Duration (s)');
ylabel('Number');
set(gca,'fontsize',32);
legend('FP','GT');

figure;
%scatter(fp_result{2}, fp_result{3});
%hold on;
scatter(duration, result{5});
xlabel('Duration (s)');
ylabel('Power (dB)');
legend('FP','GT');