clear;
addpath(genpath('TStoolbox'));
addpath(genpath('chronux_2_11'));
addpath(genpath('RatData'));

%session = 'Earth11_080618_1_pre_S1';
%list_txt = dir('results/pain_sleep/eeg/probability_EEG?_1206*.txt');
list_txt = dir('results/pain_sleep/eeg/probability_Baoling_Animal2.txt');
sham_audio_present = 0;
%% this loads the data to a  variable named EEG
for s = 1:1 %length(list_txt)
    
    fs = 200;
    
    session = 'Baoling_Animal2';
    %session = strsplit(list_txt(s).name, 'probability_');
    %session = session{2}(1:end-4);
    
    
    %session = 'S1#1_080618_2_pre_S1'; %'S1#2_082418_1_postday10_S1'; %
    %session = 'Earth11_080618_1_pre_ACC';
    EEG = load(['data/pain_sleep/', session, '.mat']);
    %EEG = EEG.AS_ch8_bp; %EEG;
    EEG = EEG.EEG;
    cut_ = length(EEG); %
    EEG_bp = EEG(1:cut_); 
    raw_data{s, :} = EEG_bp(1:cut_);
    
   
    f2 = fopen(['results/pain_sleep/eeg/probability_' session '.txt']);
    prob = fscanf(f2,'%f'); 
    cut_2 = length(prob);
    prob = prob(1:cut_2);
    fclose(f2);
    prob_data{s, :} = prob;
    
    % Readiing the output of load_model.py
    prob_thresh = 0.3; % 0.3;
    time_thresh = 0.1;
    
    prob = smoothdata(prob, 'gaussian', 3); %50);
    
    smooth1 = makeBinary(prob, prob_thresh, time_thresh*fs);
    smooth1_data{s, :} = smooth1;

    ind = find(diff(smooth1)==1);
    ind = ind+1;
    ind1 = find(diff(smooth1)==-1);
    if smooth1(1)==1
        ind = [1; ind];
    end
    if smooth1(end)==1
        ind1 = [ind1; length(smooth1)];
    end

    [avgFreqs, avgDurs, avgPows, freqMat, durMat, powMat] = getSpindleStats(EEG_bp, [ind,ind1], fs);

    dbMat = 10*log10(powMat);

    spindle_density = length(freqMat)/(length(EEG_bp)/fs/60);
    eval.session = session;
%     eval.std = std(EEG_bp);
%     eval.mean = mean(EEG_bp);
    eval.spindle_density = spindle_density;
    eval.num_spindles = length(freqMat);
    eval.sleep_time = length(EEG_bp)/fs/60;
%     eval.
    disp(eval);
%     
    out_mat = [freqMat, durMat, dbMat];
    if ~isempty(out_mat)
        xlswrite(['results/pain_sleep/eeg/spindle_stats_' session '.xlsx'], out_mat);
    end
% %end
% %%
    movingwin=[0.5 .005];
    params.Fs=fs;
    params.fpass=[0 50];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    start_time = 1;  % seconds
    end_time = start_time + length(raw_data{s, :})/200 - 10;
    datarange{s, :} = (start_time+1/fs:1/fs:end_time);
    [S,t,f] = mtspecgramc(EEG_bp(floor(datarange{s, :}*fs)),movingwin,params);colormap('jet');
    spec_data{s} = S;
    spec_t{s, :} = t;
    spec_f{s, :} = f;
    
end         
    %%    
if sham_audio_present
%     fileID = fileread('R:\jwanglab\jwanglabspace\Bassir\sleep_pain\post_cfa\EEG1\12-15-18\sham stimulus\notes.txt');
%     times = regexp(fileID,"\d{2}:\d{2}:\d{2} (asleep|awake)",'match');
%     events = contains(times,"awake");
%     rhdtime = zeros(length(times),1)';
% 
%     date = '15-Dec-2018 ';
%     timetransform = @(time) posixtime(datetime([date time]));
%     for i=1:length(times)
%         rhdtime(i) = timetransform(char(extractBetween(string(times(i)),1,8)))*1000;
%         % remember to output in ms
%     end
% 
%     % audacity default is 44.1 kHz sample for 50ms x 44.1 kHz = 2205 samples
%     % each sample is 
% %     [pinky,pinkFs] = audioread('R:\jwanglab\jwanglabspace\Bassir\sleep_pain\sleep BMI\pink noise\20minsham50.wav');
    load('R:\jwanglab\jwanglabspace\Bassir\sleep_pain\post_cfa\EEG3\12-20-18\shamstimulus\manually combined sessions\audio.mat')
        pinky = combinedsound;
    pinkFs = 44100;
%     relativetimes = (rhdtime-rhdtime(1))/1000;
%     sum_ = 0;
%     for i=1:2:length(relativetimes)
%         sum_ = sum_ + relativetimes(i+1)-relativetimes(i);
%     end
% 
%     multiples = sum_*pinkFs/length(pinky);
% 
%     for i=1:multiples
%         pinky = [pinky; pinky];
%     end
    pinky_ = pinky; %pinky(1:sum_*pinkFs);
    pinky_range = (start_time+1/pinkFs:1/pinkFs:end_time);

%     t = 0:seconds(1/pinkFs):seconds(sum_);
%     t = t(1:end-1);
%     plot(t,pinky(1:sum_*pinkFs))
%     xlabel('Time')
%     ylabel('Audio Signal')
end
%%
figure;
x1 = subplot(611);
range1 = datarange{1, :};
data1 = raw_data{1, :};
plot(range1, data1(floor(range1*fs)), 'LineWidth', 1);
axis tight;
hold on;
plot(range1, bandpass(data1(floor(range1*fs)),'bandpass', [9 16]));
title(['Channel : ' num2str(1) ' ']);
%ylim([-100, 100]);
ylim([-0.2, 0.2]);

x2 = subplot(612);
smooth1 = smooth1_data{1, :};
prob1 = prob_data{1, :};
plot(range1, smooth1(floor(range1*fs)), 'LineWidth',1);
ylim([-0.001, 1.1]);
ylabel('Predictions');
xlabel('Time (s)');
set(gca,'fontsize',16);  
grid on; hold on;
plot(range1, prob1(floor(range1*fs)));   %prob

x3 = subplot(613);
plot_matrix(spec_data{1}, spec_t{1, :}, spec_f{1, :}); colormap('jet');

% x4 = subplot(614);
% range2 = datarange{2, :};
% data2 = raw_data{2, :};
% plot(range2, data2(floor(range2*fs)), 'LineWidth', 1);
% axis tight;
% hold on;
% plot(range2, bandpass(data2(floor(range2*fs)),'bandpass', [9 16]));
% title(['Channel : ' num2str(1) ' ']);
% if sham_audio_present
%     hold on;
%     plot(pinky_range, pinky_(floor(pinky_range*pinkFs))*500, 'LineWidth', 1);
% end
% 
% x5 = subplot(615);
% smooth2 = smooth1_data{2, :};
% prob2 = prob_data{2, :};
% plot(range2, smooth2(floor(range2*fs)), 'LineWidth',1);
% ylim([-0.001, 1.1]);
% ylabel('Predictions');
% xlabel('Time (s)');
% set(gca,'fontsize',16);
% grid on; hold on;
% plot(range2, prob2(floor(range2*fs)));   %prob
% 
% x6 = subplot(616);
% plot_matrix(spec_data{2}, spec_t{2, :}, spec_f{2, :}); colormap('jet');

%linkaxes([x1, x2, x3, x4, x5, x6], 'x')
linkaxes([x1, x2, x3], 'x')

