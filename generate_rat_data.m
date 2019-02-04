clc;
clear; % is a7 and NN running the same data
addpath(genpath('TStoolbox'));
addpath(genpath('chronux_2_11'));
addpath(genpath('Sleep_spindles_FHN2015'));

% CHANGE THIS VALUE
%session = 'Earth10_081018_2_pre'; % format: [Animal name]_[Date]
%'S1#1_080818_1_pre', 'S1#1_080618_2_pre',
sessions = { 'EEG4_123118_35_post'};
folder_paths = {'R:\jwanglab\jwanglabspace\Bassir\sleep_pain\post_cfa\EEG4\12-31-18\'};
%     'S1#2_080718_1_pre', 'S1#2_080818_2_pre',...
%     'S1#1_081418_1_postday1', 'S1#1_081418_2_postday1', 'S1#2_081518_1_postday1', 'S1#1_082018_1_postday7',...
%     'S1#1_082018_2_postday7', 'S1#2_082118_1_postday7', 'S1#1_082318_1_postday10', 'S1#2_082418_1_postday10'};

%load(['RatData/' session '_BasicMetaData.mat']);
%load(['RatData/' session '_Spindles.mat']);    
%%
% this loads the data to a  variable named EEG
%[EEG, ~]= LoadBinary_bw(['RatData/' session '.eeg'], bmd.goodeegchannel);
%load('R:\jwanglab\jwanglabspace\David Rosenberg\Earth10\7_11_2018\LFP_1\Earth10_5-22-2018_250_WithLFP.mat')
%load('/media/share/jwanglab/jwanglabspace/David Rosenberg/Earth10/7_11_2018/LFP_1/Earth10_5-22-2018_250_WithLFP.mat');

% RAT LFP DATA
 %{'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\before cfa\8_8_2018_Natalie_#1_sleep recording_ for s1 only_\',...
  %  'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\before cfa\8_6_2018_Minjung_Natalie_#1_sleep recording_ for s1 only_180806_144436\',...

    % 'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\before cfa\8_7_2018_Minjung_Natalie_#2_sleep recording_ for s1 only\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\before cfa\8_8_2018_Natalie_#2_sleep recording_ for s1 only_\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\POST CFA DAY 1\8_14_2018_Natalie_Minjung_#1_S1 only_sleep recording_1 day after CFA\SESSION2\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\POST CFA DAY 1\8_14_2018_Natalie_Minjung_#1_S1 only_sleep recording_1 day after CFA\SESSION3\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\POST CFA DAY 1\8_15_2018_Natalie_Minjung_#2_S1 only_sleep recording_1 day after CFA\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\post CFA day 7\8_20_2018_haocheng #1 s1 only post CFA day7\session1\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\post CFA day 7\8_20_2018_haocheng #1 s1 only post CFA day7\session2\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\post CFA day 7\8_21_2018_haocheng #2 s1 only post CFA day7\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\post CFA day 10\2018_8_23_haocheng #1 s1 only post CFA day 10\',...
%     'R:\jwanglab\jwanglabspace\Haocheng Zhou\rat sleep project\after CFA\S1\post CFA day 10\8_24_2018_haocheng #2 s1 only post CFA day10\'};

for s = 1:length(sessions)
    load([folder_paths{s}, 'sleep.mat']);
    sleep_time = xlsread([folder_paths{s}, 'laserOn_sleep.xls']);
    session = sessions{s};
    fs = 1000;
    %% Extract sleep segments

    sleep_ind = find(~sleep_time(:,2));

    sleep_tracks = [];
    for s1 = 1:length(sleep_ind)
        sleep_start = sleep_time(sleep_ind(s1),1)*1e-3*fs;
        sleep_end = sleep_time(sleep_ind(s1)+1)*1e-3*fs;
        sleep_tracks = [sleep_tracks, amplifier_data_lfp(:, sleep_start:sleep_end)];
    end
    range = length(sleep_tracks);
    %%
    % VERIFY
    n_chan = 32;  %32
    end_chan = 24;
    
    s1_channels = [19, 34]; % this is critical, consult original RHD channel map
    pfc_channels = [15,16,17,18];
    pfc_channels = [18,18]; % remove latter noisy channel for 12/09/18 data
    
    %S1_lfp_data = zeros(range, end_chan - n_chan);

    % for chan1=1:n_chan
    %     %data1 = allUnits.LFP(chan1).LFP(1:range)*Fs;
    %     data1 = amplifier_data_lfp(chan1,:); %*Fs;
    %     %data1 = filtfilt(b,a,data1); data1 = filtfilt(b2,a2,data1);       
    %     ACC_lfp_data(:,chan1) = data1;       
    % end

    % Default fill S1 channels (all channels after n_chan+1)
    % for chan2=n_chan+1:length(amplifier_data_lfp(:,1))
    %     %data2 = allUnits.LFP(chan2).LFP(1:range)*Fs;
    %     data2 = amplifier_data_lfp(chan2,:)*Fs;
    %     %data2 = filtfilt(b,a,data2); data2 = filtfilt(b2,a2,data2);
    %     S1_lfp_data(:,chan2-n_chan) = data2;
    % end

    % EDIT THIS FOR IRREGULAR CHANNEL NUMBERS
    % for chan2=n_chan+1:end_chan
    %     %data2 = allUnits.LFP(chan2).LFP(1:range)*Fs;
    %     data2 = amplifier_data_lfp(chan2,:)*Fs;
    %     %data2 = filtfilt(b,a,data2); data2 = filtfilt(b2,a2,data2);
    %     S1_lfp_data(:,chan2-n_chan) = data2;
    % end

    %S1_lfp_data = sleep_tracks(1:n_chan, :)';
    S1_lfp_data = sleep_tracks(s1_channels, :)';
    coeff2 = pca(S1_lfp_data);
    data2 = S1_lfp_data*coeff2(:,1);

    if n_chan > 8

        %ACC_lfp_data = sleep_tracks(1:n_chan, :)';
        ACC_lfp_data = sleep_tracks(pfc_channels, :)';
        coeff1 = pca(ACC_lfp_data);
%         data1 = ACC_lfp_data*coeff1(:,1); % comment out to remove PCA
        data1 = ACC_lfp_data(:,1);

        %S1_lfp_data = sleep_tracks(n_chan+1:end, :)'; 
        S1_lfp_data = sleep_tracks(s1_channels, :)';
        coeff2 = pca(S1_lfp_data);
%         data2 = S1_lfp_data*coeff2(:,1); % comment out to remove PCA
        data2 = S1_lfp_data(:,1);
    end

    % Default start time, last 15 minutes of file
    start_ = range - 10*60*fs;
    end_ = range;

    % Start time for Rat #1 (/media/share/jwanglab/jwanglabspace/Haocheng Zhou/rat sleep project/8_6_2018_Minjung_Natalie_#1_sleep recording_ for s1 only_180806_144436)
    % start_ = 38*60*Fs;
    % end_ = start_ + 15*60*Fs;
    %% time domain

    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 50];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;

    if 1
        if n_chan > 8
            figure;
            x1 = subplot(411); bp = bandpass(data1,'bandpass',[2, 50]);
            plot(bp(start_:end_),'r-','linewidth',1);
            set(gca,'fontsize',18);xlabel([]); ylabel('Ampl (uV)','fontsize',20);title('ACC','fontsize',24);axis tight;ylim([-2500 2500]);

            x2 =subplot(412); bp = bandpass(data2,'bandpass',[2, 50]);
            plot(bp(start_:end_),'r-','linewidth',1);
            set(gca,'fontsize',18),ylabel('Ampl (uV)','fontsize',20);title('S1','fontsize',24);axis tight;ylim([-2500 2500]);

            x3 =subplot(413);bp = bandpass(data1,'bandpass',[9, 16]);
            plot(bp(start_:end_),'r-','linewidth',1);
            set(gca,'fontsize',18);xlabel([]); ylabel('Ampl (uV)','fontsize',20);title('ACC','fontsize',24);axis tight;ylim([-500 500]);

            x4 = subplot(414);bp = bandpass(data2,'bandpass',[9, 16]);
            plot(bp(start_:end_),'r-','linewidth',1);
            set(gca,'fontsize',18);xlabel([]); ylabel('Ampl (uV)','fontsize',20);title('S1','fontsize',24);axis tight;ylim([-500 500]);
            linkaxes([x1, x2, x3, x4], 'x')
        else
            figure;
            sig = bandpass(S1_lfp_data(:,1),'bandpass',[2, 50]);
            x1 = subplot(311);plot(sig(start_:end_, 1),'r-','linewidth',1);
            set(gca,'fontsize',18);xlabel([]); ylabel('Ampl (uV)','fontsize',20);title('S1','fontsize',24);axis tight;ylim([-2500 2500]);

            x2 = subplot(312);bp = bandpass(sig,'bandpass',[9, 16]);
            plot(bp(start_:end_),'r-','linewidth',1);
            set(gca,'fontsize',18);xlabel([]); ylabel('Ampl (uV)','fontsize',20);title('S1','fontsize',24);axis tight;ylim([-500 500]);

            x3 = subplot(313);
            [S,t,f]=mtspecgramc(sig, movingwin, params);colormap('jet');     
            plot_matrix(S,t,f);

            linkaxes([x1, x2, x3], 'x')   
        end
    end
    %%
    %EEG = EEG * bmd.voltsperunit*1e6;  % mircovolt
    %EEG = data2(start_:end_);
    data = [data1, data2];
    append = {'_PFC', '_S1'};
    for d = 1:2
    
        EEG = data(:, d);
        session2 = [session, append{d}];
        %EEG = data2;

        %fs = bmd.Par.lfpSampleRate;

        % notch filter  remove power-line interfence 60Hz
        wo = 60/(fs/2);  bw = wo/35;
        [b,a] = iirnotch(wo,bw);
        EEG = filtfilt(b,a,EEG);
        % [b,a]=butter(5,50/(200/2),'low');
        % EEG=filtfilt(b,a,EEG);
        target_fs = 200;
        if fs~=target_fs
            % this if statement ensures that the sampling rate is 200
            EEG = resample(EEG, target_fs, fs);
%             fs = target_fs;
        end
        %
        secA = [];
        secB = [];
%         switch session
%             case 'Dino_061814_mPFC'
%                 secA = [2000, 5000];
%                 secB = [14000, 17000];
%             case 'Dino_062014_mPFC'
%                 secA = [4000, 9000];
%                 secB = [19000, 22000];
%             case 'Dino_061914_mPFC'
%                 secA = [10000 13000];
%                 secB = [];
%             case 'BWRat21_121813'
%                 secA = [3400 6600];
%                 secB = [10000 22000];
%             case 'Dino_072314_mPFC'
%                 secA = [1600 6300];
%                 secB = [];
%             case 'Dino_072414_mPFC' % This one rn
%                 secA = [3300 4800];
%                 secB = [];
%             case 'Bogey_012615'
%                 secA = [4300 7870];
%                 secB = [9700 23300];
%             case 'BWRat20_101013'
%                 secA = [];
%                 secB = [];
%         end
        EEG = EEG';
        EEG = bandpass(EEG,'bandpass', [2, 50]);

%         if ~isempty(secB)
%             EEG=[EEG(secA(1)*fs:secA(2)*fs); EEG(secB(1)*fs:secB(2)*fs)];  %ignore the first 4000 secs in EEG recording2
%         elseif ~isempty(secA)
%             EEG=EEG(secA(1)*fs:secA(2)*fs);
%         end
        output = zeros(size(EEG));

        % get the labels
        C=cell(1,2);
        %C{1}=SpindleData.normspindles(:,1);
        %C{2}=SpindleData.normspindles(:,3);

%         if ~isempty(secB)
%             index = find(C{1}>secB(1), 1 );
%             C{1}(1:index-1)=C{1}(1:index-1)-secA(1);
%             C{2}(1:index-1)=C{2}(1:index-1)-secA(1);
%             C{1}(index:end)=C{1}(index:end)-secB(1)+secA(2)-secA(1);
%             C{2}(index:end)=C{2}(index:end)-secB(1)+secA(2)-secA(1);
%         elseif ~isempty(secA)
%             C{1}=C{1}-secA(1);
%             C{2}=C{2}-secA(1);
%         end

        % for k=1:length(C{1})
        %     range = [floor(C{1}(k)*fs):floor(C{2}(k)*fs)];
        %     output(range)=1;
        % end
        
         % Generate files to store EEG data
%         save(['data/pain_sleep/' session2, '.mat'], 'EEG');
%% 
%          h1 = fopen(['RatData/test_data/pain_sleep/' session2 '_noPCA_bp2_50.txt'], 'wt');  
%          h2 = fopen(['RatData/test_data/pain_sleep/' session2 '_noPCA_bp2_50_labels.txt'], 'wt');  
         h1 = fopen(['RatData/test_data/pain_sleep/AS-2-06_NightA_ch8_bp.txt'], 'wt');  
         h2 = fopen(['RatData/test_data/pain_sleep/AS-2-06_NightA_ch8_bp_labels.txt'], 'wt');  
% 
         fprintf(h1, '%f\n', EEG);
         fprintf(h2, '%d\n', output);
% 
         fclose all;
    end
end
