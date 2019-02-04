clearvars;
close all;
addpath(genpath('chronux_2_11'));
database = cell(1,1);
channel = 6;
target_f = 200;
%% Load NN results
f2=fopen('results/probability(alpha).txt'); 
prob=fscanf(f2,'%f');
fclose(f2);
pred = uint8(prob>0.5);
 %%
 lens = 0;
for i=1:2
    session = ['S001R0' num2str(i)];
    [hdr,record]=ReadEDF2(['BCI2000\' session '.edf']);
    fs = hdr.samplerate(channel);
    EEG = record{channel};
    if fs~=target_f
       EEG=resample(EEG, target_f, floor(fs));
       fs=fs*target_f/floor(fs); 
    end
    lens = lens+length(EEG);
    database = {[database{1}; EEG]};
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 25];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    [S,t,f]=mtspecgramc(EEG,movingwin,params);
    switch i
        case 1
            outline = 'Baseline, eyes open';
        case 2
            outline = 'Baseline, eyes clsoed';
        case 3
            outline = 'Task 1';
        case 4
            outline = 'Task 2';
        case 5
            outline = 'Task 3';
        case 6
            outline = 'Task 4';
    end
    figure(i)
    datarange=[1/fs:1/fs:length(EEG)/fs];
    subplot(411);
    plot(datarange, EEG);
    title(outline);
    xlim([min(datarange),max(datarange)]);
    subplot(412);
    plot_matrix(S,t,f);colormap('jet'); 
    subplot(413);
    plot(datarange, prob(lens-length(EEG)+1:lens));
    xlim([min(datarange), max(datarange)]);
    ylim([0,1]);
    subplot(414)
    plot(datarange, pred(lens-length(EEG)+1:lens));
    xlim([min(datarange), max(datarange)]);
    ylim([-1,2]);
end
output=zeros(size(database{1}));
%%
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
    seq = (seq-mean(seq))/ std(database{1});     %13.8275;
    [SS,tt,ff]=mtspecgramc(seq,movingwin,params);
    energyDB = 10*log10(SS);
    %     DB_LF = mean(energyDB(:,3:8),2);
    DB_BB = mean(energyDB(:,4:11),2);
    DB_spindle = mean(energyDB(:,12:22),2);
    power_BB(k+floor(fs)/2)=DB_BB;
    power_spindle(k+floor(fs)/2)=DB_spindle;
    ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
    bdpass = bandpass(seq, 'bandpass');
    band_pass(k+floor(fs)/2)=bdpass(floor(fs)/2+1);
    [up,~]=envelope(bdpass);
    up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
    power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
end
database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
%%
 h1=fopen(['SleepSpindleData4RNN/test_alpha.txt'] , 'wt');
 h2=fopen(['SleepSpindleData4RNN/test_alpha_labels.txt'] , 'wt');
 format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
 format2 = '%d\n';
 fprintf(h1,format1,database{1}');
 fprintf(h2,format2, output);
 fclose(h1);
 fclose(h2);