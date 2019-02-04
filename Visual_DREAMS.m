clearvars;
addpath(genpath('chronux_2_11'));
num_subject = {3}; %,2,3,4,5,6,7,8};
session = {};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end
num_step=50;
for idx=1:length(session)
    
    data_path = ['DatabaseSpindles/' session{idx} '.edf'];
    [hdr, record] = edfread(data_path);
    IndexC = strfind(hdr.label,'CZA1');
    Index = find(not(cellfun('isempty', IndexC)));
    if isempty(Index)
        IndexC=strfind(hdr.label,'C3A1');
        Index = find(not(cellfun('isempty', IndexC)));
    end
    EEG = record(Index,:)';
    sample_rate = hdr.frequency(Index);
    % Always resample to 200Hz for now
    target_f = 200;
    if sample_rate ~= target_f
        EEG = interp(EEG,target_f/sample_rate);
        sample_rate=target_f;
    end
    
    f = fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    C = textscan(f,'%f %f');
    fclose(f);
    fs = sample_rate;
    
    database = cell(1,1);
    database{1} = EEG;
    stages = ones(size(EEG))+1;
    Annotation_or = [C{1,1},C{1,2}];       
%% draw

    start = Annotation_or(:,1);
    duration=Annotation_or(:,2);
    t_s=[];
    t_e=[];
    start_time = 0;  % seconds
    end_time = start_time+30*60;
    datarange=(start_time+1/fs:1/fs:end_time);
    ind=find(start>datarange(1) & start<datarange(end));
    if ~isempty(ind)
        for i=1:length(ind)
            t_s(i)=start(ind(i));
            t_e(i)=t_s(i)+duration(ind(i));
        end
    end
    
    figure; 
    subplot(211);
    plot(datarange, EEG(floor(datarange*fs)));
    set(gca,'ylim',[-100,100]);
    hold on;
    if ~isempty(ind)
        for ii=1:length(t_s)
            plot([t_s(ii),t_s(ii)],[-100 100],['r','-'],'linewidth',1);
            plot([t_e(ii),t_e(ii)],[-100 100],['r','-'],'linewidth',1);
        end
    end  
    xlim([min(datarange), max(datarange)]);
    xlabel('Time (s)');
    ylabel('Amptitude (uV)');
    set(gca,'fontsize',16);
    
    subplot(212)
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 25];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');     
    plot_matrix(S,t,f);
    % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
    hold on; 
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
        plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
    end
    plot([1,29],[11 11],['b','-'],'linewidth',1);
    plot([1,29],[16 16],['b','-'],'linewidth',1);
    
    
end