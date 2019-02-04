clearvars;
addpath(genpath('chronux_2_11'));
num_subject = {1,2,3,4,5,6,7,8};
session = {};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end
num_step=50;
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
    target_f=200;
    if sample_rate~=target_f
        EEG=interp(EEG,target_f/sample_rate);
        sample_rate=target_f;
    end
    f=fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    C=textscan(f,'%f %f');
    fclose(f);
    fs=sample_rate;
    
    database = cell(1,1);
    database{1}=EEG;
    
    std_value = zeros(size(C{1,1}, 1),1);
    for i=1:size(C{1,1}, 1)
        s_data=database{1}(floor(C{1,1}(i)*fs):floor((C{1,1}(i)+C{1,2}(i))*fs));
        s_data = detrend(s_data);
        std_value(i) = std(s_data);
    end
    pd = fitdist(std_value,'Normal');
    disp(['======================================================']);
    disp(['========Working on subject - ' num2str(num_subject{idx})  '==========']);
    disp(mean(pd));
     %%   compute power

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
    database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
    save(['./data/DREAMsub' num2str(num_subject{idx}) 'Power']);
    %%
    labels = zeros(size(EEG));
    for j=1:size(C{1,1},1)
        labels(floor((C{1,1}(j))*fs) : floor((C{1,1}(j)+C{1,2}(j))*fs)) = 1;   % reduce noise 
    end

    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    format2 = '%d\n';
    h1=fopen(['data/test_data/test_' session{idx} '.txt'] , 'wt');
    h2=fopen(['data/test_data/test_' session{idx} '_labels.txt'] , 'wt');
    fprintf(h1,format1,database{1}');
    fprintf(h2,format2, labels);
    fclose(h1);
    fclose(h2);
end