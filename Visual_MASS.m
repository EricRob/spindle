clearvars;
session={'01-02-0001'};
for idx=1:length(session)
    data_path = ['SleepSpindleData4RNN/MASS_C1_SS2/version2015/' session{idx} ' PSG.edf'];
    [hdr,record]=ReadEDF2(data_path);
    IndexC=strfind(hdr.labels,'EEG');
    Index = find(not(cellfun('isempty', IndexC)));
    label_path = ['SleepSpindleData4RNN/MASS_C1_SS2/annotations/MASS-C1-SS2-SpindleE1-EDF/' session{idx} ' SpindleE1.edf'];
    [hdr1,record1]=ReadEDF2(label_path);
    start = hdr1.annotation.starttime;
    duration = hdr1.annotation.duration;
    fs=hdr.frequency(Index(1));
    start_time = 880;  % seconds
    end_time = 910;
    datarange=(start_time+1/fs:1/fs:end_time);
    ind=find(start>datarange(1) & start<datarange(end));
    if ~isempty(ind)
        for i=1:length(ind)
            t_s(i)=start(ind(i));
            t_e(i)=t_s(i)+duration(ind(i));
        end
    end
    for i=1:length(Index)/4
        figure;
        for j=1:4
            subplot(4,1,j);
            plot(datarange,record{Index(4*(i-1)+j)}(floor(datarange*fs)));
            hold on;
            title(['Channel : ' hdr.labels{Index(4*(i-1)+j)} ' ']);
            if ~isempty(ind)
                for ii=1:length(t_s)
                    plot([t_s(ii),t_s(ii)],[-100 100],['r','-'],'linewidth',1);
                    plot([t_e(ii),t_e(ii)],[-100 100],['r','-'],'linewidth',1);
                end
            end
        end
    end
end
