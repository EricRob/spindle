clear;
session = {'excerpt1'};
num_step=50;
for idx=1:length(session)
    h1=fopen(['SleepSpindleData4RNN/test_' session{idx} '.txt'] , 'wt');
    h2=fopen(['SleepSpindleData4RNN/test_' session{idx} '_labels.txt'] , 'wt');
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
    f=fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    C=textscan(f,'%f %f');
    fclose(f);
    fs=sample_rate;
    
    labels = zeros(size(EEG));
    for j=1:size(C{1,1},1)
        labels(floor((C{1,1}(j))*fs) : floor((C{1,1}(j)+C{1,2}(j))*fs)) = 1;   % reduce noise 
    end
    test_data = zeros(num_step, length(EEG)-num_step+1);
    test_label =test_data;
    for i=1:length(EEG)-num_step+1
        test_data(:,i)=EEG(i:i+num_step-1);
        test_label(:,i)=labels(i:i+num_step-1);
    end

    test_data=detrend(test_data);
    test_data = reshape(test_data, [num_step*(length(EEG)-num_step+1), 1]);
    test_label = reshape(test_label, [num_step*(length(EEG)-num_step+1), 1]);
    fprintf(h1,'%f\n', test_data);
    fprintf(h2,'%f\n', test_label);
    fclose(h1);
    fclose(h2);
end