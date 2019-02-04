clear all;
% session = {'excerpt2','excerpt4','excerpt5','excerpt6'};
session = {'excerpt7'};
h1=fopen(['SleepSpindleData4RNN/Augment_test.txt'],'wt');
h2=fopen(['SleepSpindleData4RNN/Augment_test_labels.txt'],'wt');
num_steps=50;
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
    sample_length=1800*sample_rate;  %30 mins=1800 secs. 30 minutes is length of one .edf file
    f=fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    C=textscan(f,'%f %f');
    
    % I don't care for spindles in the beginning and end of the file!!!!
    if C{1}(end) > 1800-2.25
       C{1} = C{1}(1:end-1);
       C{2} = C{2}(1:end-1);
    end
    if C{1}(1) < 2.25
       C{1} = C{1}(2:end);
       C{2} = C{2}(2:end);
    end
    output=zeros(sample_length,1);
    for i=1:length(C{1})
        output(floor(sample_rate*C{1}(i)):floor(sample_rate*(C{1}(i)+C{2}(i))))=1;
    end
    % center = zeros(length(C{1}),1);
    sequence_length = sample_rate * 2.0; %2.5;  % 2.5 seconds
    % new_squence = zeros(length(C{1}),squence_length);
    data=[];
    label=[];
    for i=1:length(C{1})
        center = floor(sample_rate*(C{1}(i) + C{2}(i)/2 ));
%         new_squence = center-squence_length/2+1:center+squence_length/2;
%         new_squence_l = center-squence_length/2+1-squence_length/5:center+squence_length/2-squence_length/5;
%         new_squence_r = center-squence_length/2+1+squence_length/5:center+squence_length/2+squence_length/5;
%         new_squence_2l = center-squence_length/2+1-2*squence_length/5:center+squence_length/2-2*squence_length/5;
%         new_squence_2r = center-squence_length/2+1+2*squence_length/5:center+squence_length/2+2*squence_length/5;
%         fprintf(h1,'%d\n',EEG(new_squence));
%         fprintf(h2,'%d\n',output(new_squence));
%         fprintf(h1,'%d\n',EEG(new_squence_l));
%         fprintf(h2,'%d\n',output(new_squence_l));
%         fprintf(h1,'%d\n',EEG(new_squence_r));
%         fprintf(h2,'%d\n',output(new_squence_r));
%         fprintf(h1,'%d\n',EEG(new_squence_2l));
%         fprintf(h2,'%d\n',output(new_squence_2l));
%         fprintf(h1,'%d\n',EEG(new_squence_2r));
%         fprintf(h2,'%d\n',output(new_squence_2r));
        for j=center-sample_rate*0.2:center+sample_rate*0.2
            start_ = j - sample_rate + 1;
            end_ = j + sample_rate;
            new_sequence = start_:end_;
            %new_squence = start+j-1:start+sequence_length+j-1;
            data = [data, reshape(EEG(new_sequence),[num_steps, length(new_sequence)/num_steps])];
            label = [label, reshape(output(new_sequence),[num_steps, length(new_sequence)/num_steps])];
        end 
        add_spindle = center-sample_rate*0.5+1:center+sample_rate*0.5;
        for k=1:80
            data = [data, reshape(EEG(add_spindle),[num_steps, length(add_spindle)/num_steps])];
            label = [label, reshape(output(add_spindle),[num_steps, length(add_spindle)/num_steps])];
        end  
    end
    fprintf(h1,'%d\n',reshape(data,[size(data,2)*num_steps,1]));
    fprintf(h2,'%d\n',reshape(label,[size(label,2)*num_steps,1]));
end
fclose(h1);
fclose(h2);