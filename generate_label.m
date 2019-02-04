session = 'excerpt7';
data_path = ['DatabaseSpindles/' session '.edf'];
[hdr,record]=edfread(data_path);
IndexC=strfind(hdr.label,'CZA1');
Index = find(not(cellfun('isempty', IndexC)));
if isempty(Index)
    IndexC=strfind(hdr.label,'C3A1');
    Index = find(not(cellfun('isempty', IndexC)));
end
EEG=record(Index,:)';
sample_rate=hdr.frequency(Index);
sample_length=1800*sample_rate;
f=fopen(['SleepSpindleData4RNN/' session '_label.txt'],'r');
C=textscan(f,'%f %f');
output=zeros(sample_length,1);
for i=1:length(C{1})
    output(floor(sample_rate*C{1}(i)):floor(sample_rate*(C{1}(i)+C{2}(i))))=1;
end
fclose(f);
h=fopen(['SleepSpindleData4RNN/' session '_labels.txt'],'wt');
fprintf(h,'%d\n',output);
fclose(h);