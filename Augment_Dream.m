session = {'excerpt1', 'excerpt2','excerpt3','excerpt4','excerpt5','excerpt6','excerpt7','excerpt8'};
Spindles=cell(1,2);
database = cell(1,1);
output = database;
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
    if sample_rate~=200
        EEG=interp(EEG,200/sample_rate);
        sample_rate=200;
    end
    f=fopen(['SleepSpindleData4RNN/' session{idx} '_label.txt'],'r');
    total_time = 30*60;
    C=textscan(f,'%f %f');
    fclose(f);
    fs=sample_rate;
    
    labels = zeros(size(EEG));
    for j=1:size(C{1,1},1)
        labels(floor((C{1,1}(j))*fs) : floor((C{1,1}(j)+C{1,2}(j))*fs)) = 1;   % reduce noise 
%         figure;
%         plot(EEG(C{1,1}(j)*fs : floor((C{1,1}(j)+C{1,2}(j))*fs)));
    end
    
    database = {[database{1}; EEG]};
    output = {[output{1}; labels]};
    
    C{1,1} = C{1,1} + total_time*(idx-1);
    % I don't care for spindles in the beginning and end of the file!!!!    
    if C{1,1} < total_time*(idx-1)+2.25
       C{1,1} = C{1,1}(2:end);
    end
    if C{1,1}(end) > total_time*idx-2.25
       C{1,1} = C{1,1}(1:end-1);
    end    
    Spindles={[Spindles{1};C{1,1}],[Spindles{2};C{1,2}]};
end

%% get the variance of spindles
% std_value = zeros(size(Spindles{1}, 1),1);
% for i=1:size(Spindles{1}, 1)
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
%     std_value(i) = std(s_data);
% end
% pd = fitdist(std_value,'Normal');
% disp(mean(pd));
% for i=31:60
%     figure;
%     s_data=database{1}((Spindles{1}(i)-0.5)*fs:floor((Spindles{1}(i)+Spindles{2}(i)+0.5)*fs));
% %     s_data = detrend(s_data);
%     plot([1:length(s_data)], s_data);
%     hold on 
%     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
% %     s_data = detrend(s_data);
%     plot([101:length(s_data)+100],s_data,'r');
% end
%% set subset fraction
subset = 1;

%%  get baseline
num_steps = 50;   % sequency length of one example
baseline=database{1};
baseline(output{1}==1)=[];
baseline =  baseline(1:floor(length(baseline)/num_steps)*num_steps);
num_base = length(baseline)/num_steps;
baseline = reshape(baseline, [num_steps, num_base]);
train_b = floor(num_base*0.7*subset);  % take 70% of the baseline as traning data
valid_b = floor(num_base*0.2*subset);  % take 20% of the baseline as validation data
test_b = floor(num_base*0.1*subset); % num_base-train_b-valid_b;
ind_b = randperm(num_base);
train_base = reshape(baseline(:,ind_b(1:train_b)), [num_steps*train_b , 1]);
valid_base = reshape(baseline(:,ind_b(train_b+1:train_b+valid_b)), [num_steps*valid_b , 1]);
test_base = reshape(baseline(:,ind_b(train_b+valid_b+1:train_b+valid_b+test_b)), [num_steps*test_b , 1]);

%% Augment Spindles
S = [Spindles{1}, Spindles{2}];
num_spindles = size(S,1);
train_s = floor(num_spindles*0.7*subset);
valid_s = floor(num_spindles*0.2*subset);
test_s = floor(num_spindles*0.1*subset);%num_spindles-train_s-valid_s;
ind_s = randperm(num_spindles);
train_spindles = S(ind_s(1:train_s) , :);
valid_spindles =  S(ind_s(train_s+1:train_s+valid_s), :);
test_spindles = S(ind_s(train_s+valid_s+1:train_s+valid_s+test_s), :);

h1=fopen(['SleepSpindleData4RNN/Augment_train_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base.txt'],'wt');
h2=fopen(['SleepSpindleData4RNN/Augment_train_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base_labels.txt'],'wt');
h3=fopen(['SleepSpindleData4RNN/Augment_valid_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base.txt'],'wt');
h4=fopen(['SleepSpindleData4RNN/Augment_valid_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base_labels.txt'],'wt');
h5=fopen(['SleepSpindleData4RNN/Augment_test_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base.txt'],'wt');
h6=fopen(['SleepSpindleData4RNN/Augment_test_Dream_nb_' num2str(num_steps) '_' num2str(subset*100) '%_base_labels.txt'],'wt');

for m=1:size(train_spindles, 1)
     data=cell(1,1);
     label=data;
     center = floor(fs*(train_spindles(m, 1) + train_spindles(m, 2)/2 ));
     for j=center-fs*0.2:center+fs*0.2
            start_ = j - fs + 1;
            end_ = j + fs;
            new_sequence = start_:end_;
            %new_squence = start+j-1:start+sequence_length+j-1;
            data = {[data{1}, reshape(database{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
            label = {[label{1}, reshape(output{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
     end 
     add_spindle = center-fs*0.5+1:center+fs*0.5;
     for k=1:80
           data = {[data{1}, reshape(database{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
           label = {[label{1}, reshape(output{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
     end    
     data{1}=detrend(data{1});
     fprintf(h1,'%f\n',reshape(data{1},[size(data{1},2)*num_steps,1]));
     fprintf(h2,'%d\n',reshape(label{1},[size(label{1},2)*num_steps,1]));
end


for m=1:size(valid_spindles, 1)
     data=cell(1,1);
     label=data;
     center = floor(fs*(valid_spindles(m, 1) + valid_spindles(m, 2)/2 ));
     for j=center-fs*0.2:center+fs*0.2
            start_ = j - fs + 1;
            end_ = j + fs;
            new_sequence = start_:end_;
            %new_squence = start+j-1:start+sequence_length+j-1;
            data = {[data{1}, reshape(database{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
            label = {[label{1}, reshape(output{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
     end 
     add_spindle = center-fs*0.5+1:center+fs*0.5;
     for k=1:80
           data = {[data{1}, reshape(database{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
           label = {[label{1}, reshape(output{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
     end    
     data{1}=detrend(data{1});
     fprintf(h3,'%f\n',reshape(data{1},[size(data{1},2)*num_steps,1]));
     fprintf(h4,'%d\n',reshape(label{1},[size(label{1},2)*num_steps,1]));
end

% P = 0; N=0;
for m=1:size(test_spindles, 1)
     data=cell(1,1);
     label=data;
     center = floor(fs*(test_spindles(m, 1) + test_spindles(m, 2)/2 ));
     for j=center-fs*0.2:center+fs*0.2
            start_ = j - fs + 1;
            end_ = j + fs;
            new_sequence = start_:end_;
            %new_squence = start+j-1:start+sequence_length+j-1;
            data = {[data{1}, reshape(database{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
            label = {[label{1}, reshape(output{1}(new_sequence),[num_steps, length(new_sequence)/num_steps])]};
     end 
     add_spindle = center-fs*0.5+1:center+fs*0.5;
     for k=1:80
           data = {[data{1}, reshape(database{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
           label = {[label{1}, reshape(output{1}(add_spindle),[num_steps, length(add_spindle)/num_steps])]};
     end     
     data{1}=detrend(data{1});
%      tmp=sum(label{1}, 1);
%      P=P+length(find(tmp>50/2));
%      N=N+length(tmp)-length(find(tmp>50/2));
     fprintf(h5,'%f\n',reshape(data{1},[size(data{1},2)*num_steps,1]));
     fprintf(h6,'%d\n',reshape(label{1},[size(label{1},2)*num_steps,1]));
end

%% add baseline to file
fprintf(h1,'%f\n',train_base);
fprintf(h2,'%d\n',zeros(size(train_base)));
fprintf(h3,'%f\n',valid_base);
fprintf(h4,'%d\n',zeros(size(valid_base)));
fprintf(h5,'%f\n',test_base);
fprintf(h6,'%d\n',zeros(size(test_base)));

fclose(h1);
fclose(h2);
fclose(h3);
fclose(h4);
fclose(h5);
fclose(h6);