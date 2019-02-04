addpath(genpath('chronux_2_11'));
addpath(genpath('Sleep_spindles_FHN2015'));
session = {'excerpt1'};
f1=fopen('predictions1.txt');
f2=fopen('probability1.txt');
predict=fscanf(f1,'%d');
prob=fscanf(f2,'%f');
fclose(f1);
fclose(f2);
ind = find(diff(predict)==1);
ind = ind+1;
thresh = 25;
smooth = predict;
for i=1:length(ind)-thresh
  if sum(predict(ind(i):(ind(i)+thresh-1))) < thresh
      smooth(ind(i):(ind(i)+thresh-1)) = 0;
  end
end
Spindles=cell(1,2);
data_path = ['DatabaseSpindles/' session{1} '.edf'];
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
%     EEG = -1 + 2*(EEG-min(EEG))/(max(EEG)-min(EEG)); 
f=fopen(['SleepSpindleData4RNN/' session{1} '_label.txt'],'r');
total_time = 30*60;
C=textscan(f,'%f %f');
fclose(f);
fs=sample_rate;
out=bandpass(EEG);

%% evaluate latency
smooth = smooth(1:179950);
detect = find(diff(smooth)==1)+1;
offset = find(diff(smooth)==-1)+1;
for t=1:size(offset)
    ind = find(detect>offset(t)&detect<=offset(t)+20, 1);
    if ~isempty(ind)
        smooth(offset(t):detect(ind(end)))=1;
    end
end
detect = find(diff(smooth)==1)+1;
num_spindles = 48;%size(C{1},1);
t_event = 0;
result = cell(1,3);
ii=[];
for k=1:num_spindles
    index = find((detect+49)>(C{1}(k)-0.3)*fs & (detect+49)<(C{1}(k)+0.5)*fs);
    if ~isempty(index)
        index = index(end);
        onset = (detect(index)+49)/fs;
        t_event=t_event+1;
        result{1} = [result{1};C{1}(k)];
        result{2} = [result{2};onset];
        result{3} = [result{3};onset-C{1}(k)];
        ii = [ii;index];
    else
        result{1} = [result{1};C{1}(k)];
        result{2} = [result{2};999];
        result{3} = [result{3};999];
    end
    
end
latency = result{3};
latency(latency==999)=[];
disp(['Mean: ' num2str(mean(latency)) '   Std: ' num2str(std(latency))]);
f_event = detect;
f_event(ii)=[];
disp(['Fasle event rate: ' num2str(length(f_event)/30) '  events/min']);
disp(['True Postive event: ' num2str(t_event) '   Total spindles:  '  num2str(num_spindles)]);
%% draw figure
datarange=floor([360:1/fs:390]*fs);   %104001:110000  110001:116000
params.Fs=200;
params.fpass=[0 50];
params.tapers=[3 5];
params.pad=0;

ind=find(C{1}>datarange(1)/fs & C{1}<datarange(end)/fs);
if ~isempty(ind)
    for i=1:length(ind)
        t_s(i)=C{1}(ind(i));
        t_e(i)=t_s(i)+C{2}(ind(i));
    end
%%
    figure;
    subplot(311);
    plot(datarange/fs,EEG(floor(datarange)));
    hold on;
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)],[-100 100],['r','-'],'linewidth',1);
        plot([t_e(i),t_e(i)],[-100 100],['r','-'],'linewidth',1);
    end
    xlim([datarange(1),datarange(end)]/fs);
    ylim([-100,100]);
    title('Original EEG segment');
    ylabel('Amplitude(uV)');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    subplot(312);
    plot(datarange/fs,smooth(floor(datarange)-49));
    ylim([-1,2]);
    ylabel('Predictions');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    subplot(313);
    plot(datarange/fs,prob(floor(datarange)-49));   %prob
    ylim([-1,2]);
    ylabel('Probabilities');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    envelope(out(datarange),1e2,'analytic');
    hold on;
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]*sample_rate-datarange(1),[-30 30],['r','-'],'linewidth',1);
        plot([t_e(i),t_e(i)]*sample_rate-datarange(1),[-30 30],['r','-'],'linewidth',1);
    end
    xlim([0,length(datarange)]);
    title('Band-passed EEG segment zero phase');
    ylabel('Amplitude (uV)');
    xlabel('Samples');
    set(gca,'fontsize',16);
    ylim([-30,30]);
%     figure;
%     movingwin=[1 .01];
%     params.tapers=[3 5];
%     params.err=0;
%     % [S,t,f]=mtspecgramc(EEG(datarange),movingwin,params);
%     [S,t,f]=mtspecgramc(EEG(datarange),movingwin,params);
%     plot_matrix(S,t,f);
%     colormap('jet')
%     % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
%     hold on 
%     for i=1:length(t_s)
%         plot([t_s(i),t_s(i)]-datarange(1)/sample_rate,[0 50],['r','-'],'linewidth',1);
%         plot([t_e(i),t_e(i)]-datarange(1)/sample_rate,[0 50],['r','-'],'linewidth',1);
%     end
%     plot([1,29],[11 11],['b','-'],'linewidth',1);
%     plot([1,29],[16 16],['b','-'],'linewidth',1);
%     grid on;
    % figure;
    % spectrogram(EEG(datarange),kaiser(1000,10),255,512,100,'yaxis');
%     figure(3);
%     [S,f]=mtspectrumc(EEG(datarange),params);
%     plot_vector(S,f,[],[],'r',0.5);
    % periodogram(EEG(datarange),rectwin(length(EEG(datarange))),length(EEG(datarange)),fs);
end