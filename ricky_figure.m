clearvars;
addpath(genpath('chronux_2_11'));
figure;
freq = [200, 100, 50, 34];
loc = [1,2,5,6,3,4,7,8];
for index=1:length(freq)
    if loc(index) == 2
        l = 1;
    end
    load(['./data/MASSsub1Power' num2str(freq(index)) 'Hz.mat']);
    if floor(fs) == 200
        f2=fopen(['results/Cross_valid/probability(sub1).txt']);
    else
        f2=fopen(['results/MASS_' num2str(floor(fs)) 'Hz/probability_sub01_'  num2str(floor(fs)) 'Hz.txt']);
    end
    prob=fscanf(f2,'%f');
    fclose(f2);
    
    start = Annotation_or(:,1);
    duration=Annotation_or(:,2);
    t_s=[];
    t_e=[];
    start_time = 4712;  % seconds
    window = 10;
    end_time = start_time+window;
    datarange=(start_time+1/fs:1/fs:end_time);
    ind=find(start>datarange(1) & start<datarange(end));
    if ~isempty(ind)
        for i=1:length(ind)
            t_s(i)=start(ind(i));
            t_e(i)=t_s(i)+duration(ind(i));
        end
    end
    subplot(4,2, loc(index));
    %subplot(211);
    plot(datarange,EEG(floor(datarange*fs)),'Linewidth',1);
    scale = 35;
    hold on;
    if ~isempty(ind)
        for ii=1:length(t_s)
            plot([t_s(ii),t_s(ii)],[-scale scale],['r','-'],'linewidth',3);
            plot([t_e(ii),t_e(ii)],[-scale scale],['r','-'],'linewidth',3);
            highlight = (datarange >= t_s(ii)) & (datarange <= t_e(ii));
            A = datarange(highlight);
            plot(A,EEG(floor(A*fs)),'Linewidth',1,'Color',[.6 0 0]);
        end
    end
    
    yyaxis left;
    
    set(gca,'ylim',[-scale, scale]);
    ylabel('uV');
    
    title([num2str(floor(fs)) 'Hz'],'FontSize',18);
    xlim([(start_time+0.5294) (end_time - 0.4076)]);
    
    yyaxis right;
    prob_th = 0.6;
    if floor(fs) > 50
        prob_th  = 0.8;
    end
    threshed = prob >= prob_th;
    plot(datarange, threshed(floor(datarange*fs)),'Linewidth',3, 'Color',[0 1 0], 'LineStyle', '-');
    plot(datarange, prob(floor(datarange*fs)),'Linewidth',2, 'Color',[0 0 0], 'LineStyle', '-');
    ylabel('Detection Probability');
    subplot(4,2, loc(index + 4));
    %subplot(212)
    yyaxis left
    params.fpass = [0 17];
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');
    plot_matrix(S,t,f);
    % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
    hold on
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
        plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
    end
    plot([0.5,window+.5],[11 11],['w','-'],'linewidth',2);
    plot([0.5,window+.5],[16 16],['w','-'],'linewidth',2);
    xticks(1:(window-1));
    xticklabels(xticks() + start_time);
    set(gca,'fontsize',10);
    ylabel('Hz');
    xlabel('Time (s)');
    colorbar('off')
    yyaxis right
    ylabel('Detection Probability');
    plot(datarange - datarange(1), threshed(floor(datarange*fs)),'Linewidth',3, 'Color',[0 1 0]);
    plot(datarange - datarange(1), prob(floor(datarange*fs)),'Linewidth',2, 'Color',[0 0 0], 'LineStyle', '-');
    
    grid on;
end
%%
%load('/media/share/jwanglab/jwanglabspace/David Rosenberg/Earth10/7_11_2018/LFP_1/Earth10_5-22-2018_250_WithLFP.mat');
f1 = fopen(['RatData/NN_output/probability_Earth10_071118.txt']);
net_prob = fscanf(f1, '%f');
fclose(f1);

f2 = fopen(['RatData/test_data/Earth10_071118_bp2_50.txt']);
pca_out = fscanf(f2, '%f');
fclose(f2);

%data = allUnits.LFP(34).LFP;
%%

fs = 200;


start_time = 9500;

end_time = start_time + 1000;

probThresh = 0.3; % P
timeThresh = 0.06; % seconds

wangDet = makeBinary(net_prob, probThresh, timeThresh*fs); % wangDet

figure;
subplot(3,1,1);
plot(start_time:end_time, pca_out(start_time:end_time));

subplot(3,1,2);
plot(start_time:end_time, net_prob(start_time:end_time));
hold on;
plot(start_time:end_time, wangDet(start_time:end_time));

subplot(3,1,3);
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 25];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    [SS,tt,ff]=mtspecgramc(pca_out(start_time:end_time),movingwin,params);
    plot_matrix(SS);colormap('jet');     
