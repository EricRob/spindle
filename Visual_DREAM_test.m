clearvars;
addpath(genpath('chronux_2_11'));
num_subject = {1,2,3,4,5,6,7,8};
session1 = {};
for i=1:length(num_subject)
    session1={session1{:},['excerpt' num2str(num_subject{i})]};
end
num_step=50;
%%
for idx1 = 1:length(session1)
    % evaluate latency
%     movingwin=[1 1/fs];
%     params.Fs=fs;
%     params.fpass=[0 25];
%     params.tapers=[3 5];
%     params.err=0;
%     params.pad=0;
%     [SS,tt,ff]=mtspecgramc(EEG,movingwin,params);
%     energyDB = 10*log10(SS);
%     DB_LF = mean(energyDB(:,3:8),2);
%     DB_BB = mean(energyDB(:,4:11),2);
%     DB_spindle = mean(energyDB(:,12:21),2);
%     save Testsub1
    load(['./data/DREAMsub', session1{idx1}(8), 'Power.mat']);
    
    % Load NN results
    %f1=fopen('results/predictions(sub18-or).txt');
    %f2=fopen('results/DREAMS/probability(MASS1_8).txt'); 
    %f2=fopen('results/DREAMS/probability_synth1-4_7-8_6_1e-5.txt'); 
    %prob=fscanf(f2,'%f');
    %fclose(f1);
    %fclose(f2);
    
    prob = load(['./results/mcsleep/', session1{idx1}, '.mat']);
    prob = prob.spindles;
    prob = prob';
    
    num_steps = 50;
    prob=prob(num_steps:end);
    %
    prob_th = 0.8; %0.85;
    time_th = 30;
    stages = ones(size(EEG))+1;
    Annotation_or = [C{1,1},C{1,2}];
    if 0
        
        range = [0.02:0.05:0.95, 0.96:0.01:1];
%         range= [0:5:100];
        sensitivity = zeros(length(range),1);
        specificity = zeros(length(range),1);
        fdr = zeros(length(range),1);
        f1score = zeros(length(range),1);
        for i=1:length(range)
%             time_th=range(i);
            prob_th=range(i);
            [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_steps, 0);
        end
%         figure; plot3(sensitivity, specificity, 1-fdr);xlabel('sensitivity');ylabel('specificity');zlabel('1-fdr');xlim([0,1]);ylim([0,1]);
%         [~, I]=min((1-sensitivity).^2+(1-specificity).^2+fdr.^2);
%         disp(range(I));
        area=0;
        for k=1:length(sensitivity)
            if k==1
                area=area+(sensitivity(k))*(specificity(k));
            else
                area=area+(sensitivity(k-1)+sensitivity(k))*(specificity(k)-specificity(k-1))/2;
            end
        end
        disp(['AUROC Area:  '  num2str(area)]);
        figure; plot(1- specificity,sensitivity,'.-');ylabel('sensitivity');xlabel('1-specificity');xlim([0,1]);ylim([0,1]);
        [~, I]=min((1-sensitivity).^2+(1-specificity).^2);
        disp(range(I));
    else
        [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_steps, 1);
    end
end
    %%
    duration=Annotation_or(:,2);
    figure;subplot(211);
    histogram(fp_result{3});
    hold on;histogram(result{5});
    legend('FP','GT');
%     temp1 = result{2};
%     temp2 = result{4};
%     temp2(temp1==999)=[];
%     temp1(temp1==999)=[];
%     histogram(DB_spindle(floor((temp1+temp2/2-tt(1))*fs)));
    xlabel('Power (DB)');
    ylabel('Number');
    subplot(212);histogram(fp_result{2});
    hold on;histogram(duration);
    xlabel('Duration (s)');
    ylabel('Number');
    legend('FP','GT');
    
%     subplot(313);histogram(fp_result{5});
%     hold on;histogram(result{8});
%     xlabel('Frequency (s)');
%     ylabel('Number');
%     figure;
%     scatter3(fp_result{5},fp_result{2}, fp_result{3}); 
%     hold on;
%     scatter3(result{8},duration,result{5});
%     grid on;
%     xlabel('Frequency (Hz)');
%     ylabel('Duration (s)');
%     zlabel('Power (dB)')
%     figure;[N, bin] = histc(fp_result{4},[-5:.05:5]);
%     bar([-5:.05:5],N);
%     hold on;[N, bin] = histc(result{6},[-5:.05:5]);
%     bar([-5:.05:5],N);
%     xlabel('Power ratio');
%     ylabel('Number');
    %%
    figure;subplot(411);
    histogram(result{5}(expert==2 | expert==3),'FaceColor','b','EdgeColor','r');
    hold on;histogram(result{5}(expert==1 | expert==3),'FaceColor','y');
    legend('Expert-2','Expert-1');
    xlabel('Power (dB)');
    subplot(412);
    histogram(duration(expert==2 | expert==3),'FaceColor','b','EdgeColor','r');
    hold on;histogram(duration(expert==1 | expert==3),'FaceColor','y');
    tmp = duration;
    tmp(result{5}(expert==2 | expert==3)<2)=[];
    histogram(tmp,'FaceColor','g');
    legend('Expert-2','Expert-1');
    xlabel('Duration (s)');
    subplot(414);
    [N, bin] = histc(result{6}(expert==2 | expert==3),[-5:.05:5]);
    bar([-5:.05:5],N,'b');
    hold on;[N, bin] = histc(result{6}(expert==1 | expert==3),[-5:.05:5]);
    bar([-5:.05:5],N,'g');
    legend('Expert-2','Expert-1');
    tmp1=result{6}(expert==2 | expert==3);
    tmp1(result{5}(expert==2 | expert==3)<2)=[];
    [N, bin] = histc(tmp1,[-5:.05:5]);
    bar([-5:.05:5],N,'r');
    xlabel('Power Ratio');
    subplot(413);
%     histogram(result{7}(expert==2 | expert==3),'FaceColor','b','EdgeColor','r');
%     hold on;histogram(result{7}(expert==1 | expert==3),'FaceColor','y');
    tmp2=result{7}(expert==2 | expert==3);
    tmp2(result{5}(expert==2 | expert==3)>2)=[];
%     histogram(tmp2,'FaceColor','r');
%     legend('Expert-2','Expert-1','Low power');
    scatter(result{7}(expert==2 | expert==3), result{5}(expert==2 | expert==3),'b');
    hold on;
    tmp3=result{5}(expert==2 | expert==3);
    tmp3(tmp3>2)=[];
    scatter(tmp2, tmp3,'r');
%% draw
    fp_EEG = zeros(size(EEG));
    for m=1:length(fp_result{1,1})
        fp_EEG(floor(fp_result{1,1}(m)*fs):floor((fp_result{1,1}(m)+fp_result{1,2}(m))*fs))=1;
    end
    start = Annotation_or(:,1);
    duration=Annotation_or(:,2);
    t_s=[];
    t_e=[];
    start_time = 1200;  % seconds
    end_time = start_time+30;
    datarange=(start_time+1/fs:1/fs:end_time);
    ind=find(start>datarange(1) & start<datarange(end));
    if ~isempty(ind)
        for i=1:length(ind)
            t_s(i)=start(ind(i));
            t_e(i)=t_s(i)+duration(ind(i));
        end
    end
    
    figure; 
    subplot(411);
    plot(datarange,EEG(floor(datarange*fs)));
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
    subplot(412)
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');     
    plot_matrix(S,t,f);
    % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
    hold on 
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
        plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
    end
    plot([1,29],[11 11],['b','-'],'linewidth',1);
    plot([1,29],[16 16],['b','-'],'linewidth',1);
    subplot(413);
    plot(datarange,smooth1(floor(datarange*fs)-49));
    hold on;
    plot(datarange,fp_EEG(floor(datarange*fs)),'r');
    plot(datarange,stages(floor(datarange*fs)),'g');
    ylim([-1,4]);
    xlim([min(datarange), max(datarange)]);
    ylabel('Predictions');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    subplot(414);
    plot(datarange,prob(floor(datarange*fs)-49));   %prob
    ylim([-1,2]);
    ylabel('Probabilities');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    xlim([min(datarange), max(datarange)]);
    %%
    bpEEG=bandpass(EEG(floor(datarange*fs)), 'bandpass');
    figure;
    subplot(311);
    plot(datarange,EEG(floor(datarange*fs)));
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
    subplot(312)
    [up_env,~]=envelope(bpEEG,1e2,'analytic');
    envelope(bpEEG,1e2,'analytic');
    hold on;
    for ii=1:length(t_s)
            plot([(t_s(ii)-start_time)*fs,(t_s(ii)-start_time)*fs],[-100 100],['r','-'],'linewidth',1);
            plot([(t_e(ii)-start_time)*fs,(t_e(ii)-start_time)*fs],[-100 100],['r','-'],'linewidth',1);
    end
    xlim([0,length(datarange)]);
    title('Band-passed EEG segment zero phase');
    ylabel('Amplitude (uV)');
    xlabel('Samples');
    set(gca,'fontsize',16);
    subplot(313)
    plot(datarange,smooth([diff(up_env,2);0;0]));
    xlim([min(datarange), max(datarange)]);
    ylabel('Amplitude (uV)');
    xlabel('Time (s)');
    set(gca,'fontsize',16);
    
    %%
    subplot(411);
    % [S,t,f]=mtspecgramc(EEG(datarange),movingwin,params);
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');     
    plot_matrix(S,t,f);
    % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
    hold on 
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
        plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
    end
    plot([1,29],[11 11],['b','-'],'linewidth',1);
    plot([1,29],[16 16],['b','-'],'linewidth',1);
%     grid on;
    subplot(412);
    energyDB1 = 10*log10(S);
    %DB_LF = mean(energyDB(:,3:8),2);
    DB_BB1 = mean(energyDB1(:,4:11),2);
    DB_spindle1 = mean(energyDB1(:,12:21),2);
    ratio = DB_spindle1./DB_BB1; 
    %plot(t,ratio,'-','linewidth',2);axis tight;
    plot(t,DB_spindle1,'-','linewidth',2);axis tight;
    ylabel('Spindle Power');
    set(gca,'fontsize',16);
    subplot(413);
    plot(t,DB_BB,'-','linewidth',2);axis tight;
    ylabel('Baseline Power');
    set(gca,'fontsize',16);
    subplot(414);
    plot(t,ratio,'-','linewidth',2);axis tight;
    ylabel('Power ratio');
    set(gca,'fontsize',16);
    ylim([0,1]);
    
    new_datarange=((start_time+t(1)+1/fs): 1/fs: (start_time+t(end)+1/fs ));
    data = smooth1(floor(new_datarange*fs)-49);
    temp = data;
    temp(ratio<=0.4)= 0;
    subplot(712); hold on;
    plot(new_datarange, temp,'r');
    
%end
