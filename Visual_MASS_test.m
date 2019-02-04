clearvars;
addpath(genpath('chronux_2_11'));
%num_subject = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
num_subject = {16,17,18,19};
session2={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session2={session2{:},['01-02-000' num2str(num_subject{i})]};
    else
        session2={session2{:},['01-02-00' num2str(num_subject{i})]};
    end
end
    
% total_time = zeros(length(session), 1);
% database = cell(1,1);
% output = cell(1,3);   % 1-and / 2-or / 3-soft
% Spindles = cell(2,2);
%% Get STD from wake data
% for idx = 1:length(session)
%     
%     [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = MASS_Reader(session{idx}, 1);
%     disp(session{idx});
%     std(EEG(stages == 0))
% end
%%
for idx2 = 1:length(session2)
    %[Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = MASS_Reader(session{idx}, 1);
    %[database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);           
    % evaluate latency
    
    session2{idx2}
    fs = 200;
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 99];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    %[SS,tt,ff]=mtspecgramc(EEG,movingwin,params);
    %energyDB = 10*log10(SS); 
%     DB_LF = mean(energyDB(:,3:8),2);
    %DB_BB = mean(energyDB(:,4:11),2);
    %DB_spindle = mean(energyDB(:,12:21),2);
%     save Testsub1
%
    %load('data/MASSsub1Power200Hz.mat');length(find(stages==2))*5/1000/60
%     load(['data/MASSsub' num2str(k) 'Power200Hz.mat']);disp([size(Annotation_and,1),size(Annotation_or,1)])

    s1 = strsplit(session2{idx2}, '-');
    
    load(['./data/MASSsub', num2str(str2num(s1{3})), 'Power200Hz.mat']);
    %load(strcat("./data/MASSsub",  num2str(str2num(s1{3})), "_EEG.mat")); % data
    
    %load(['./data/synthetic1Power.mat']);
    %save(strcat("./data/MASSsub", num2str(str2num(s1{3})), "_anno.mat"), "Annotation_or");
    %save(strcat("./data/MASSsub", num2str(str2num(s1{3})), "_EEG.mat"), "EEG");
    
    % Load NN results
    %f1=fopen('results/predictions(sub18-or).txt');
    %f2 = fopen(['results/MASS_STD_group/probability(sub', num2str(str2num(s1{3})), '_std).txt']);
    %f2 = fopen(['results/MASS_STD_group/probability(sub18_std_35.725).txt']);
    %f2 = fopen(['results/label_noise/probability(0.9_5).txt']);
    
%     f2=fopen('results/MASS_Resampling/200_from_50_probability_sub04.txt'); 
%     f2=fopen('results/probability(power_env).txt'); 
     %f2=fopen('results/Cross_valid/probability(sub3).txt');
%     f2=fopen('/home/wanglab/spindle_test/probability_1_plh.txt');
%     predict=fscanf(f1,'%d');
    
    %prob=fscanf(f2,'%f');
    %fclose(f1);
    %fclose(f2);
    %%
    num_step = 50;
    %prob = prob(num_step:end);
    
    % For comparing with McSleep
%     prob1 = load(['./results/mcsleep/', session{idx}, '.mat']);
%     prob1 = prob1.spindles;
%     prob1 = prob1';
%     prob1 = prob1(num_step:end);

    % For comparing with spindler
    spin2 = load(['./results/spindler/', session2{idx2}, '.mat']);
    fs1 = 200;
    spin2 = spin2.events;
    prob2 = zeros(length(EEG), 1);
    for spin = 1:length(spin2)
        if spin2(spin, 1)*fs1 > length(EEG) || spin2(spin, 2)*fs1 > length(EEG)
            continue;
        end
        if spin2(spin, 1)*fs1 == 0 || spin2(spin, 2)*fs1 == 0
            continue;
        end
        prob2(round(spin2(spin, 1)*fs1):round(spin2(spin, 2)*fs1)) = 1;
    end
    prob = prob2;
    %%
    prob_th=0.9;
    time_th=20;
    if 0
        range = [0.01:0.01:0.04, 0.05:0.05:0.95, 0.96:0.01:1];
%         range= [0:5:100];
        sensitivity = zeros(length(range),1);
        specificity = zeros(length(range),1);
        fdr = zeros(length(range),1);
        f1score = zeros(length(range),1);
        stat = struct('Prob_th',[],'TP',[],'TN',[],'FP',[],'FN',[]);
        for i=1:length(range)
%             time_th=range(i);
            prob_th=range(i);
            [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 0);
            stat.Prob_th = [stat.Prob_th;prob_th];
            stat.TP = [stat.TP;eval.TP];
            stat.TN = [stat.TN;eval.TN];
            stat.FP = [stat.FP;eval.FP];
            stat.FN = [stat.FN;eval.FN];
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
        figure(1);hold on; plot(1- specificity,sensitivity,'.-','Linewidth',2);ylabel('sensitivity');xlabel('1-specificity');xlim([0,1]);ylim([0,1]);
        [~, I]=min((1-sensitivity).^2+(1-specificity).^2);
        disp(range(I));
    else
        [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
        disp(eval);disp(['Accuracy: '  num2str((eval.TP+eval.TN)/(eval.TP+eval.TN+eval.FP+eval.FN))]);
    end
end
    %%
    % Added by Eric:
    % display stats and AUC in same run (copy of above code without the
    % if/else statements).
    target_f = 200;
    original_f = 34;
    subjects = {5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    milliseconds = 100;
    for idx=1:length(subjects)
        subject = subjects{idx};
        disp(['==================== Subject ' num2str(subject) ' ====================']);
        load(['./data/MASSsub' num2str(subject) 'Power' num2str(target_f) 'Hz.mat']);
        % Load NN results
        %f1=fopen('results/predictions(sub18-or).txt');
        if subject < 10
            subject = ['0' num2str(subject)];
        else
            subject = num2str(subject);
        end
        
        if target_f==original_f
            % Testing on original frequency:
            f2=fopen(['results/MASS_' num2str(target_f) 'Hz/probability_sub' subject '_' num2str(target_f) 'Hz.txt']);
        else
            % Testing on upsampled data:
            f2=fopen(['results/MASS_Resampling/' num2str(target_f) '_from_' num2str(original_f) '_probability_sub' subject '.txt']);
        end
        %     f2=fopen('results/probability(power_env).txt');
        %     predict=fscanf(f1,'%d');
        prob=fscanf(f2,'%f');
        %fclose(f1);
        fclose(f2);
        num_step = floor((250*0.001)/(1/target_f));
        prob=prob(num_step:end);
        prob_th=0.8;
        time_th=floor((milliseconds*0.001)/(1/target_f));
        [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
        disp(eval);
        range = [0.05:0.05:0.95, 0.96:0.01:1];
        %         range= [0:5:100];
        sensitivity = zeros(length(range),1);
        specificity = zeros(length(range),1);
        fdr = zeros(length(range),1);
        f1score = zeros(length(range),1);
        stat = struct('Prob_th',[],'TP',[],'TN',[],'FP',[],'FN',[]);
        for i=1:length(range)
            %             time_th=range(i);
            prob_th=range(i);
            [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 0);
            stat.Prob_th = [stat.Prob_th;prob_th];
            stat.TP = [stat.TP;eval.TP];
            stat.TN = [stat.TN;eval.TN];
            stat.FP = [stat.FP;eval.FP];
            stat.FN = [stat.FN;eval.FN];
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
        figure;
        %.plot(1- specificity,sensitivity,'.-');ylabel('sensitivity');xlabel('1-specificity');xlim([0,1]);ylim([0,1]);
        [~, I]=min((1-sensitivity).^2+(1-specificity).^2);
        disp(range(I));
        
    end

    %% Scatter plot of onset latency of 2 experts
    figure;
    scatterhist(gap(:,1),gap(:,2), 'Location', 'SouthEast', 'Direction', 'out', 'NBins',[15,15]);
    ylabel('duration (s)');
    xlabel('onset latency (s)');
%     legend('expert1 - expert2');
    set(gca,'fontsize',32);
%     subplot(211);histogram(gap(:,1));legend('expert1-expert2');
%             
%     subplot(212);scatter(gap(:,1),gap(:,2));ylabel('duration');
    %% performance comparison between more traningdata vs less traningdata
    f2=fopen('results/probability(power_env).txt'); 
    prob1=fscanf(f2,'%f');
    %fclose(f1);
    fclose(f2);
    num_step = 50;
    prob1=prob1(num_step:end);
    [~, ~, ~, ~, ~, ~, ~,~, probabilty1]=eval_performance(prob1, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
    figure;
    h = [0:.04:1];
    [n1,bin1] = histc(probabilty,h);
    [n2,bin2] = histc(probabilty1,h); 
    hold on; 
    plot(h(1:end-1),n1(1:end-1),'r-','linewidth',2)
    plot(h(1:end-1)+0.01,n2(1:end-1),'b-','linewidth',2)
  
    bar(h,n1,'r');
    %histogram(probabilty2,[0:.01:1]);

    %histogram(probabilty1,[0:.01:1]);

    bar(h+0.01,n2,'b');
    plot(h(1:end-1),n1(1:end-1),'r-','linewidth',2)
    plot(h(1:end-1)+0.01,n2(1:end-1),'b-','linewidth',2)
    hold off
    alpha(0.3);
    legend('more training data','less training data');
    ylabel('Number');
    xlabel('Probability');
    xlim([0 1]);
    set(gca,'fontsize',18)
    %% compute instantaneous frequency
    instfreq = zeros(length(start),1);
    instfreq1 = zeros(length(start),1);
    for kk=1:length(start)
        datarange = [start(kk)+1/fs:1/fs:start(kk)+duration(kk)];
        y = bandpass(EEG(floor(datarange*fs)),'bandpass',[9,16]);
        z = hilbert(y);
        instfreq1(kk) = mean((fs/(2*pi)*diff(unwrap(angle(z)))));      
        [pxx,f]=pspectrum(y,fs);[M,I]=max(pxx);
        instfreq(kk) = f(I);
    end
    figure;scatter(instfreq1,instfreq,'filled');hold on; plot([9,16],[9,16]);
%     figure;histogram(instfreq);xlabel('frequency (Hz)');ylabel('number of spindles');
    %% Comparison between of duration and power of FPs and GT
    duration=Annotation_or(:,2);
    figure;subplot(211);
    histogram(fp_result{3});clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,5,6,7,9,10,11,12,13,14,17,18,19};
num_subject = {1};
session={};
    for i=1:length(num_subject)
        if num_subject{i}<10
            session={session{:},['01-02-000' num2str(num_subject{i})]};
        else
            session={session{:},['01-02-00' num2str(num_subject{i})]};
        end
    end
    
total_time = zeros(length(session), 1);
database = cell(1,1);
output = cell(1,3);   % 1-and / 2-or / 3-soft
Spindles = cell(2,2);
for idx=1:length(session)
    [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = MASS_Reader(session{idx}, 1);
    [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs);           
    %% evaluate latency
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[0 99];
    params.tapers=[3 5];
    params.err=0;
    params.pad=0;
    [SS,tt,ff]=mtspecgramc(EEG,movingwin,params);
    energyDB = 10*log10(SS); 
%     DB_LF = mean(energyDB(:,3:8),2);
    DB_BB = mean(energyDB(:,4:11),2);
    DB_spindle = mean(energyDB(:,12:21),2);
%     save Testsub1
%%
    %load('data/MASSsub1Power200Hz.mat');length(find(stages==2))*5/1000/60
%     load(['data/MASSsub' num2str(k) 'Power200Hz.mat']);disp([size(Annotation_and,1),size(Annotation_or,1)])
    load('./data/MASSsub19Power200Hz.mat');
    % Load NN results
    %f1=fopen('results/predictions(sub18-or).txt');
    %
%     f2=fopen('results/MASS_Resampling/200_from_50_probability_sub04.txt'); 
%     f2=fopen('results/probability(power_env).txt'); 
%%
    f2=fopen('results/MASS_input/probability(only_EEG_sub19).txt');
%     f2=fopen('/home/wanglab/spindle_test/probability_1_plh.txt');
%     predict=fscanf(f1,'%d');
    prob=fscanf(f2,'%f');
    %fclose(f1);
    fclose(f2);
    num_step = 50;
    prob=prob(num_step:end);
    %%
    prob_th=0.9;
    time_th=20;
    if 0
        range = [0.01:0.01:0.04, 0.05:0.05:0.95, 0.96:0.01:1];
%         range= [0:5:100];
        sensitivity = zeros(length(range),1);
        specificity = zeros(length(range),1);
        fdr = zeros(length(range),1);
        f1score = zeros(length(range),1);
        stat = struct('Prob_th',[],'TP',[],'TN',[],'FP',[],'FN',[]);
        for i=1:length(range)
%             time_th=range(i);
            prob_th=range(i);
            [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 0);
            stat.Prob_th = [stat.Prob_th;prob_th];
            stat.TP = [stat.TP;eval.TP];
            stat.TN = [stat.TN;eval.TN];
            stat.FP = [stat.FP;eval.FP];
            stat.FN = [stat.FN;eval.FN];
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
        figure(1);hold on; plot(1- specificity,sensitivity,'.-','Linewidth',2);ylabel('sensitivity');xlabel('1-specificity');xlim([0,1]);ylim([0,1]);
        [~, I]=min((1-sensitivity).^2+(1-specificity).^2);
        disp(range(I));
    else
        [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
        disp(eval);disp(['Accuracy: '  num2str((eval.TP+eval.TN)/(eval.TP+eval.TN+eval.FP+eval.FN))]);
    end
    
    %%
    % Added by Eric:
    % display stats and AUC in same run (copy of above code without the
    % if/else statements).
    target_f = 100;
    original_f = 100;
    subject = 4;
    milliseconds = 100;
    
    load(['./data/MASSsub' num2str(subject) 'Power' num2str(target_f) 'Hz.mat']);
    % Load NN results
    %f1=fopen('results/predictions(sub18-or).txt');  
    
    if target_f==original_f
        % Testing on original frequency:
        f2=fopen(['results/MASS_' num2str(target_f) 'Hz/probability_sub0' num2str(subject) '_' num2str(target_f) 'Hz.txt']);
    else
        % Testing on upsampled data:
        f2=fopen(['results/MASS_Resampling/' num2str(target_f) '_from_' num2str(original_f) '_probability_sub0' num2str(subject) '.txt']);
    end
%     f2=fopen('results/probability(power_env).txt'); 
%     predict=fscanf(f1,'%d');
    prob=fscanf(f2,'%f');
    %fclose(f1);
    fclose(f2);
    num_step = floor((250*0.001)/(1/target_f));
    prob=prob(num_step:end);
    prob_th=0.8;
    time_th=floor((milliseconds*0.001)/(1/target_f));
    [sens, spec, fdr, f1score, result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
    disp(eval);
    range = [0.05:0.05:0.95, 0.96:0.01:1];
    %         range= [0:5:100];
    sensitivity = zeros(length(range),1);
    specificity = zeros(length(range),1);
    fdr = zeros(length(range),1);
    f1score = zeros(length(range),1);
    stat = struct('Prob_th',[],'TP',[],'TN',[],'FP',[],'FN',[]);
    for i=1:length(range)
        %             time_th=range(i);
        prob_th=range(i);
        [sensitivity(i), specificity(i), fdr(i),  f1score(i), result, fp_result, smooth1,eval, probabilty]=eval_performance(prob, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 0);
        stat.Prob_th = [stat.Prob_th;prob_th];
        stat.TP = [stat.TP;eval.TP];
        stat.TN = [stat.TN;eval.TN];
        stat.FP = [stat.FP;eval.FP];
        stat.FN = [stat.FN;eval.FN];
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
        
    

    %% Scatter plot of onset latency of 2 experts
    figure;
    scatterhist(gap(:,1),gap(:,2), 'Location', 'SouthEast', 'Direction', 'out', 'NBins',[15,15]);
    ylabel('duration (s)');
    xlabel('onset latency (s)');
%     legend('expert1 - expert2');
    set(gca,'fontsize',32);
%     subplot(211);histogram(gap(:,1));legend('expert1-expert2');
%             
%     subplot(212);scatter(gap(:,1),gap(:,2));ylabel('duration');
    %% performance comparison between more traningdata vs less traningdata
    f2=fopen('results/probability(power_env).txt'); 
    prob1=fscanf(f2,'%f');
    %fclose(f1);
    fclose(f2);
    num_step = 50;
    prob1=prob1(num_step:end);
    [~, ~, ~, ~, ~, ~, ~,~, probabilty1]=eval_performance(prob1, fs, tt, power_spindle, power_BB,  stages, Annotation_or, prob_th, time_th, num_step, 1);
    figure;
    h = [0:.04:1];
    [n1,bin1] = histc(probabilty,h);
    [n2,bin2] = histc(probabilty1,h); 
    hold on;histogram(result{5});
    legend('FP','GT');
%     temp1 = result{2};
%     temp2 = result{4};
%     temp2(temp1==999)=[];
%     temp1(temp1==999)=[];
%     histogram(DB_spindle(floor((temp1+temp2/2-tt(1))*fs)));
    xlabel('Normalized Power (DB)');
    ylabel('Number');
    set(gca,'fontsize',32);
    subplot(212);histogram(fp_result{2});
    hold on;histogram(duration);
    xlabel('Duration (s)');
    ylabel('Number');
    set(gca,'fontsize',32);
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
    duration = Annotation_or(:,2);
    figure;subplot(311);
    histogram(result{5}(expert==2 | expert==3),[-30:-10],'FaceColor','b','EdgeColor','r');
    hold on;histogram(result{5}(expert==1 | expert==3),[-30:-10],'FaceColor','y','EdgeColor','k');
    legend('Expert-2','Expert-1');
    xlabel('Normalized Power (dB)');
    set(gca,'fontsize',32);
    subplot(312);
    histogram(duration(expert==2 | expert==3),[0:.1:4],'FaceColor','b','EdgeColor','r');
    hold on;histogram(duration(expert==1 | expert==3),[0:.1:4],'FaceColor','y','EdgeColor','k');
%     tmp = duration;
%     tmp(result{5}(expert==2 | expert==3)<2)=[];
%     histogram(tmp,'FaceColor','g');
    legend('Expert-2','Expert-1');
    xlabel('Duration (s)');
    set(gca,'fontsize',32);
    subplot(313);   
    r = [0:.1:4];
    histogram(result{6}(expert==2 | expert==3),r,'FaceColor','b','EdgeColor','r');
    hold on;histogram(result{6}(expert==1 | expert==3),r,'FaceColor','y','EdgeColor','k');
%     [N, bin] = histc(result{6}(expert==2 | expert==3),r);
%     bar(r,N,'b');
%     hold on;[N, bin] = histc(result{6}(expert==1 | expert==3),r);
%     bar(r,N,'y');
    legend('Expert-2','Expert-1');
    xlabel('Power Ratio');
    set(gca,'fontsize',32);
%     xlim([0,4]);
    
%     subplot(413);
%     tmp2=result{7}(expert==2 | expert==3);
%     tmp2(result{5}(expert==2 | expert==3)>2)=[];
%     scatter(result{7}(expert==2 | expert==3), result{5}(expert==2 | expert==3),'b');
%     hold on;
%     tmp3=result{5}(expert==2 | expert==3);
%     tmp3(tmp3>2)=[];
%     scatter(tmp2, tmp3,'r');
%% Draw standard results
    fp_EEG = zeros(size(EEG));
    for m=1:length(fp_result{1,1})
        fp_EEG(floor(fp_result{1,1}(m)*fs):floor((fp_result{1,1}(m)+fp_result{1,2}(m))*fs))=1;
    end
%%
    start = Annotation_or(:,1);
    duration=Annotation_or(:,2);
    t_s=[];
    t_e=[];
    start_time = 1; %2471;  % seconds
    window = 0.25;
    end_time = start_time+window;
    datarange=(start_time+1/fs:1/fs:end_time);
    ind=find(start>datarange(1) & start<datarange(end));
    if ~isempty(ind)
        for i=1:length(ind)
            t_s(i)=start(ind(i));
            t_e(i)=t_s(i)+duration(ind(i));
        end
    end
  
    figure; 
    subplot(311);
    plot(datarange,EEG(floor(datarange*fs)),'Linewidth',2);
    set(gca,'ylim',[-100,100]);
    hold on;
    if ~isempty(ind)
        for ii=1:length(t_s)
            plot([t_s(ii),t_s(ii)],[-100 100],['r','-'],'linewidth',2);
            plot([t_e(ii),t_e(ii)],[-100 100],['r','-'],'linewidth',2);
        end
    end
%     plot(datarange,bandpass(EEG(floor(datarange*fs)),'bandpass',[0.5,2]),'Linewidth',2,'color','k');
%     det = find(smooth1(floor(datarange*fs)-49)==1);
%     tmp = EEG(floor(datarange*fs)); 
%     plot(datarange(det),tmp(det),'Linewidth',2,'color','r');
%     [pxx,f]=pspectrum(bandpass(tmp(det),'bandpass',[9,16]),fs);[M,I]=max(pxx);title(['Spindle frequency: ' num2str(f(I))]);
%     z=hilbert(bandpass(tmp(det),'bandpass',[9,16])); 
%     title(['Spindle frequency: ' num2str(mean((fs/(2*pi)*diff(unwrap(angle(z))))))]); 
%     plot(datarange, 50*output{1,2}(floor(datarange*fs)));
    %xlim([min(datarange)+0.5, max(datarange)-0.5]);
%     xlabel('Time (s)');
    ylabel('Amptitude (uV)');
    set(gca,'fontsize',24);
    grid off;
    subplot(312)
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');     
    plot_matrix(S,t,f);
    % spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
    hold on 
    for i=1:length(t_s)
        plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
        plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',2);
    end
    plot([0.5,window+.5],[11 11],['b','-'],'linewidth',2);
    plot([0.5,window+.5],[16 16],['b','-'],'linewidth',2);
    set(gca,'fontsize',24);
    grid on;
    subplot(313);
    plot(datarange,smooth1(floor(datarange*fs)-49),'linewidth',2,'Color','r');
%     hold on;
%     plot(datarange,fp_EEG(floor(datarange*fs)),'r');
%     plot(datarange,stages(floor(datarange*fs)),'g');
    ylim([-1,4]);
    xlim([min(datarange)+0.5, max(datarange)-0.5]);
    ylabel('Predictions');
    set(gca,'fontsize',24);
%     xlabel('Time (s)');
    grid on;
    
    hold on;
    plot(datarange,prob(floor(datarange*fs)-49),'linewidth',2,'Color','b');   %prob
    plot(datarange,prob1(floor(datarange*fs)-49),'linewidth',2,'Color','g'); 
    ylim([-1,2]);
    ylabel('Probabilities');
    xlabel('Time (s)');
    set(gca,'fontsize',24);
    xlim([min(datarange)+0.5, max(datarange)-0.5]);
    legend('Prediction','Probability','McSleep');
    grid on;
    %%
    bpEEG=bandpass(EEG(floor(datarange*fs)), 'bandpass', [9, 16]);
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
    figure;
    subplot(411);
    % [S,t,f]=mtspecgramc(EEG(datarange),movingwin,params);
    [S,t,f]=mtspecgramc(EEG(floor(datarange*fs)),movingwin,params);colormap('jet');     
    plot_matrix(S,t,f);
    spectrogram(EEG(datarange),kaiser(256,10),255,512,100,'yaxis');
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
    plot(t,DB_BB1,'-','linewidth',2);axis tight;
    ylabel('Baseline Power');
    set(gca,'fontsize',16);
    subplot(414);
    plot(t,ratio,'-','linewidth',2);axis tight;
    ylabel('Power ratio');
    set(gca,'fontsize',16);
    ylim([0,1]);
    
%     new_datarange=((start_time+t(1)+1/fs): 1/fs: (start_time+t(end)+1/fs ));
%     data = smooth1(floor(new_datarange*fs)-49);
%     temp = data;
%     temp(ratio<=0.4)= 0;
%     subplot(415); hold on;
%     plot(new_datarange, temp,'r');
    
end

%% plot MASS statistcs
range = [1:19];
NREAM_duration = [248.332,303.6645,165.3321,238.6651,210.9987,225.332,231.3316,181.9988,246.3315,231.9984,212.6653,229.6653,272.6644,243.3321,167.9989,174.3323,265.6651,244.9985,267.9982];
num_expert1 = [1044,1143,143,253,341,150,912,385,814,795,606,709,700,712,97,452,470,1164,316];
num_expert2 = [2445,2210,602,0,1198,840,1604,0,1665,1939,1541,1202,1438,1616,0,0,1191,1680,1058];
mean_onset = [0.17325,0.16621,0.1478,0,0.14279,0.13863,0.15503,0,0.18472,0.15844,0.18003,0.15426,0.20578,0.1873,0,0,0.21533,0.11278,0.18181];
std_onset = [0.14288,0.16642,0.16326,0,0.14779,0.13452,0.18229,0,0.17314,0.15202,0.16266,0.17125,0.16936,0.14806,0,0,0.16303,0.16533,0.14934];
num_spindle = [880,968,122,253,317,119,633,385,595,651,495,557,476,513,97,452,369,991,288];
spindle_std = [13.8275,14.9942,15.5654,12.9483,12.2577,14.8564,17.8164,19.8062,17.2921,19.2211,14.5055,19.9593,15.3751,16.2978,14.1444,23.005,15.7555,17.8625,11.5192];
figure;subplot(411);bar(range,NREAM_duration);ylabel('NEAM Duration');
subplot(412);bar(range,num_expert2);hold on;bar(range,num_expert1);legend('Expert-2','Expert-1');ylabel('Num Spindles');
subplot(413);bar(range,mean_onset);hold on;errorbar(range,mean_onset,std_onset./sqrt(num_spindle),'.');ylabel('Onset difference');
subplot(414);bar(range,spindle_std);ylabel('Spindle Std');
%% plot cross-validation result
sensitivity = [0.8739,0.92701,0.71429,0.96047,0.93535,0.71527,0.89001,0.9974,0.78822,0.73548,0.95397,0.88544,0.97107,0.9774,0.98969,0.98451,0.90867,0.94441,0.96041];
specificity = [0.98885,0.98295,0.9971,0.94042,0.96336,0.99186,0.96716,0.8895,0.99263,0.99873,0.95967,0.98052,0.94136,0.90183,0.95893,0.92875,0.98286,0.94909,0.96085];
FDR = [0.056682,0.10887,0.057203,0.77521,0.26825,0.14187,0.19043,0.75194,0.060127,0.0096962,0.22145,0.16806,0.34856,0.41446,0.81028,0.61572,0.17614,0.27205,0.36012];
F1score = [0.90728,0.90872,0.81279,0.36432,0.82112,0.78021,0.84789,0.49771,0.85739,0.84408,0.85738,0.85786,0.77977,0.73234,0.31841,0.5528,0.86419,0.82218,0.76804];
AUROC = [0.99445,0.99514,0.99541,0.97598,0.99002,0.99022,0.9864,0.9808,0.99055,0.99534,0.99233,0.9912,0.99085,0.98073,0.99381,0.98582,0.99444,0.98881,0.99274];
range = [1:19];
figure;
subplot(511);bar(range,sensitivity);ylabel('sensitivity');
subplot(512);bar(range,specificity);ylabel('specificity');
subplot(513);bar(range,FDR);ylabel('FDR');
subplot(514);bar(range,F1score);ylabel('F1score');
subplot(515);bar(range,AUROC);ylabel('AUROC');
%%
figure;
plot(sensitivity,'linewidth',2);hold on;plot(specificity,'linewidth',2);hold on;plot(FDR,'linewidth',2);hold on;plot(F1score,'linewidth',2);hold on;plot(AUROC,'linewidth',2);
legend('sensitivity','specificity','FDR','F1score','AUROC');