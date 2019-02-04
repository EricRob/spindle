clearvars;
addpath(genpath('chronux_2_11'));
% num_subject = {1,2,3,4,5,6,7,8,9,10};
num_subject = {2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
session={};
for i=1:length(num_subject)
    if num_subject{i}<10
        session={session{:},['01-02-000' num2str(num_subject{i})]};
    else
        session={session{:},['01-02-00' num2str(num_subject{i})]};
    end
end
% total_time = zeros(length(session), 1);
% database = cell(1,1);
% output = cell(1,3);   % 1-and / 2-or / 3-soft
% Spindles = cell(2,2);
target_f = 200;
for idx=1:length(session)
    total_time = 0;
    database = cell(1,1);
    output = cell(1,3);   % 1-and / 2-or / 3-soft
    Spindles = cell(2,2);
    % reading data from .edf file
    [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = MASS_Reader(session{idx}, 0, target_f);
    % appending data 
    [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, 1, total_time, fs);           

    % save MassDataBase database output Spindles
    std_value = zeros(size(Spindles{2,1}, 1),1);
    for i=1:size(Spindles{2,1}, 1)
        s_data=database{1}(floor(Spindles{2,1}(i)*fs):floor((Spindles{2,1}(i)+Spindles{2,2}(i))*fs));
        s_data = detrend(s_data);
        std_value(i) = std(s_data);
    end
    pd = fitdist(std_value,'Normal');
    disp(['======================================================']);
    disp(['========Working on subject - ' num2str(num_subject{idx})  '==========']);
    disp(mean(pd));
    
    % for i=2040:2070
    %     figure;
    %     s_data=database{1}((Spindles{1}(i)-0.5)*fs:floor((Spindles{1}(i)+Spindles{2}(i)+0.5)*fs));
    % %     s_data = detrend(s_data);
    %     plot([1:length(s_data)], s_data);
    %     hold on 
    %     s_data=database{1}(Spindles{1}(i)*fs:floor((Spindles{1}(i)+Spindles{2}(i))*fs));
    % %     s_data = detrend(s_data);
    %     plot([floor(0.5*fs)+1:length(s_data)+floor(0.5*fs)],s_data,'r');
    % end
    %% compute power ratio
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
        power_BB(k+floor(fs)-1)=DB_BB;
        power_spindle(k+floor(fs)-1)=DB_spindle;
        ratio(k+floor(fs)-1)=DB_spindle./DB_BB;
        bdpass = bandpass(seq, 'bandpass', [9,16], floor(fs));
        band_pass(k)=bdpass(1);
        [up,~]=envelope(bdpass);
        up_env(k) = up(1);
        power_feat(k+floor(fs)-1, :) = energyDB(:,4:22);
    end
    database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
    save(['./data/power_refine/MASSsub' num2str(num_subject{idx}) 'Power' num2str(floor(fs)) 'Hz.mat']);
    %% set subset fraction
    subset = 1;

    %  get baseline
    num_steps = floor((250*0.001)/(1/fs));   % sequency length of one example
    baseline=database{1};
    baseline(stages~=2, :)=[];  %remove stage except N2
    t_output = output{1};
    t_output(stages~=2, :)=[];  %remove spindles
    baseline(t_output==1,:)=[];
    baseline = baseline(1:floor(length(baseline)/num_steps)*num_steps, :);
    num_base = size(baseline,1)/num_steps; 
    baseline = permute(reshape(baseline, [num_steps, num_base, size(baseline,2)]),[1,3,2]);
    train_b = floor(num_base*0.7*subset);  % take 70% of the baseline as traning data
    valid_b = floor(num_base*0.2*subset);  % take 20% of the baseline as validation data
    test_b = floor(num_base*0.1*subset); % num_base-train_b-valid_b;
    ind_b = randperm(num_base);
    train_base = reshape(permute(baseline(:,:,ind_b(1:train_b)),[1,3,2]), [num_steps*train_b , size(baseline,2)]);
    valid_base = reshape(permute(baseline(:,:,ind_b(train_b+1:train_b+valid_b)),[1,3,2]), [num_steps*valid_b , size(baseline,2)]);
    test_base = reshape(permute(baseline(:,:,ind_b(train_b+valid_b+1:train_b+valid_b+test_b)),[1,3,2]), [num_steps*test_b , size(baseline,2)]);

    %% Augment Spindles
    S = [Spindles{2,1}, Spindles{2,2}];
    num_spindles = size(S,1);
    train_s = floor(num_spindles*0.7*subset);
    valid_s = floor(num_spindles*0.2*subset);
    test_s = floor(num_spindles*0.1*subset);%num_spindles-train_s-valid_s;
    ind_s = randperm(num_spindles);
    train_spindles = S(ind_s(1:train_s) , :);
    valid_spindles =  S(ind_s(train_s+1:train_s+valid_s), :);
    test_spindles = S(ind_s(train_s+valid_s+1:train_s+valid_s+test_s), :);

    h1=fopen(['data/augment_refine/Augment_train_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf.txt'],'wt');
    h2=fopen(['data/augment_refine/Augment_train_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf_labels.txt'],'wt');
    h3=fopen(['data/augment_refine/Augment_valid_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf.txt'],'wt');
    h4=fopen(['data/augment_refine/Augment_valid_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf_labels.txt'],'wt');
    h5=fopen(['data/augment_refine/Augment_test_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf.txt'],'wt');
    h6=fopen(['data/augment_refine/Augment_test_MASS_sub' num2str(num_subject{idx}) '_' num2str(num_steps) '_' num2str(subset*100) '%_f' num2str(target_f) '_mf_labels.txt'],'wt');

    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    format2 = '%d\n';
    %%
    augment_data(train_spindles, h1, h2, num_steps, database, output{1,2}, fs, 0,  format1, format2);
    augment_data(valid_spindles, h3, h4, num_steps, database, output{1,2}, fs, 0, format1, format2);
    augment_data(test_spindles, h5, h6, num_steps, database, output{1,2}, fs, 0, format1, format2);

    % add baseline to file
    fprintf(h1,format1,train_base');
    fprintf(h2,format2,zeros(size(train_base,1),1));
    fprintf(h3,format1,valid_base');
    fprintf(h4,format2,zeros(size(valid_base,1),1));
    fprintf(h5,format1,test_base');
    fprintf(h6,format2,zeros(size(test_base,1),1));

    fclose(h1);
    fclose(h2);
    fclose(h3);
    fclose(h4);
    fclose(h5);
    fclose(h6);

end