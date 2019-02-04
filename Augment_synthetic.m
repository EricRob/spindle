clearvars;
addpath(genpath('chronux_2_11'));

subjs = [3, 4, 5, 6, 7, 8, 1, 2];
    
for sub = 1:length(subjs)
    
    load(['re_synth_DREAMS_', num2str(subjs(sub)), '.mat']);
    
    %% 
    pd = synth_database.std;
    database{1} = synth_database.data;
    output{1,1} = synth_database.label(:);
    output{1,2} = synth_database.label(:);

    fs = 200;
    target_f = 200;
    %% power features
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
    power_feat = zeros(length(database{1}), 19);
    %pd = mean(fitdist(std(detrend(spindles))', 'Normal'));

    for k = 1:length(database{1})-floor(fs)+1

        seq = database{1}(k:k+floor(fs)-1);
        seq = (seq-mean(seq))/pd;
        [SS, tt, ff] = mtspecgramc(seq, movingwin, params);
        energyDB = 10*log10(SS);
        DB_BB = mean(energyDB(:,4:11),2);
        DB_spindle = mean(energyDB(:,12:22),2);
        power_BB(k+floor(fs)/2) = DB_BB;
        power_spindle(k+floor(fs)/2) = DB_spindle;
        ratio(k+floor(fs)/2) = DB_spindle./DB_BB;
        bdpass = bandpass(seq, 'bandpass', [9,16]);
        band_pass(k+floor(fs)/2) = bdpass(floor(fs)/2+1);
        [up, ~] = envelope(bdpass);
        up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
        power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
    end
    database{1} = [database{1}, band_pass, ratio, up_env, power_BB, power_spindle, power_feat];
    %% set subset fraction
    subset = 1;

    %%  get baseline
    num_steps = 50;   % sequency length of one example
    baseline = database{1};
    baseline(output{1} == 1) = [];
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

    %%load synthectic
    Spindles{1,1}=(find(diff(output{1,2})==1)+1)/fs;
    Spindles{1,2}=find(diff(output{1,2})==-1)/fs - Spindles{1,1};
    S = [Spindles{1,1}, Spindles{1,2}];
    % synth_database = synth_database1;
    num_spindles = size(S,1);
    train_s = floor(num_spindles*0.7*subset);
    valid_s = floor(num_spindles*0.2*subset);
    test_s = floor(num_spindles*0.1*subset);%num_spindles-train_s-valid_s;
    ind_s = randperm(num_spindles);


    train_spindles = S(ind_s(1:train_s) , :);
    valid_spindles =  S(ind_s(train_s+1:train_s+valid_s), :);
    test_spindles = S(ind_s(train_s+valid_s+1:train_s+valid_s+test_s), :);

    h1=fopen(['SleepSpindleData4RNN/Augment_train_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth.txt'],'wt');
    h2=fopen(['SleepSpindleData4RNN/Augment_train_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth_labels.txt'],'wt');
    h3=fopen(['SleepSpindleData4RNN/Augment_valid_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth.txt'],'wt');
    h4=fopen(['SleepSpindleData4RNN/Augment_valid_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth_labels.txt'],'wt');
    h5=fopen(['SleepSpindleData4RNN/Augment_test_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth.txt'],'wt');
    h6=fopen(['SleepSpindleData4RNN/Augment_test_DREAMS_sub' num2str(subjs(sub)) '_' num2str(num_steps) '_synth_labels.txt'],'wt');

    format1 = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    format2 = '%d\n';

    augment_data(train_spindles, h1, h2, num_steps, database, output{1,2}, fs, 0,  format1, format2, 2);
    augment_data(valid_spindles, h3, h4, num_steps, database, output{1,2}, fs, 0, format1, format2, 2);
    augment_data(test_spindles, h5, h6, num_steps, database, output{1,2}, fs, 0, format1, format2, 2);
    %% add baseline to file
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