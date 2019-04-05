function [] = generate_ECOG_data(sub, c3_std, channel, start_, end_)

    fs = 512;
    
    [~, data] = edfread(['R:\jwanglab\jwanglabspace\Param\data\Anli_ECoG_spindles\', sub, '\' , sub, '_Acoustic_Liu.edf']);

    outFile   = fopen(['test_data/test_ecog_', sub, '_50_100%_f200_mf.txt'], 'wt');
    dummyFile = fopen(['test_data/test_ecog_', sub, '_50_100%_f200_mf_labels.txt'] , 'wt');

    wo = 60/(fs/2);  bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    data = filtfilt(b,a,data(channel, start_:end_));

    data = resample(data', 200, 512);

    % a06 = 60.596038120423010
    % a08 = 29.052569717319660
    % a09 = 56.584019535192205
    %c3_std = 1288.5; % Standard deviation values for spindle amplitudes

    %%%%%%% First channel

    database = cell(1,1);
    database{1} = data; %'; % MAKE SURE THIS IS A COLUMN VECTOR

    movingwin=[1 1/fs];
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
        seq = (seq-mean(seq))/c3_std;
        [SS,tt,ff]=mtspecgramc(seq,movingwin,params);
        energyDB = 10*log10(SS);
        %     DB_LF = mean(energyDB(:,3:8),2);
        DB_BB = mean(energyDB(:,4:11),2);
        DB_spindle = mean(energyDB(:,12:22),2);
        power_BB(k+floor(fs)/2)=DB_BB;
        power_spindle(k+floor(fs)/2)=DB_spindle;
        ratio(k+floor(fs)/2)=DB_spindle./DB_BB;
        bdpass = bandpass(seq, 'bandpass',[9,16]);
        band_pass(k+floor(fs)/2)=bdpass(floor(fs)/2+1);
        [up,~]=envelope(bdpass);
        up_env(k+floor(fs)/2) = up(floor(fs)/2+1);
        power_feat(k+floor(fs)/2, :) = energyDB(:,4:22);
    end

    database{1} = [database{1},band_pass,ratio,up_env,power_BB,power_spindle, power_feat];
    sizeOfData = size(band_pass);
    sizeOfData = sizeOfData(1);
    dummyLabels = zeros(sizeOfData, 1);
    
    format = '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n';
    fprintf(outFile,format,database{1}');
    fclose(outFile);

    % We need to have labels to run the NN, but don't have any, so here we create a file with the
    % correct dimensions to feed to the algorithm.
    fprintf(dummyFile, '%d\n' , dummyLabels);
    fclose(dummyFile);

end