clearvars;
addpath(genpath('chronux_2_11'));

num_subject = {1,2,3,4,5,6,7,8};
session={};
for i=1:length(num_subject)
    session={session{:},['excerpt' num2str(num_subject{i})]};
end
%% Generate synthetic params for DREAMS data

params_all = cell(size(session));
syn_pass_all = cell(size(session));

x0 = [0, 0, -10, 0, 84.5, -0.9]; % reasonable initialization based on MASS dataset params
fun=@(x,xdata)exp(x(1)+x(2)*xdata+x(3)*xdata.*xdata).*cos(x(4)+x(5)*xdata+x(6)*xdata.*xdata);

for idx=1:length(session)
    total_time = 0;
    database = cell(1,1);
    output = cell(1,3);   % 1-and / 2-or / 3-soft
    Spindles = cell(2,2);
    % reading data form .edf file
    [Annotation_and, Annotation_or, Annotation_soft, EEG, stages, gap, target_f, fs, expert] = DREAMS_Reader(session{idx}, 0);
    [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, 1, total_time, fs); 
    
    params = [];
    syn_pass = {};
    for i=1:size(Spindles{2,1}, 1)
        s_data=database{1}(floor(Spindles{2,1}(i)*fs):floor((Spindles{2,1}(i)+Spindles{2,2}(i))*fs));
        s_data = detrend(s_data)';
        s_pass = bandpass(s_data, 'bandpass', [9,16]);
        sequence_length = length(s_pass)/fs;
        range = -sequence_length/2+1/fs:1/fs:sequence_length/2;
        x = lsqcurvefit(fun,x0,range,s_pass);
        params = [params; x];
        syn_pass = [syn_pass; fun(x, range)];
    end
    
    params_all{idx} = params;
    syn_pass_all{idx} = syn_pass;
end

params_all = reshape(params_all, [length(session) 1]);
%% Save params and raw spindles
save('synth_DREAMS.mat', 'params_all', 'syn_pass_all');