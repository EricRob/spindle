function [] = generate_trimmed_data()
clear;
name = '300007';
study = 'chat';
target_f = 200;
stage = 2;

[header, data] = edfread(['./edfs/' study '-baseline-' name '.edf']);
index_C3 = find(contains(header.label,'C3'));
data = data(index_C3,:);

original_f = header.frequency(index_C3);

data = resample(data', target_f, original_f)';
data = isolateStage(stage, name, data, target_f);

if header.units{1,index_C3} == 'mV'
    data = data*1000;
end

save(['trimmedData/data_' study name '.mat'])
end