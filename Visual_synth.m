clearvars;
addpath(genpath('chronux_2_11'));

load('synth_DREAMS_3.mat');
%%

pd = synth_database.std;
database = synth_database.data';
output{1,1} = synth_database.label(:);
output{1,2} = synth_database.label(:);
fs = 200;
%%
Spindles{1,1}=(find(diff(output{1,2})==1)+1)/fs;
Spindles{1,2}=find(diff(output{1,2})==-1)/fs - Spindles{1,1};
S = [Spindles{1,1}, Spindles{1,2}];
start = S(:, 1);
duration = S(:, 2);
%%
start_time = 0;  % seconds
end_time = start_time+30*60;
datarange = (start_time+1/fs:1/fs:end_time);
ind = find(start>datarange(1) & start<datarange(end));
t_s = [];
t_e = [];
if ~isempty(ind)
    for i = 1:length(ind)
        t_s(i) = start(ind(i));
        t_e(i) = t_s(i) + duration(ind(i));
    end
end

figure;
subplot(211);
plot(datarange, database(floor(datarange*fs)));
set(gca,'ylim',[-100,100])
hold on;
if ~isempty(ind)
    for ii = 1:length(t_s)
        plot([t_s(ii),t_s(ii)],[-100 100],['r','-'],'linewidth',1);
        plot([t_e(ii),t_e(ii)],[-100 100],['r','-'],'linewidth',1);
    end
end

movingwin = [1 1/fs];
params.Fs = fs;
params.fpass = [0 25];
params.tapers = [3 5];
params.err = 0;
params.pad = 0;

[S,t,f] = mtspecgramc(database(floor(datarange*fs)), movingwin, params);
subplot(212);
plot_matrix(S, t, f);colormap('jet');

hold on 
for i=1:length(t_s)
    plot([t_s(i),t_s(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
    plot([t_e(i),t_e(i)]-datarange(1),[0 50],['r','-'],'linewidth',1);
end
plot([1,29],[11 11],['b','-'],'linewidth',1);
plot([1,29],[16 16],['b','-'],'linewidth',1);

