% Written by Charles Zhang
% June 2018

% Given a few parameters and a row vector EEG/LFP type data, graphs the data and its
% corresponding spectrogram

% Input:
%   startTime -- Time in seconds after beginning of data to begin graph
%   endTime -- Time in seconds after beginning of data to end graph
%   fs -- sampling rate, used to translate between seconds and sampling frames
%   records -- The data itself, a single row vector.
% Output:
%   [S, t, f] -- Spectrogram variables, not expected to be used.

function [S, t, f] = makeGraphsOld(startTime, endTime, fs, records)
    wo = 60/(fs/2);  bw = wo/35;
    [b,a] = iirnotch(wo,bw);
    records = filtfilt(b,a,records);

    addpath(genpath('chronux_2_11'));
    startFrame = startTime*fs + 1;
    endFrame = endTime*fs + 1;
   % x = [startTime: 1/fs: endTime];
    x = [0: 1/fs: endTime-startTime];
    y = records(startFrame:endFrame);
    
    figure('name', strcat(int2str(startTime), ' - ', int2str(endTime)));
    ax1 = subplot(2, 1, 1);
    plot(x, y, '-')
    title('Raw Output')
    xlabel('Time elapsed (s)')
    ylabel('Raw Output (uV/mV)')
    
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[5 20];
    params.trialave=false;
    params.tapers=[3 5];
    
    params.err=0;
    params.pad=0;
    [S, t, f] = mtspecgramc(records(startFrame:endFrame),movingwin, params);
    colormap('jet');
    %disp([S, t, f]);
    
    ax2 = subplot(2, 1, 2);
    plot_matrix(S, t, f);
    colorbar('off');
    hold on
    y=10;
    plot(t,y*ones(size(t)), 'r')
    hold on
    y = 15;
    plot(t,y*ones(size(t)), 'r')
    title('Spectrogram C3')
    xlabel('Time elapsed (s)')
    ylabel('Frequency (Hz)')
    
    linkaxes([ax1, ax2], 'x');
end


