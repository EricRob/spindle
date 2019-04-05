% Written by Charles Zhang
% June 2018
% Modified July 2018

% Graphs the contents of data along with spectrograms, probability of spindle as dictated by the
% NN, and binary detection based on power and time threshholds.
% "Data, Spectrogram, Detections"

% Input:
%   startTime -- The beginning timestamp in seconds of the segment you want to graph
%   endTime -- The ending timestamp in seconds of the segment you want to graph
%   fs -- Samping rate of data (NN output is always 200 Hz, make a variable if that changes)
%   data -- Single channel of EEG/LFP data. 
%   probRecord -- Probability (floating point, 0 to 1) data at 200 Hz.
%                 length(probRecord)*200 = length(data)*fs
% Output:
%   []
        
function [] = dataSpecDet(startTime, endTime, fs, data, probRecord)
    addpath(genpath('chronux_2_11'));
    rawUnit = 'µV'; % CHAT uses mV sometimes, most are µV, CHANGE THIS

    startFrame = startTime*fs + 1;
    endFrame = endTime*fs + 1;
    x = [0: 1/fs: endTime-startTime];
    y = data(startFrame:endFrame);
    
    figure('name', strcat(int2str(startTime), ' - ', int2str(endTime)));
    ax1 = subplot(3, 1, 1);
    plot(x, y, '-')%, legend('C3')
    title('ECOG Subject A06')
    xlabel('Time elapsed (s)')
    ylabel('Amplitude (µV)')
    
    hold on
    
    x = [0: 1/200: endTime - startTime];
    y = probRecord(startTime*200+1:endTime*200+1)';
    binProbs = makeBinary(y', 0.2, fs*0.5); % 500ms, 0.5
    plot(x, (binProbs-0.5)*1200, 'r')
    
    set(gca, 'FontSize', 16);
    
    
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[5 20];
    params.trialave=false;
    params.tapers=[3 5];
    
    params.err=0;
    params.pad=0;
    [S, t, f] = mtspecgramc(data(1,startFrame:endFrame),movingwin, params);
    colormap('jet');
    %disp([S, t, f]);
    
    ax2 = subplot(3, 1, 2);
    plot_matrix(S, t, f);
    hold on
    y=10;
    plot(t,y*ones(size(t)), 'r')
    hold on
    y = 15;
    plot(t,y*ones(size(t)), 'r')
    title('Spectrogram')
    xlabel('Time elapsed (s)')
    ylabel('Frequency (Hz)')
    set(gca, 'FontSize', 16);
    
    nnfs = 200; % neural network output rate (200hz)
    % Probability file begins here

    x = [0: 1/nnfs: endTime - startTime];
    %x = x(startTime*fs:endTime*fs);
    y = probRecord(startTime*nnfs+1:endTime*nnfs+1)';

    ax3 = subplot(3, 1, 3);
    plot(x, y)
    title('NN Output (Spindle Probabilities)')
    xlabel('Time elapsed (s)')
    ylabel('Probability (P)')
    set(gca, 'FontSize', 16);

    hold on

    plot(x, binProbs, 'r')
    legend('Probabilities', 'Final Detections');
        
    x = [0: 1/fs: endTime-startTime];
    y = data(startFrame:endFrame);
    
    linkaxes([ax1, ax2, ax3], 'x');
end


