% Written by Charles Zhang
% June 2018

% Graphs the contents of records along with spectrograms, probability of spindle as dictated by the
% NN, and binary detection based on power and time threshholds.

% Input:
%   startTime -- The beginning timestamp in seconds of the segment you want to graph
%   endTime -- The ending timestamp in seconds of the segment you want to graph
%   fs -- Samping rate of records (NN output is always 200 Hz, make a variable if that changes)
%   records -- records generated  by edfread, pared down to only C3 and C4 EEG channels. 
%   probRecord -- The entire output of the neural network, read into a matrix.
%   argHasProbs -- Tells whether we have a probability file to graph (1 if true, 0 if not)
%                   If 0, simply graphs both spectrograms instead
%   argWhichData -- Tells which EEG channel to use. If 2, plots the C4 EEG instead of C3.
% Output:
%   []
        
function [] = makeGraphs(startTime, endTime, fs, records, probRecord, argHasProbs, argWhichData)
    hasProbs = 0;
    whichData = 0;
    if nargin > 4
        hasProbs = argHasProbs;
    end
    if nargin > 5
        whichData = argWhichData;
    end
    
    rawUnit = 'mV'; % CHAT uses mV, most are uV, CHANGE THIS

    startFrame = startTime*fs;
    endFrame = endTime*fs;
   % x = [startTime: 1/fs: endTime];
    x = [0: 1/fs: endTime-startTime];
    y = records(1,startFrame:endFrame);
    z = records(1,startFrame:endFrame);
    
    figure('name', strcat(int2str(startTime), ' - ', int2str(endTime)));
    subplot(3, 1, 1)
    plot(x, y, x, z, '-'), legend('C3', 'C4')
    title('Raw Output')
    xlabel('Time elapsed (s)')
    ylabel('Raw Output')
    
    movingwin=[1 1/fs];
    params.Fs=fs;
    params.fpass=[5 20];
    params.trialave=false;
    params.tapers=[3 5];
    
    params.err=0;
    params.pad=0;
    [S, t, f] = mtspecgramc(records(1,startFrame:endFrame),movingwin, params);
    colormap('jet');
    %disp([S, t, f]);
    
    subplot(3, 1, 2)
    plot_matrix(S, t, f);
    hold on
    y=10;
    plot(t,y*ones(size(t)), 'r')
    hold on
    y = 15;
    plot(t,y*ones(size(t)), 'r')
    title('Spectrogram C3')
    xlabel('Time elapsed (s)')
    ylabel('Frequency (Hz)')
    
    [S, t, f] = mtspecgramc(records(1,startFrame:endFrame),movingwin, params);
    
    if hasProbs == 0
        subplot(3, 1, 3)
        plot_matrix(S, t, f);
        hold on
        y = 10;
        plot(t,y*ones(size(t)), 'r')
        hold on
        y = 15;
        plot(t,y*ones(size(t)), 'r')
        title('Spectrogram C4')
        xlabel('Time elapsed (s)')
        ylabel('Frequency (Hz)')
    elseif whichData == 2
        subplot(3, 1, 2)
        
        plot_matrix(S, t, f);
        colorbar('off');
        hold on
        y = 10;
        plot(t,y*ones(size(t)), 'r')
        hold on
        y = 15;
        plot(t,y*ones(size(t)), 'r')
        title('Spectrogram C4')
        xlabel('Time elapsed (s)')
        ylabel('Frequency (Hz)')
    end
    
    if hasProbs == 1
        nnfs = 200; % neural network output rate (200hz)
        % Probability file begins here
        
        x = [0: 1/nnfs: endTime - startTime];
        %x = x(startTime*fs:endTime*fs);
        y = probRecord(startTime*nnfs:endTime*nnfs)';

        subplot(3, 1, 3)
        plot(x, y)
        title('NN Output (Spindle Probabilities)')
        xlabel('Time elapsed (s)')
        ylabel('Probability (P)')
        
        hold on
        
        binProbs = makeBinary(y, 0.9, 100); % 500ms, 0.9
        %subplot(4, 1, 4)
        plot(x, binProbs, 'r')
        %title('NN Threshold Binary Values')
        %xlabel('Time elapsed (s)')
        %ylabel('Spindle? (y/n)')
        
        x = [0: 1/fs: endTime-startTime];
        y = records(1,startFrame:endFrame);
        z = records(2,startFrame:endFrame);
        
        subplot(3, 1, 1)
        %plot(x, y, x, z, '-'), legend('C3', 'C4')
        plot(x, y, '-')
        title('Raw Output')
        xlabel('Time elapsed (s)')
        ylabel(['Raw Output (', rawUnit, ')'])
        %hold on
        %x = [0: 1/nnfs: endTime - startTime];
        %plot(x, (binProbs-.5)*150,'m'), legend('C3', 'C4', 'Spindles')
        
    end % if hasProbs
    
end


