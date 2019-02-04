[hdr,record]=edfread('DatabaseSpindles/excerpt1.edf');
EEG=record(7,:);
EEG2=record(1,:);
EEG3=record(8,:);
out=bandpass(EEG);
out2=bandpass(EEG2);
out3=bandpass(EEG3);
datarange=[42000:43000];
figure;
subplot(311);
plot((0:0.01:floor(length(datarange)/100)),EEG(datarange));
title('Original EEG segment');
ylabel('Amplitude(uV)')
subplot(312);
plot((0:0.01:floor(length(datarange)/100)),EEG2(datarange));
title('Original EEG segment');
ylabel('Amplitude(uV)')
subplot(313);
plot((0:0.01:floor(length(datarange)/100)),EEG3(datarange));
title('Original EEG segment');
ylabel('Amplitude(uV)')
% subplot(212);
% envelope(out(datarange),1e2,'analytic');
% xlim([0,length(datarange)]);
% title('Band-passed EEG segment zero phase');
% ylabel('Amplitude (uV)')
% xlabel('Time (s)')