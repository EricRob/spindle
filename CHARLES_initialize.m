fileNum = '300001';
whichSet = 'chat';

hasProbs = 1;
[hdr, records] = quickRead(fileNum, whichSet);
window = 30;

if hasProbs == 1
    fs = 200; % Downsampled to 200

    probFile = fopen('probability_chat_1_C3.txt','rt');

    probRecord = zeros(7200*fs, 1);
    i = 1;
    while true
        line = fgetl(probFile);
        if ~ischar(line)
          break; 
        end  %end of file

        probRecord(i) = str2double(line);
        i = i + 1;
    end % while
end


if lower(whichSet) == 'mros'
    fs = 256;
elseif lower(whichSet) == 'chat'
    fs = 512;
end

format compact;