function out=bandpass(in, type,fr, fs)
    if nargin <  4
        fs = 200;
    end
    n=3;
%     fr=[9,16];
    switch type
        case 'bandpass'
            [b,a]=butter(n,fr/(fs/2),'bandpass');
        case 'bandstop'
            [b,a]=butter(n,fr/(fs/2),'stop');
        otherwise
            disp(['Input Type Error!']);
    end
%     [b,a]=butter(n,11/(fs/2),'low');
    
    out=filtfilt(b,a,in);
end


