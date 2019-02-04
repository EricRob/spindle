test =  smooth1(floor(range1*fs)-49);
spindles = [];
for i=1:length(test)-1
    if test(i+1)>test(i)
        spindles = [spindles i/fs/60];
    end
end
length(spindles)
histogram(spindles,round((length(range1)/fs/60)))