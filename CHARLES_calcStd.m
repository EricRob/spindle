numSpindles = 3;
std_value = zeros(numSpindles, 1);
for i=1:numSpindles
    s_data = data(floor(Spindles(i,1)*fs):floor(Spindles(i,2)*fs));
    s_data = detrend(s_data);
    std_value(i) = std(s_data);
end
pd = fitdist(std_value,'Normal');
meanStd = mean(pd);