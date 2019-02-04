function s1 = simulate_spindle(sample_rate,sequence_length, sub_id)

    %DREAMS params:
    %ALL:%Mean:2.0868    0.0191  -10.0934   -0.1654   83.3979   -2.5526
         %STD:2.3757    3.8199   10.9169    4.4900    5.5615   11.6703 
    mean_v = zeros(8, 6);
    std_v = zeros(8, 6);
    mean_v(1, :) = [2.2973   -0.9022   -9.3442   -0.7033   82.0857   -0.7713];
    std_v(1, :) = [1.9447    5.0398    9.6044    4.7710    5.4483    9.4527];
    mean_v(2, :) = [1.9898    0.4256   -8.5041    0.1562   81.3177   -1.8422];
    std_v(2, :) = [2.9540    2.8859    9.6588    4.2284    4.6889    8.2462];
    mean_v(3, :) = [2.0818   -0.3808   -5.3801   -0.6210   86.5599    0.1358];
    std_v(3, :) = [2.2451    1.6186    4.9715    4.0745    4.1177    6.8045];
    mean_v(4, :) = [1.8891    1.1645   -8.4880    0.3594   79.8493   -2.2071];
    std_v(4, :) = [1.0571    4.9878    8.1936    4.5987    8.3357   14.1654];
    mean_v(5, :) = [2.3559    0.1035  -10.8570   -0.0094   83.7545   -4.1233];
    std_v(5, :) = [1.4887    2.0299    8.7176    4.3228    3.5877   10.4127];
    mean_v(6, :) = [1.8546    0.1312  -12.7581   -0.1933   85.2031   -5.5187];
    std_v(6, :) = [3.3427    3.6176   14.6312    4.5082    4.0942   13.3078];
    mean_v(7, :) = [2.4305   -1.7746   -9.0078    1.7301   84.1736    1.8600];
    std_v(7, :) = [0.8813    4.9045    8.1337    3.9419    7.1361    7.9012];
    mean_v(8, :) = [1.8127    0.9877  -13.2582   -0.4378   86.1299   -2.5747];
    std_v(8, :) = [2.8144    3.2615   14.7995    4.7977    5.2614   17.8331];
            
    %a=normrnd(2.75,sqrt(0.168)); %MASS
    a = 0; % To generate spindles with amplitude normalized to 1
    %b = normrnd(0.45,sqrt(1.087)); %MASS
    b = 0; % To generate spindles with amplitude normalized to 1  
    %c=normrnd(-10,sqrt(15)); %MASS
    %c= -3;
    c = normrnd(mean_v(sub_id, 3), std_v(sub_id, 3)); %DREAMS
    %d=normrnd(0,sqrt(22.027)); %MASS
    d = normrnd(mean_v(sub_id, 4), std_v(sub_id, 4)); %DREAMS
    %e=normrnd(84.5,sqrt(14.928)); %MASS
    e = normrnd(mean_v(sub_id, 5), std_v(sub_id, 5)); %DREAMS
    %f=normrnd(-0.9,sqrt(24.456)); %MASS
    f = normrnd(mean_v(sub_id, 6), std_v(sub_id, 6)); %DREAMS
    
    fs=200;
    s=zeros(sequence_length*fs,1);
    range=-sequence_length/2+1/fs:1/fs:sequence_length/2;
    for i=1:length(range)
        t=range(i);
        s(i)=exp(a+b*t+c*t*t)*cos(d+e*t+f*t*t);
    end
    s1=resample(s,sample_rate,fs);
    % figure;
    % subplot(2,1,1);
    % plot(range,s);
    % subplot(2,1,2);
    % plot(-1.5+1/200:1/200:1.5,s1);
end