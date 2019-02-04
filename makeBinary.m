function [binaryProbs] = makeBinary(probs, prob_thresh,  time_thresh)
    predict = ( probs > prob_thresh ); % Predict now stores indices of places where the probability is high
    ind = find(diff(predict)==1);
    ind = ind+1;
    ind1 = find(diff(predict)==-1);
    if predict(1)==1
        ind = [1; ind];
    end
    if predict(end)==1
        ind1 = [ind1; length(predict)];
    end
    smooth1 = predict;
    for i= 1:length(ind)
        if sum(predict(ind(i):ind1(i))) <= time_thresh
            smooth1(ind(i):ind1(i)) = 0;
        end
    end
    
    %%
    num_step = 100; % 200;
    detect = find(diff(smooth1)==1)+1;
    detect_end = find(diff(smooth1)==-1);
    if smooth1(1)==1
        detect = [1;detect];
    end
    if smooth1(end)==1
        detect_end = [detect_end;length(smooth1)];
    end
    for t=1:size(detect_end)
        ind = find(detect > detect_end(t) & detect <= detect_end(t) + num_step, 1);
        if ~isempty(ind)
            smooth1(detect_end(t):detect(ind(end)))=1;
        end
    end
    detect = find(diff(smooth1)==1)+1;
    if smooth1(1)==1
        detect = [1;detect];
    end
    detect_end = find(diff(smooth1)==-1);
    if smooth1(end)==1
        detect_end = [detect_end;length(smooth1)];
    end
    
    
    binaryProbs = smooth1;
end