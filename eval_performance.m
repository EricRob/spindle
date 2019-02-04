function [sens, spec, fdr, f1score, result, fp_result, smooth1, eval, probabilty]=eval_performance(prob, fs, tt, DB_spindle, DB_BB, stages, Annotation, prob_th, time_th, num_step, verbose)
    predict=(prob>prob_th);
    ind = find(diff(predict)==1);
    ind = ind+1;
    ind1 = find(diff(predict)==-1);
    if predict(1)==1
        ind = [1;ind];
    end
    if predict(end)==1
        ind1 = [ind1;length(predict)];
    end
    thresh = time_th;
    smooth1 = predict;
    for i=1:length(ind)
        if sum(predict(ind(i):ind1(i))) <= thresh
            smooth1(ind(i):ind1(i)) = 0;
        end
    end
    
%     soft_ind=find(output{1,3} ~=1 & output{1,3} ~=0);
%     figure;plot(prob(soft_ind),output{1,3}(soft_ind),'.','markersize',2);
%     figure;histogram(output{1,3}(soft_ind));
    % evaluate latency
    result = cell(1,8);
    
    detect = find(diff(smooth1)==1)+1;
    detect_end = find(diff(smooth1)==-1);
    if smooth1(1)==1
        detect = [1;detect];
    end
    if smooth1(end)==1
        detect_end = [detect_end;length(smooth1)];
    end
    for t=1:size(detect_end)
        ind = find(detect>detect_end(t)&detect<=detect_end(t)+num_step, 1);
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
    %remove too close detect event
%     ind_ = find(diff(detect)<fs*1)+1;
%     detect(ind_)=[];
%     detect_end(ind_)=[];
    
    num_spindles = size(Annotation,1);
    start = Annotation(:,1);
    duration = Annotation(:,2);
    probabilty = zeros(size(start));
    for k=1:length(start)
        probabilty(k)=median(prob(floor(start(k)*fs-num_step+1):floor((start(k)+duration(k))*fs)-num_step+1));
    end
    t_event = 0;
        
    ii=[];
    det_thresh=0.5;
    detect_ = [detect,detect_end];
    for k=1:num_spindles
        interval = .25;
        index = find(((detect_(:,1)+num_step-1)>(start(k))*fs & (detect_(:,1)+num_step-1)<(start(k)+duration(k)+interval)*fs) | (detect_(:,2)>(start(k))*fs & detect_(:,2)<(start(k)+duration(k)+interval)*fs) | ((detect_(:,1)+num_step-1)<(start(k))*fs & (detect_(:,2)+num_step-1)>(start(k)+duration(k))*fs));
        
%         index1=find((detect_end+49)>(start(k)+duration(k)-det_thresh)*fs & (detect_end+49)<(start(k)+duration(k)+det_thresh)*fs);
%         index=unique([index;index1]);
%         index=intersect(index,index1);
        if ~isempty(index)
            [~, I]= min(abs((detect(index)+num_step-1)/fs-start(k)));
            index2=index(I);
            onset = (detect(index2)+num_step-1)/fs;
            offset = (detect_end(index2)+num_step-1)/fs;
            %[~,~,fmax]=cal_fft(EEG(floor(fs*start(k)):floor(start(k)*fs+duration(k)*fs)),fs);            
            t_event=t_event+1;
            result{1} = [result{1};start(k)];  %groud truth onset
            result{2} = [result{2};onset];     %algo onset
            result{3} = [result{3};onset-start(k)];  %algo latency
            result{4} = [result{4};offset-onset]; %algo duration
            result{5} = [result{5};DB_spindle(floor((start(k)+duration(k)/2-tt(1))*fs))]; %spindle power
            result{6} = [result{6};DB_spindle(floor((start(k)+duration(k)/2-tt(1))*fs))/DB_BB(floor((start(k)+duration(k)/2-tt(1))*fs))]; %power ratio
            result{7} = [result{7};prob(floor((start(k)+duration(k)/2)*fs)-num_step+1)];  
            %result{8} = [result{8};fmax];
            ii = [ii;index];
        else
            result{1} = [result{1};start(k)];
            result{2} = [result{2};999];
            result{3} = [result{3};999];
            result{4} = [result{4};999];
            result{5} = [result{5};DB_spindle(floor((start(k)+duration(k)/2-tt(1))*fs))];
            result{6} = [result{6};DB_spindle(floor((start(k)+duration(k)/2-tt(1))*fs))/DB_BB(floor((start(k)+duration(k)/2-tt(1))*fs))];
            result{7} = [result{7};prob(floor((start(k)+duration(k)/2)*fs)-num_step+1)];
            %result{8} = [result{8};fmax];
        end

    end
        
    latency = result{3};
    tp = result{6};
    fn = find(latency==999);
    tp(fn) = [];
    latency(fn)=[];
    du = start;
    du(fn)=[];
    if verbose
        disp(['Mean latency: ' num2str(mean(latency)) '  Median latency: ' num2str(median(latency)) '   Std: ' num2str(std(latency))  '  Max: ' num2str(max(latency))  '  Min: ' num2str(min(latency))]);
    end
    f_event = detect;
    f_event_end = detect_end;
    f_event(ii)=[];
    f_event_end(ii)=[];
    fp_result=cell(1,5);
    % only detect N2 stage
    N2_lens = length(find(stages(51:end)==2));
    num_fevent = 0;
    for k=1:length(f_event)
        if stages(f_event(k)+num_step-1)==2
             %[~,~,fmax]=cal_fft(EEG(floor(f_event(k)+49):floor(f_event_end(k)+49)),fs);
            num_fevent=num_fevent+1;
            begin = (f_event(k)+num_step-1)/fs;
            fp_result{1} = [fp_result{1};begin];  %onset
            fp_result{2} = [fp_result{2};(f_event_end(k)+1-f_event(k))/fs];  %duration
            if floor((f_event(k)+f_event_end(k))/2+num_step-1)>tt(1)*fs
                fp_result{3} = [fp_result{3};DB_spindle(floor((f_event(k)+f_event_end(k))/2+num_step-1-tt(1)*fs))]; %power
                fp_result{4} = [fp_result{4}; DB_spindle(floor((f_event(k)+f_event_end(k))/2+num_step-1-tt(1)*fs)) / DB_BB(floor((f_event(k)+f_event_end(k))/2+num_step-1-tt(1)*fs))];
            else
                fp_result{3} = [fp_result{3};DB_spindle(1)];
                fp_result{4} = [fp_result{4};DB_spindle(1)/DB_BB(1)];
            end
            
            %fp_result{5} = [fp_result{5};fmax];
        end
    end
    relative_ratio=-1e10;
    tn = floor(N2_lens/fs) - num_spindles - (num_fevent-length(find(fp_result{4}<relative_ratio)));
    sens = (t_event-length(find(tp<relative_ratio)))/num_spindles;
    spec = tn/(floor(N2_lens/fs) - num_spindles);
    fdr = (num_fevent - length(find(fp_result{4} < relative_ratio))) / ((num_fevent - length(find(fp_result{4} < relative_ratio))) + (t_event - length(find(tp < relative_ratio))));
    precision = (t_event-length(find(tp<relative_ratio)))/(t_event-length(find(tp<relative_ratio))+num_fevent-length(find(fp_result{4}<relative_ratio)));
    f1score = 2*precision*sens/(precision+sens);
    eval.TP = t_event-length(find(tp<relative_ratio));
    eval.FP = num_fevent-length(find(fp_result{4}<relative_ratio));
    eval.FN = num_spindles-eval.TP;
    eval.TN = tn;
    
    if verbose
        disp(['False event rate: ' num2str((num_fevent-length(find(fp_result{4}<relative_ratio)))/(N2_lens/(fs*60))) '  (events/min)'  '  Spindle Density: ' num2str(num_spindles/(N2_lens/(fs*60))) '  (events/min)']);
        disp(['True event rate: ' num2str((t_event-length(find(tp<relative_ratio)))/(N2_lens/(fs*60)))  '(events/min)   Total spindles:  '  num2str(num_spindles) ]);  
        disp(['Sensitivity:  ' num2str(sens) '   Specificity:  '  num2str(spec) '   FDR:  '  num2str(fdr) '  F-1 score:  '  num2str(f1score)]);
        disp(['Estimated True Negative:  ' num2str(tn) ' events']);
        disp(['=============================================================']);
        % Added by Eric for copy/paste into spreadsheets
        disp([num2str(mean(latency)) ' ' num2str(median(latency)) ' ' num2str(std(latency)) ' ' num2str((num_fevent-length(find(fp_result{4}<relative_ratio)))/(N2_lens/(fs*60))) ' ' num2str((t_event-length(find(tp<relative_ratio)))/(N2_lens/(fs*60))) ' ' num2str(num_spindles/(N2_lens/(fs*60))) ' ' num2str(sens) ' ' num2str(spec) ' ' num2str(fdr) ' ' num2str(f1score) ' ' num2str(eval.TP) ' ' num2str(eval.FP) ' ' num2str(eval.FN) ' ' num2str(eval.TN)]);
    end
    
    eval.TP = t_event-length(find(tp<relative_ratio));
    eval.FP = num_fevent-length(find(fp_result{4}<relative_ratio));
    eval.FN = num_spindles-eval.TP;
    eval.TN = tn;
    
   
end