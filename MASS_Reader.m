% �uthor:  Zhengdong Xiao, 
% 7th-March-2018
%
% Decription:
%           Reader for Mass Spindle Dataset
%
% Input:
%           session - name of the subject
%           verbose - whether to show figures of gap
% Output:
%           Annotation_and - combine the two expert annotations using AND criteria 
%           Annotation_or - combine the two expert annotations using OR criteria
%           Annotation_soft - combine the two expert annotations using SOFT          
%           EEG - raw EEG recordings(unit: microvolt)
%           stages - sleep stage for EEG 
%           gap - shows the differents of the spindle onset caused by two experts
%           target_f - the target resampling frequnecy
%           fs - actual resampling frequnecy

function [Annotation_and, Annotation_or, Annotation_soft, EEG,  stages, gap, target_f, fs, expert] = MASS_Reader(session, verbose, target_f)
    %% load data
    
    if nargin <  3
        target_f = 200;
    end
    
    data_path = ['SleepSpindleData4RNN/MASS_C1_SS2/version2015/' session ' PSG.edf'];
    [hdr,record]=ReadEDF2(data_path);
    IndexC = strfind(hdr.labels,'EEG C3-CLE');
    Index = find(not(cellfun('isempty', IndexC)));
    fs=hdr.frequency(Index(1));
    EEG=record{Index};
    % resample to 200 Hz
    if fs~=target_f
        EEG=resample(record{Index}, target_f, floor(fs));
        fs=fs*target_f/floor(fs);
    else
        EEG=record{Index};
    end
    
    %% load annotations
    label_path_E1 = ['SleepSpindleData4RNN/MASS_C1_SS2/annotations/MASS-C1-SS2-SpindleE1-EDF/' session ' SpindleE1.edf'];
    label_path_E2 = ['SleepSpindleData4RNN/MASS_C1_SS2/annotations/MASS-C1-SS2-SpindleE2-EDF/' session ' SpindleE2.edf'];
    stage_path = ['SleepSpindleData4RNN/MASS_C1_SS2/annotations/MASS-C1-SS2-Base-EDF/' session ' base.edf'];
    [hdr_stg, record_stg]=ReadEDF2(stage_path);
    stages = zeros(size(EEG));
        
    for ss=1:length(hdr_stg.annotation.event)
        event = hdr_stg.annotation.event(ss);
        starttime = hdr_stg.annotation.starttime(ss);
        endtime = starttime+hdr_stg.annotation.duration(ss);
        stage_range = floor(starttime*fs:endtime*fs);
        if strcmp(event, 'Sleep stage W')
            stages(stage_range)=0;
        elseif strcmp(event, 'Sleep stage R')
            stages(stage_range)=5;
        elseif strcmp(event, 'Sleep stage 1')
            stages(stage_range)=1;
        elseif strcmp(event, 'Sleep stage 2')
            stages(stage_range)=2;
        elseif strcmp(event, 'Sleep stage 3')
            stages(stage_range)=3;
        elseif strcmp(event, 'Sleep stage 4')
            stages(stage_range)=4;
        end
    end
    
    [hdr_Anno,record_Anno]=ReadEDF2(label_path_E1);
    Annotation_E1 = [hdr_Anno.annotation.starttime, hdr_Anno.annotation.duration];
    % remove spindle duration less than 0.5s
%     Annotation(Annotation(:,2)<0.5,:)=[];
    tmp = find(diff(Annotation_E1(:,1)) < 1);
    if ~isempty(tmp)
        j=[];
        for i=1:length(tmp)
          if Annotation_E1(tmp(i),1)+Annotation_E1(tmp(i),2)>Annotation_E1(tmp(i)+1,1)
              Annotation_E1(tmp(i),2) = max(Annotation_E1(tmp(i),1)+Annotation_E1(tmp(i),2), Annotation_E1(tmp(i)+1,1)+Annotation_E1(tmp(i)+1,2))-Annotation_E1(tmp(i),1);
          else
              j=[j,i];
          end
        end
        tmp(j)=[];
        Annotation_E1(tmp+1,:)=[];
    end
    Annotation_and = Annotation_E1;  
    Annotation_or = Annotation_E1;  
    Annotation_soft = Annotation_E1;
    gap = [];
    if exist(label_path_E2, 'file') == 2
        [hdr_Anno_E2,record_Anno_E2]=ReadEDF2(label_path_E2);
        Annotation_E2 = [hdr_Anno_E2.annotation.starttime, hdr_Anno_E2.annotation.duration]; 
        % remove spindle duration less than 0.5s
%       Annotation_E2(Annotation_E2(:,2)<0.5,:)=[];
        % merge two consecutive event less than 0.5s
        tmp = find(diff(Annotation_E2(:,1)) < 1);
        if ~isempty(tmp)
            j=[];
            for i=1:length(tmp)
                if Annotation_E2(tmp(i),1)+Annotation_E2(tmp(i),2)>Annotation_E2(tmp(i)+1,1)
                  Annotation_E2(tmp(i),2) = max(Annotation_E2(tmp(i),1)+Annotation_E2(tmp(i),2), Annotation_E2(tmp(i)+1,1)+Annotation_E2(tmp(i)+1,2))-Annotation_E2(tmp(i),1);
                else
                  j=[j,i];
                end
            end
            tmp(j)=[];
            Annotation_E2(tmp+1,:)=[];
        end
        new_Anno = [Annotation_E1; Annotation_E2];
        [rank, I] = sort(new_Anno(:,1));
        new_Anno = new_Anno(I,:);
        % remove onset less than 1 secs
        ind = find(diff(new_Anno(:,1)) < 0.5);
        gap = zeros(length(ind),2);
        for i=1:length(ind)
            if I(ind(i))<=size(Annotation_E1,1)
                gap(i,1) = Annotation_E1(I(ind(i)), 1) - Annotation_E2(I(ind(i)+1)-size(Annotation_E1,1), 1);
                gap(i,2) = Annotation_E2(I(ind(i)+1)-size(Annotation_E1,1), 2);
            else
                gap(i,1) = Annotation_E1(I(ind(i)+1), 1) - Annotation_E2(I(ind(i))-size(Annotation_E1,1), 1);
                gap(i,2) = Annotation_E2(I(ind(i))-size(Annotation_E1,1), 2);
            end
        end
        if verbose
            figure;subplot(211);histogram(gap(:,1));legend('expert1-expert2');
            subplot(212);scatter(gap(:,1),gap(:,2));ylabel('duration');
            disp(['num-E1 :  '  num2str(length(Annotation_E1))  '  num-E2 :  ' num2str(length(Annotation_E2))]);
            disp(['median:  ' num2str(median(gap(:,1))) '  mean:  ' num2str(mean(gap(:,1)))  '  std:  '  num2str(std(gap(:,1))) '  Corr:  '  num2str(corr(gap(:,1),gap(:,2)))]);
        end
        new_Anno_or = new_Anno;
        
        for i=1:length(ind)
            new_Anno(ind(i)+1,2) = min(new_Anno(ind(i),1)+new_Anno(ind(i),2), new_Anno(ind(i)+1,1)+new_Anno(ind(i)+1,2)) - new_Anno(ind(i)+1,1);
        end
        Annotation_and = new_Anno(ind+1,:);   %use the AND criteria
        
        for i=1:length(ind)
            new_Anno_or(ind(i),2) = max(new_Anno_or(ind(i),1)+new_Anno_or(ind(i),2), new_Anno_or(ind(i)+1,1)+new_Anno_or(ind(i)+1,2)) - new_Anno_or(ind(i),1);
        end
        Annotation_soft = new_Anno_or(ind,:);  %use the SOFT criteria
        new_Anno_or(ind+1,:) = [];     %use the OR criteria
        Annotation_or = new_Anno_or;
    end

    start = Annotation_or(:,1);
    expert = ones(size(start));
    if exist(label_path_E2, 'file') == 2
        for m=1:length(start)
            tmp1=find(abs(Annotation_E1-start(m))<0.5, 1);
            tmp2=find(abs(Annotation_E2-start(m))<0.5, 1);
            if ~isempty(tmp1) && ~isempty(tmp2)
                expert(m)=3;   % annotated by expert 1 and expert 2 
            elseif ~isempty(tmp1)
                expert(m)=1;   % only annotated by expert 1
            elseif ~isempty(tmp2) 
                expert(m)=2;   % only annotated by expert 2
            end
        end
    end
    Annotation_or = [Annotation_or, expert];

end