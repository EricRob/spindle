% Àuthor:  Zhengdong Xiao, 
% 30th-May-2018
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

function [Annotation_and, Annotation_or, Annotation_soft, EEG,  stages, gap, target_f, fs, expert] = MASS_Reader(session, verbose)
    %% load data
    data_path = ['DatabaseSpindles/' session '.edf'];
    [hdr,record]=edfread(data_path);
    IndexC=strfind(hdr.label,'CZA1');
    Index = find(not(cellfun('isempty', IndexC)));
    if isempty(Index)
        IndexC=strfind(hdr.label,'C3A1');
        Index = find(not(cellfun('isempty', IndexC)));
    end
    EEG=record(Index,:)';
    sample_rate=hdr.frequency(Index);
    target_f=200;
    % resample to 200 Hz
    if sample_rate~=target_f
        EEG=interp(EEG,target_f/sample_rate);
        sample_rate=target_f;
    end
    
    
    %% load annotations
    f=fopen(['SleepSpindleData4RNN/' session '_label.txt'],'r');
    C=textscan(f,'%f %f');
    fclose(f);
    fs=sample_rate;
    stages = ones(size(EEG))+1;
    
    Annotation_E1 = [C{1}, C{2}];
    %%
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

    start = Annotation_or(:,1);
    expert = ones(size(start));
   
    Annotation_or = [Annotation_or, expert];

end