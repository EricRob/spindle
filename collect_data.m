% Àuthor:  Zhengdong Xiao, 
% 7th-March-2018
%
% Decription:
%           Appending data to cell
%
% Input:
%           Annotation_and - combine the two expert annotations using AND criteria 
%           Annotation_or - combine the two expert annotations using OR criteria
%           Annotation_soft - combine the two expert annotations using SOFT
%           EEG - raw EEG recordings(unit: microvolt)
%           database - A cell include entire multiple EEG raw recordings 
%           output - A cell  include entire multiple EEG spindle annotations
%           Spindles - A cell include entire spindle statistics(onset & duration) 
%           idx - index of current subject
%           total_time - accumulate time of EEG 
%           fs - sampling rate
% Output:
%           database - A cell include entire multiple EEG raw recordings 
%           output - A cell  include entire multiple EEG spindle annotations
%           Spindles - A cell include entire spindle statistics(onset & duration) 
%           total_time - accumulate time of EEG
function  [database,output,Spindles,total_time]=collect_data(Annotation_and, Annotation_or, Annotation_soft, EEG, database, output, Spindles, idx, total_time, fs)   
        
        database = {[database{1}; EEG]};
        labels = zeros(length(EEG), 3);
        total_time_1 = sum(total_time);
        total_time(idx) = length(EEG)/fs;
        total_time_2 = sum(total_time);
        for k=1:2
            if k==1
                Annotation = Annotation_and;
            else
                Annotation = Annotation_or;
            end
            
            for j=1:size(Annotation,1)
                labels(floor(Annotation(j,1)*fs) : floor((Annotation(j,1)+Annotation(j,2))*fs), k) = 1;
            end

            output{1, k} = [output{1, k}; labels(:,k)];

            Annotation(:,1) = Annotation(:,1) + total_time_1;
            % I don't care for spindles in the beginning and end of the file!!!!    
            if Annotation(1, 1) < sum(total_time_1)+2.25
               Annotation = Annotation(2:end, :);
            end           
            if Annotation(end, 1) > sum(total_time_2)-2.25
               Annotation = Annotation(1:end-1, :);
            end    
            Spindles{k,1}=[Spindles{k,1};Annotation(:,1)];
            Spindles{k,2}=[Spindles{k,2};Annotation(:,2)];
        end
        
        % soft labels
        for j=1:size(Annotation_soft,1)
                labels(floor(Annotation_soft(j,1)*fs) : floor((Annotation_soft(j,1)+Annotation_soft(j,2))*fs), 3) = 1;
        end

        tmp = labels(:,3)-labels(:,1);
        onset = find(diff(tmp)==1)+1;
        offset = find(diff(tmp)==-1);
        if ~isempty(onset)
            for k=1:length(onset)
                r1 = 0.5/(offset(k)-onset(k)+2);
                if  labels(onset(k)-1, 3)==0
                    labels(onset(k):offset(k), 3)=[0.5+r1:r1:1-r1]';
                else
                    labels(onset(k):offset(k), 3)=[1-r1:-r1:r1+0.5]';
                end
            end
        end
        output{1, 3} = [output{1, 3}; labels(:, 3)];
 end