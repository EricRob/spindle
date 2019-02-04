% �uthor:  Zhengdong Xiao, 
% 7th-March-2018
%
% Decription:
%           function of  augment spindles
%
% Input:
%           spindles - spindle annotations
%           h1 - header of data file
%           h2 - header of label file
%           num_steps - number of samples in one spindle example
%           database - entire EEG data
%           output - entire labels
%           fs - sampling rate
%           verbose-flag to show number of examples
% Output:
%           None

function augment_data(spindles, h1, h2, num_steps, database, output, fs, verbose, format1, format2,factor)
    
    if nargin<11
       factor = 4.5;
    end
    P = 0;N=0;
    for m=1:size(spindles, 1)
         data=cell(1,1);
         label=data;
         center = floor(fs*(spindles(m, 1) + spindles(m, 2)/2 ));
         start_ = floor(center - num_steps/2 -floor(num_steps*factor)); % 7/11/18 - Eric changed to floor()
         end_ = floor(center + num_steps/2+floor(num_steps*factor)-1); % 7/11/18 - Eric changed to floor()
         new_sequence = start_:end_;
         seq = database{1}(new_sequence, :);
         labels = output(new_sequence, :);
         for j=1:size(seq, 1)-num_steps+1
             data = {[data{1}; seq(j:j+num_steps-1, :)]};
             label = {[label{1}; labels(j:j+num_steps-1, :)]};
         end
       
         for n=1:size(data{1},1)/num_steps
             data{1}((n-1)*num_steps+1:n*num_steps, 1)=detrend(data{1}((n-1)*num_steps+1:n*num_steps, 1));
         end
         if verbose
             label_=reshape(label{1},[num_steps,length(label{1})/num_steps]);
             tmp2=sum(label_, 1);
             P=P+length(find(tmp2>num_steps/2));
             N=N+length(tmp2)-length(find(tmp2>num_steps/2));
         else
             fprintf(h1,format1,data{1}');
             fprintf(h2,format2,label{1});
          end
     end
     if verbose
         disp(['Positive examples:   ' num2str(P) '   Negative examples:   ' num2str(N)]);
     end
end