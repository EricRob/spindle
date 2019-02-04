% generate binary file for the input of RNN 
% 
% Zhengdong Xiao 2017-12-29
% update : 

% NOTE: 
% 
% 
%%

h1=fopen(['SleepSpindleData4RNN/Augment_data.txt'],'rt');
h2=fopen(['SleepSpindleData4RNN/Augment_labels.txt'],'rt');

C1=textscan(h1,'%f');
C2=textscan(h2,'%f');
C = [C1{1},C2{1}];

fclose(h1);
fclose(h2);

f=fopen('Augment1.bin', 'a+');
for i=1:size(C,1)
    fwrite(f,C(i,1),'float');
    fwrite(f,C(i,2),'int8');
end
fclose(f);