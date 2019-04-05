probFile = fopen('probability_MrOS_18_C3.txt','rt');

probRecord = [];
i = 1;
while true
  line = fgetl(probFile);
  if ~ischar(line)
      break; 
  end  %end of file
  
  probRecord(i) = str2double(line);
  i = i + 1;
end % while

fs = 200;

x = [0: 1/fs: 1200];
x = x(1:240000);
y = probRecord(:)';

plot(x, y, '-');