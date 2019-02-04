function  [f,P1,fmax]=cal_fft(x,fs)
      y=fft(x);
      L=length(x);
      P2=abs(y/L);
      P1 = P2(1:floor(L/2)+1);
      P1(2:end-1) = 2*P1(2:end-1);
      f = fs*(0:(L/2))/L;
      ind=find(f>8 & f<17);
      [~,I]=max(P1(ind));
      fmax=f(ind(I));
%       fmax=mean(f(ind)) ;
      if isempty(fmax)
          disp(['HaHa']);
      end
      %disp(['Spindle frequency:  ' num2str(fmax)]);
end