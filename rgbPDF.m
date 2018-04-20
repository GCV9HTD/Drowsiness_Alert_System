function q_u=rgbPDF(Image,center,w_halfsize,redBins,greenBins,blueBins)
% *************************************************************************
% function: calculate targe model
% parameters: 
%       Image: the 1th image
%       center: the center of target window
%       w_halfsize: the bandwidth of target window
%   
% returns:
%       q_u: target model
%


sum_q=0;
histo=zeros(redBins,greenBins,blueBins);  % initialize the target histogram

rmin=center(1)-w_halfsize(1);
rmax=center(1)+w_halfsize(1);
cmin=center(2)-w_halfsize(2);
cmax=center(2)+w_halfsize(2);

%
wmax=(rmin-center(1)).^2+(cmin-center(2)).^2+1;
for i=rmin:rmax                           
    for j=cmin:cmax
        d=(i-center(1)).^2+(j-center(2)).^2;
        w=wmax-d;                         % weight value corresponding to pixel(i,j) 
        R=floor(Image(i,j,1)/(256/redBins))+1;
        G=floor(Image(i,j,2)/(256/greenBins))+1;
        B=floor(Image(i,j,3)/(256/blueBins))+1;
        histo(R,G,B)=histo(R,G,B)+w;      % target histogram
    end
end

% convert histo (matrix) into 1D vector
for i=1:redBins
    for j=1:greenBins
        for k=1:blueBins
            index=(i-1)*greenBins*blueBins+(j-1)*blueBins+k;
            q_u(index)=histo(i,j,k);
            sum_q=sum_q+q_u(index);
        end
    end
end
q_u=q_u/sum_q;   % normalize the target model