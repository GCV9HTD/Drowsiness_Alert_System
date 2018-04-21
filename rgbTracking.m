function center=rgbTracking(Image,center,w_halfsize,q_u,redBins,greenBins,blueBins,minDist,maxIterNum,incre)
% ********************************************************************
% function: mean shift tracking 
% parameters:
%       Image         current frame to track
%       center        initial location of target
%       w_halfsizet   bandwidth of target window
%       q_u           target model
%       redBins,greenBins,blueBins: quantification scheme of rgb space
%       minDist       convergence condition
%       maxIterNum    maxmimal iterative number
%       incre         increase the candidate winodw to robust tracking
% return:
%       center        center of tracking-result winodw
% ****************************************************************

sum_p=0; 
histo=zeros(redBins,greenBins,blueBins);      %  initialize the target histogram

iterations=0;                                 %  initialize iteration number
center_old=center;

rmin=center(1)-w_halfsize(1)-incre;
rmax=center(1)+w_halfsize(1)+incre;
cmin=center(2)-w_halfsize(2)-incre;
cmax=center(2)+w_halfsize(2)+incre;

height=size(Image,1);
width=size(Image,2);
  
while 1
    wmax=(rmin-center(1)).^2+(cmin-center(2)).^2+1;   
    for i=rmin:rmax         % calculate the candidate model
        for j=cmin:cmax
            if (i>=1 & i<=height & j>=1 & j<=width)
                d=(i-center(1)).^2+(j-center(2)).^2;
                w=wmax-d;  % weight of pixel (i,j) 
                R=floor(Image(i,j,1)/(256/redBins))+1;
                G=floor(Image(i,j,2)/(256/greenBins))+1;
                B=floor(Image(i,j,3)/(256/blueBins))+1;
                histo(R,G,B)=histo(R,G,B)+w;
            end
        end
    end

    % convert histo matrix into 1D vector
    for i=1:redBins
        for j=1:greenBins
            for k=1:blueBins
                index=(i-1)*greenBins*blueBins+(j-1)*blueBins+k;
                p_u(index)=histo(i,j,k);
                sum_p=sum_p+p_u(index);
            end
        end
    end
    % normalize
    p_u=p_u/sum_p;   

    n=1;
    % compute the weight value of each pixel in the candidate window
    for i=rmin:rmax   
        for j=cmin:cmax
            if (i>=1 & i<=height & j>=1 & j<=width)  
                R=floor(Image(i,j,1)/(256/redBins))+1;
                G=floor(Image(i,j,2)/(256/greenBins))+1;
                B=floor(Image(i,j,3)/(256/blueBins))+1;
                u=(R-1)*greenBins*blueBins+(G-1)*blueBins+B;
                x(1,n)=i;
                x(2,n)=j;                
                w_i(n)=sqrt(q_u(u)/p_u(u));                                                                
                n=n+1;
            end
        end
    end

    center_r=(x*w_i'/sum(w_i))';               % new center    
    MS=sqrt(sum((center_r-center_old).^2));    % normal of mean shift vector  
    iterations=iterations+1;                   % count iteration number
    
    % Does mean shift algorithm converge?
    if (MS<minDist | iterations>=maxIterNum)
        break;
    end
    
    center_old=center_r;                       % save tracking result      
    center=floor(center_r);                    % 

    % reInitialize the candidate winodw
    rmin=center(1)-w_halfsize(1)-incre;        
    rmax=center(1)+w_halfsize(1)+incre;
    cmin=center(2)-w_halfsize(2)-incre;
    cmax=center(2)+w_halfsize(2)+incre;
    
    histo=zeros(redBins,greenBins,blueBins);   % reInitialize the target histogram
    sum_p=0;         
end