clear all;
close all;
clc;

warning('off')

[fn,pn] = uigetfile({'*.*';},'The file name should  have avi extension.');

% M = aviread(strcat([pn,fn]));

videoFileReader = vision.VideoFileReader(strcat(pn,fn));

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% select tracking algorithm
minDist=0.1;        % the minimum distance between two mean shift iteration
maxIterNum=15;      % maximal iteration number
incre=5;            % enlarge the search region

% color quantification
redBins=8; greenBins=8; blueBins=8;

% lbp threshold
lbpThreshold=8;

% set double buffer
set(gcf,'DoubleBuffer','on');
tic

flg=1;
k=0;

while(flg==1)
videoFrame      = step(videoFileReader);
bbox            = step(faceDetector, videoFrame);
k=k+1;

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox,'Face Region');
figure(1), imshow(videoOut), title('Detected face');

if(size(bbox,1)==1)
eyesDetector = vision.CascadeObjectDetector('EyePairBig');
faceImage    = imcrop(videoFrame,bbox(1,:));
eyesBBox     = step(eyesDetector,faceImage);

if(bbox(1,3)<200 && bbox(4)<240)
    disp('here..')
    continue;
end
    
else
    continue;
end

% The nose bounding box is defined relative to the cropped face image.
if size(eyesBBox,1)==1
eyesBBox(1,1:2) = eyesBBox(1,1:2) + bbox(1,1:2);
flg=0;

figure(1)
hold on
rectangle('Position',eyesBBox(1,:),'LineWidth',2,'EdgeColor',[0 1 0])

end
end
disp(['Frames required to detect face= ' num2str(k)])
disp(['Time required to detect face = ' num2str(toc) ' Seconds'])

frame00=videoFrame;                    % get start-frame to select to tracking object

height=size(frame00,1);
width=size(frame00,2);

% select Face tracking window automatically
%[ cmin, cmax, rmin, rmax ] = select( frame00);

cmin1 = round(bbox(1,1));
cmax1 = round(bbox(1,1)+bbox(1,3));
rmin1 = round(bbox(1,2));
rmax1 = round(bbox(1,2)+bbox(1,4));
center1(1,1)=floor((rmin1+rmax1+1)/2);          % the center of window
center1(1,2)=floor((cmin1+cmax1+1)/2);          %                   
            
w_halfsize1(1) = round(abs(rmax1 - rmin1)/2);   % half height of window  
w_halfsize1(2) = round(abs(cmax1 - cmin1)/2);   % half width of window


% select Eyes tracking window automatically
%[ cmin, cmax, rmin, rmax ] = select( frame00);

cmin2 = round(eyesBBox(1,1));
cmax2 = round(eyesBBox(1,1)+eyesBBox(1,3));
rmin2 = round(eyesBBox(1,2));
rmax2 = round(eyesBBox(1,2)+eyesBBox(1,4));
center2(1,1)=floor((rmin2+rmax2+1)/2);          % the center of window
center2(1,2)=floor((cmin2+cmax2+1)/2);          %                   
            
w_halfsize2(1) = round(abs(rmax2 - rmin2)/2);   % half height of window  
w_halfsize2(2) = round(abs(cmax2 - cmin2)/2);   % half width of window

% calculate the target model
% target model with rgb histogram
    q_u1=rgbPDF(double(frame00),center1,w_halfsize1,redBins,greenBins,blueBins);
    q_u2=rgbPDF(double(frame00),center2,w_halfsize2,redBins,greenBins,blueBins);

% set the domain to tracking
startFrm=k;                                  % 

disp('Tracking...')
tic
i=startFrm;

while ~isDone(videoFileReader)
                   
    if i==startFrm                       
        framei=frame00;
        
        figure(2)
        subplot(2,2,1)
        imshow(framei)
        title('Frame 1')
        
        subplot(2,2,2)
        imhist(framei(:,:,1))
        title('R-plane histogram')
        
        subplot(2,2,3)
        imhist(framei(:,:,2))
        title('G-plane histogram')
        
        subplot(2,2,4)
        imhist(framei(:,:,1))
        title('B-plane histogram')
        
        itrack_part=framei(rmin1:rmax1,cmin1:cmax1,:);
        i=i+1;
        
        pause(2)
        
    elseif i>startFrm  % 
        % get the current frame
        framei=step(videoFileReader);
                             % stand mean shift tracking algorithm
        center1=rgbTracking(double(framei),center1,w_halfsize1,q_u1,redBins,greenBins,blueBins,minDist,maxIterNum,incre);        
        center2=rgbTracking(double(framei),center2,w_halfsize2,q_u2,redBins,greenBins,blueBins,minDist,maxIterNum,incre);
        k=k+1;
    end
    
    % window corresponding to Face tracking result
    rmin1=center1(1)-w_halfsize1(1);             
    rmax1=center1(1)+w_halfsize1(1);            
    cmin1=center1(2)-w_halfsize1(2);           
    cmax1=center1(2)+w_halfsize1(2);           
    % 
    [rmin1,rmax1,cmin1,cmax1]=normWindow(rmin1,rmax1,cmin1,cmax1,height,width);
    
    % making tracking result
    trackim=framei;                            
    for r= rmin1:rmax1
        trackim(r, cmin1-1:cmin1,:) = 255;       
        trackim(r, cmax1:cmax1+1,:) = 255;        
    end
    for c= cmin1:cmax1
        trackim(rmin1-1:rmin1, c,:) = 255;
        trackim(rmax1:rmax1+1, c,:) = 255;        
    end
        
    % window corresponding to Eyes tracking result
    rmin2=center2(1)-w_halfsize2(1);             
    rmax2=center2(1)+w_halfsize2(1);            
    cmin2=center2(2)-w_halfsize2(2);           
    cmax2=center2(2)+w_halfsize2(2);           
    % 
    [rmin2,rmax2,cmin2,cmax2]=normWindow(rmin2,rmax2,cmin2,cmax2,height,width);
    
    % making tracking result                          
    for r= rmin2:rmax2
        trackim(r, cmin2-1:cmin2,3) = 255;       
        trackim(r, cmax2:cmax2+1,3) = 255;        
    end
    for c= cmin2:cmax2
        trackim(rmin2-1:rmin2, c,3) = 255;
        trackim(rmax2:rmax2+1, c,3) = 255;        
    end 
    
    %display tracking result
    figure(3)
    imshow(trackim);
    title(['Tracking Frame no. ' num2str(k)]);
    drawnow;
    imwrite(trackim,['.\Results\Algorithm2\' num2str(k) ,'.jpg']);
    
end

disp(['Total number of frames = ' num2str(k)])
disp(['Total time required  = ' num2str(toc) ' Seconds'])

% Clean up
release(videoFileReader);


    
    