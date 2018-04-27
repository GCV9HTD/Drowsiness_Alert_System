clc
clear all
close all
warning('off')

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video file.
[fn,pn] = uigetfile({'*.*';},'The file name should  have avi extension.');
videoFileReader = vision.VideoFileReader(strcat(pn,fn));

flg=1;
k=0;
tic

%run the face detector
while(flg==1)
videoFrame      = step(videoFileReader);
bbox1            = step(faceDetector, videoFrame);
k=k+1;

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox1,'Face Region');
figure(1), imshow(videoOut), title('Detected face');

if(size(bbox1,1)==1)
eyesDetector = vision.CascadeObjectDetector('EyePairBig');
faceImage    = imcrop(videoFrame,bbox1(1,:));
eyesBBox     = step(eyesDetector,faceImage);
else
    continue;
end

% The nose bounding box is defined relative to the cropped face image.
if size(eyesBBox,1)==1
eyesBBox(1,1:2) = eyesBBox(1,1:2) + bbox1(1,1:2);
flg=0;

figure(1)
hold on
rectangle('Position',eyesBBox(1,:),'LineWidth',2,'EdgeColor',[0 0 1])

end

end
disp(['Frames required to detect face= ' num2str(k)])
disp(['Time required to detect face = ' num2str(toc) ' Seconds'])

% Convert the first box to a polygon.
% This is needed to be able to visualize the rotation of the object.
x = bbox1(1, 1);
y = bbox1(1, 2); 
w = bbox1(1, 3); 
h = bbox1(1, 4);
bboxPolygon1 = [x, y, x+w, y, x+w, y+h, x, y+h];

x1=bboxPolygon1(1,1);
y1=bboxPolygon1(1,2);
x2=bboxPolygon1(1,3);
y2=bboxPolygon1(1,4);
x3=bboxPolygon1(1,5);
y3=bboxPolygon1(1,6);
x4=bboxPolygon1(1,7);
y4=bboxPolygon1(1,8);


xn1=x1+(x4-x1)/4;
yn1=y1+(y4-y1)/4;
xn2=x2+(x3-x2)/4;
yn2=y2+(y3-y2)/4;
xn3=x2+(x3-x2)/2;
yn3=y2+(y3-y2)/2;
xn4=x1+(x4-x1)/2;
yn4=y1+(y4-y1)/2;

eyespoly=[xn1,yn1,xn2,yn2,xn3,yn3,xn4,yn4];

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon1);
videoFrame = insertShape(videoFrame, 'Polygon', eyespoly);

figure(2); imshow(videoFrame); title('Detected Face and Eyes');

% Detect feature points in the face region.
points1 = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox1);

% Display the detected points.
figure(3), imshow(videoFrame), hold on, title('Detected features');
plot(points1);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker1 = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points1 = points1.Location;
initialize(pointTracker1, points1, rgb2gray(videoFrame));

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 videoInfo.VideoSize(1:2)+30]);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints1 = points1;

contr=0;%%% Warning threshold  %%%%
ath=30; %%% Face tilt threshold in degrees 
nfth=30; %%% No. of Frames Threshold tilt
eth=15; %%%% No. of Frames threshold eye close
contre=0; %%%% Eye close counter  

% Track the Points
while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);

    % Track the points. Note that some points may be lost.
    [points1, isFound1] = step(pointTracker1, rgb2gray(videoFrame));
    visiblePoints1 = points1(isFound1, :);
    oldInliers1 = oldPoints1(isFound1, :);
    
       if size(visiblePoints1, 1) >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform1, oldInliers1, visiblePoints1] = estimateGeometricTransform(...
            oldInliers1, visiblePoints1, 'similarity', 'MaxDistance', 4);
        
              % Apply the transformation to the bounding box
        [bboxPolygon1(1:2:end), bboxPolygon1(2:2:end)] ...
            = transformPointsForward(xform1, bboxPolygon1(1:2:end), bboxPolygon1(2:2:end));
       
        x1=bboxPolygon1(1,1);
        y1=bboxPolygon1(1,2);
        x2=bboxPolygon1(1,3);
        y2=bboxPolygon1(1,4);
        x3=bboxPolygon1(1,5);
        y3=bboxPolygon1(1,6);
        x4=bboxPolygon1(1,7);
        y4=bboxPolygon1(1,8);

        xn1=x1+(x4-x1)/4;
        yn1=y1+(y4-y1)/4;
        xn2=x2+(x3-x2)/4;
        yn2=y2+(y3-y2)/4;
        xn3=x2+(x3-x2)/2;
        yn3=y2+(y3-y2)/2;
        xn4=x1+(x4-x1)/2;
        yn4=y1+(y4-y1)/2;

        eyespoly=[xn1,yn1,xn2,yn2,xn3,yn3,xn4,yn4];
        
        %%% Find face inclination %%%%%%%
        deg=atand((y2-y1)/(x2-x1));
        disp(['Face at ' num2str(deg) ' Degrees.'])
        
        %%%%%%%%%%%%%%%%%%%
        currentframe=videoFrame;
        x=[xn1,xn2,xn3,xn4];
        y=[yn1,yn2,yn3,yn4];
        [m,n, ~]=size(currentframe);

        bw=poly2mask(x, y, m, n);
%         figure(5)
%         imshow(bw)
        bwc(:,:,1)=bw;
        bwc(:,:,2)=bw;
        bwc(:,:,3)=bw;

        eyesreg=bwc.*currentframe;
%         figure
%         imshow(eyesreg)
        im=imrotate(eyesreg,deg,'crop');
%         figure
%         imshow(im)

        imbw=im2bw(im,0.0001);
%         figure
%         imshow(imbw)
        L=bwlabel(imbw);
        BB=regionprops(L,'BoundingBox');
        bx=BB.BoundingBox;

        eyeimg=im(bx(2)+20:bx(2)+bx(4), bx(1)+20:bx(1)+bx(3) -50 , : );

        figure(4)
        subplot(2,1,1)
        imshow(eyeimg)
        title('Eye Region')
        
        eyebw=im2bw(eyeimg,0.1);%%% Threshold That may change
        area=sum(sum(imcomplement(eyebw (20:end-20, 30:end-40)  )));
        subplot(2,1,2)
        imshow(eyebw)
        title(num2str(area))
        
       %%%%%%%%%%Eye Close Warning  %%%%%%%%%
        if(area<10) %%%%%%%%%% Eye close area thresh %%%%
            contre=contre+1;
            if(contre>eth) %%%%%%%% No. of Frames Threshold %%%
                disp('Eye  ALERT!       Eye  ALERT!           Eye  ALERT!')
                rockytop2;
                pause(0.5)
                contre=0;
            end
            
        else
            contre=0;
        end
        
        
       %%%%%%%%%%%%Face Tilt Warning  %%%%%%%%%
        if(abs(deg)>ath) %%%%%%%%%% Angle Threshold %%%%
            contr=contr+1;
            if(contr>nfth) %%%%%%%% No. of Frames Threshold %%%
                disp('Face  ALERT!       Face  ALERT!           Face  ALERT!')
                rockytop2;
                pause(0.5)%%%
                contr=0;
            end
        else
            contr=0;
        end
        
        % Insert a bounding box around the object being tracked
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon1);
        videoFrame = insertShape(videoFrame, 'Polygon', eyespoly);                
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints1, '+', ...
            'Color', 'white');       
               
        % Reset the points
        oldPoints1 = visiblePoints1;
        setPoints(pointTracker1, oldPoints1);        
       
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
    k=k+1;
    
end
disp(['Total number of frames= ' num2str(k)])
disp(['Total time required  = ' num2str(toc) ' Seconds'])

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker1);


%  imshow(videoFrame)
%  x=[xn1,xn2,xn3,xn4];
%  y=[yn1,yn2,yn3,yn4];
%  [m,n, ~]=size(videoFrame);
%  
%  bw=poly2mask(x, y, m, n);
%  imshow(bw)
%  bwc(:,:,1)=bw;
%  bwc(:,:,2)=bw;
%  bwc(:,:,3)=bw;
%  
%  eyesreg=bwc.*videoFrame;
%  figure
%  imshow(eyesreg)
%  im=imrotate(eyesreg,-deg,'crop');
%  figure
%  imshow(im)
%  
%  imbw=im2bw(im,0.0001);
%  figure
%  imshow(imbw)
%  L=bwlabel(imbw);
%  BB=regionprops(L,'BoundingBox');
%  bx=BB.BoundingBox;
%  
%  eyeimg=im(bx(2)+20:bx(2)+bx(4), bx(1)+20:bx(1)+bx(3) -50 , : );
%  
%  figure
%  imshow(eyeimg)
 
% eyebw=im2bw(eyeimg,0.7); 
% imshow(eyebw)   

