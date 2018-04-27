clc
clear all
close all

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

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox1,'Face Region');
figure(1), imshow(videoOut), title('Detected face');

if(size(bbox1,1)==1)
    if(bbox1(1,3)<200 && bbox1(4)<240)
        disp('here..')
        continue;
    end
else
    continue;
end

eyesDetector = vision.CascadeObjectDetector('EyePairBig');
faceImage    = imcrop(videoFrame,bbox1(1,:));
eyesBBox     = step(eyesDetector,faceImage);

% The nose bounding box is defined relative to the cropped face image.
if size(eyesBBox,1)==1
eyesBBox(1,1:2) = eyesBBox(1,1:2) + bbox1(1,1:2);
flg=0;

figure(1)
hold on
rectangle('Position',eyesBBox(1,:),'LineWidth',2,'EdgeColor',[0 0 1])

end
k=k+1;
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

x = eyesBBox(1, 1);
y = eyesBBox(1, 2); 
w = eyesBBox(1, 3); 
h = eyesBBox(1, 4);
bboxPolygon2 = [x, y, x+w, y, x+w, y+h, x, y+h];

% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon1);
videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon2);

figure(2); imshow(videoFrame); title('Detected Face and Eyes');

% Detect feature points in the face region.
points1 = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', bbox1);
points2 = detectMinEigenFeatures(rgb2gray(videoFrame), 'ROI', eyesBBox);

% Display the detected points.
figure(3), imshow(videoFrame), hold on, title('Detected features');
plot(points1);
hold on
plot(points2);

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker1 = vision.PointTracker('MaxBidirectionalError', 2);
pointTracker2 = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points1 = points1.Location;
initialize(pointTracker1, points1, rgb2gray(videoFrame));
points2 = points2.Location;
initialize(pointTracker2, points2, rgb2gray(videoFrame));


% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 videoInfo.VideoSize(1:2)+30]);

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints1 = points1;
oldPoints2 = points2;

% Track the Points
while ~isDone(videoFileReader)
    % get the next frame
    videoFrame = step(videoFileReader);

    % Track the points. Note that some points may be lost.
    [points1, isFound1] = step(pointTracker1, rgb2gray(videoFrame));
    visiblePoints1 = points1(isFound1, :);
    oldInliers1 = oldPoints1(isFound1, :);
    
    [points2, isFound2] = step(pointTracker2, rgb2gray(videoFrame));
    visiblePoints2 = points2(isFound2, :);
    oldInliers2 = oldPoints2(isFound2, :);
    
    if size(visiblePoints1, 1) >= 2  && size(visiblePoints2, 1) >= 2% need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform1, oldInliers1, visiblePoints1] = estimateGeometricTransform(...
            oldInliers1, visiblePoints1, 'similarity', 'MaxDistance', 4);
        
        [xform2, oldInliers2, visiblePoints2] = estimateGeometricTransform(...
            oldInliers2, visiblePoints2, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box
        [bboxPolygon1(1:2:end), bboxPolygon1(2:2:end)] ...
            = transformPointsForward(xform1, bboxPolygon1(1:2:end), bboxPolygon1(2:2:end));
        
        [bboxPolygon2(1:2:end), bboxPolygon2(2:2:end)] ...
            = transformPointsForward(xform2, bboxPolygon2(1:2:end), bboxPolygon2(2:2:end));
        
        % Insert a bounding box around the object being tracked
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon1);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon2);
                
        % Display tracked points
        videoFrame = insertMarker(videoFrame, visiblePoints1, '+', ...
            'Color', 'white');       
        videoFrame = insertMarker(videoFrame, visiblePoints2, '+', ...
            'Color', 'white');       
        
        % Reset the points
        oldPoints1 = visiblePoints1;
        oldPoints2 = visiblePoints2;
        setPoints(pointTracker1, oldPoints1);        
        setPoints(pointTracker2, oldPoints2);        
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFrame);
    imwrite(videoFrame,['.\Results\Final_algorithm\' num2str(k) ,'.jpg']);
    k=k+1;
end
disp(['Total number of frames = ' num2str(k)])
disp(['Total time required  = ' num2str(toc) ' Seconds'])

% Clean up
release(videoFileReader);
release(videoPlayer);
release(pointTracker1);
release(pointTracker2);

