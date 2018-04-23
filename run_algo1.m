clc
clear all

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
[fn,pn]=uigetfile({'*.*'},'Select Face Video file');

videoFileReader = vision.VideoFileReader(strcat(pn,fn));

flg=1;
k=0;
tic
while(flg==1)
videoFrame      = step(videoFileReader);
bbox1            = step(faceDetector, videoFrame);

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox1,'Face Region');
figure(1), imshow(videoOut), title('Detected face');

% Step 2: Identify Facial Features To Track

[hueChannel,~,~] = rgb2hsv(videoFrame);
if(size(bbox1,1)==1)
% Display the Hue Channel data and draw the bounding box around the face.
    figure(2), imshow(hueChannel), title('Hue channel data');
    rectangle('Position',bbox1(1,:),'LineWidth',2,'EdgeColor',[1 1 0])
    if(bbox1(1,3)<200 && bbox1(4)<240)
        disp('here..')
        continue;
    end
else
    continue;
end
% Step 3: Track the Face

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

figure(2)
hold on
rectangle('Position',eyesBBox(1,:),'LineWidth',2,'EdgeColor',[0 0 1])

end
k=k+1;
end
disp(['Frames required to detect face= ' num2str(k)])
disp(['Time required to detect face = ' num2str(toc) ' Seconds'])

% Create a tracker object.
tracker1 = vision.HistogramBasedTracker;
tracker2 = vision.HistogramBasedTracker;

% Initialize the tracker histogram using the Hue channel pixels from the
% nose.
initializeObject(tracker1, hueChannel, bbox1(1,:));
initializeObject(tracker2, hueChannel, eyesBBox(1,:));

% Create a video player object for displaying video frames.
videoInfo    = info(videoFileReader);
videoPlayer  = vision.VideoPlayer('Position',[300 300 videoInfo.VideoSize+30]);

% Track the face over successive video frames until the video is finished.
while ~isDone(videoFileReader)
    
    k=k+1;
    % Extract the next video frame
    videoFrame = step(videoFileReader);
    
    % RGB -> HSV
    [hueChannel,~,~] = rgb2hsv(videoFrame);
    
    % Track Face using the Hue channel data
    bbox1 = step(tracker1, hueChannel);
    disp(['Face Size =' num2str(bbox1(3)) 'X' num2str(bbox1(4))]);
    
    % Track Eyes using the Hue channel data
    bbox2 = step(tracker2, hueChannel);
    disp(['Eyes Size =' num2str(bbox2(3)) 'X' num2str(bbox2(4))]);
    
    % Insert a bounding box around the object being tracked
    videoOut = insertObjectAnnotation(videoFrame,'rectangle',bbox1,'Face');    
    videoOut = insertObjectAnnotation(videoOut,'rectangle',bbox2,'Eyes');  
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoOut);
    imwrite(videoOut,['.\Results\Algorithm1\' num2str(k) ,'.jpg']);
     
end

disp(['Total number of frames = ' num2str(k)])
disp(['Total time required  = ' num2str(toc) ' Seconds'])

% Release resources
release(videoFileReader);
release(videoPlayer);


