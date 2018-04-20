function [rmin,rmax,cmin,cmax]=normWindow(rmin,rmax,cmin,cmax,height,width)
% function: examine if the tracking window is in the range of image
% parameters:
%        rmin,ramx,cmin,cmax: window of tracking result
%        height,widht: the size of image

if rmin<2
    rmin=2;
end

if rmax>height-1;
    rmax=height-1;
end

if cmin<2
    cmin=2;
end

if cmax>width-1
    cmax=width-1
end