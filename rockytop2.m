function rockytop2
%playtune('C D E F G A B C^4');
%playtune('C C# D D# E F Gb G Ab A Bb B C^4');
% a text string that represents the song to be played
% The notes are space delimited
% each note has this format:
% starts with a single character representing the note (A-G)
% optional # or b (sharp or flat)
% optional ^ and v characters to shift the octave. (A^^ plays two octaves up)
% optional duration
% no duration means a quarter note
% a number without a / multiplies (A2 plays a half note)
% a number with a / divides (A/2 plays an eighth note)
%figure(1);
%hold on;
%title('Rocky Top');
%axis([0 120 400 900]);
%xlabel('Time');
%ylabel('Frequency (Hz)');
tune={
'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'

}
playtune(tune);
return
function note(pitch, duration, oct)
% play a note
% pitch is a number from 0 to 12 where 0 is middle C
% duration is 1 for a quarter note
% oct is an +/- integer for the desired octaves (0 is middle C)
duration=duration*4; % a quarter note is 1/4 of a second
hz = 660 * 2^(pitch / 12+oct);
% combine three harmonics for a better sound
mid = tone(1.0*hz, duration);
hi = tone(2.0*hz, duration);
lo = tone(0.5*hz, duration);
a = .5*mid + .25*hi + .25*lo;
sound(a,14400);
return
function a = tone(hz, duration)
% generate data for a given tone
SAMPLES_PER_SECOND = 14400;
N = floor(SAMPLES_PER_SECOND * duration);
i = 0:N;
a = sin (2 * pi * i * hz / SAMPLES_PER_SECOND);
return
function playtune(a)
% this routine parses the input as described above
% convert the cell array into a single text array
if iscell(a) 
t='';
for i=1:length(a)
t=[t a{i} ' '];
end
else
t=a;
end
px=0;
% loop through the song
while true
[str, t] = strtok(t); % get the next note
if isempty(str), break; end
%fprintf('%s\n',str);
n=0; d=1; o=0;
for i=1:length(str)
c=str(i);
if c=='/'
d=1/str2num(str(i+1:end));
break;
end
if c>='0' && c<='9'
d=str2num(str(i:end));
break
end
switch c
case 'C'
n=0;
case 'D'
n=2;
case 'E'
n=4;
case 'F'
n=5;
case 'G'
n=7;
case 'A'
n=9;
case 'B'
n=11;
case '#'
n=n+1;
case 'b'
n=n-1;
case '^';
o=o+1;
case 'v'
o=o-1;
end
end
note(n,d,o);
f=n+o*12;
f=440 * 2^(n/12+o);
%plot([px px+d],[f f],'-','markerfacecolor','r','markeredgecolor','r');
px=px+d;
%drawnow;
end
return
