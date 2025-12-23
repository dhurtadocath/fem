function folderName = createFolder(scriptName)
if nargin<1
    stackInfo = dbstack();
    scriptName = stackInfo(end).name;
end
currentDateTime = now;
formattedDateTime = datestr(currentDateTime, 'yymmdd_HHMM');
folderName = [formattedDateTime '_' scriptName '/'];
mkdir(folderName);
end