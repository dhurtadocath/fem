function folderName = getfolderName(searchString)
if nargin<1
    stackInfo = dbstack();
    searchString = stackInfo(end).name;
end
items = dir; 
folders = items([items.isdir] & ~ismember({items.name}, {'.', '..'})); 
filteredFolders = folders(contains({folders.name}, searchString)); 
folderNames = {filteredFolders.name};
sortednames = sort(folderNames);
folderName = sortednames{end};

end