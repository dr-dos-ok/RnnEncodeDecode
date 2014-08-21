%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assumming the format                                 %
%         vocabSize                                    %
%         word [path]                                  %
%  for each entry in the dictionary file.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [words, paths, dirs] = loadDictionary(params, side, fullPath) % if fullPath == 0, this will omit the leave of the tree for each word
  language = sprintf('%sLang', side);
  pathsFile = sprintf('%s/%s.paths', params.dataPath, params.(language));
  wordsFile = sprintf('%s/%s.words', params.dataPath, params.(language));

  fprintf('# Reading %s dictionary from files "%s" and "%s"...', language, pathsFile, wordsFile);
  wordsID = fopen(wordsFile, 'r');
  words = textscan(wordsID, '%s');
  words = words{1};
  fclose(wordsID);
  
  vocabSize = length(words);
  
  pathID = fopen(pathsFile, 'r');
  paths = cell(vocabSize, 1);
  dirs = cell(vocabSize, 1);
  for i = 1 : vocabSize
    path = sscanf(fgetl(pathID), '%d');
    dirs{i} = sscanf(fgetl(pathID), '%d');
    if fullPath
      paths{i} = path(1 : length(path));
    else
      paths{i} = path(1 : length(path)-1);
    end
  end
  fclose(pathID);
  
  fprintf('done\n');
