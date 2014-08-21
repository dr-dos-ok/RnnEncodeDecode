%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assumming the format                                 %
%         vocabSize                                    %
%         word [path]                                  %
%  for each entry in the dictionary file.              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [words, paths] = loadDictionary(params)
  pathsFile = sprintf('%s/%s.paths', params.dataPath, params.tgtLang);
  wordsFile = sprintf('%s/%s.words', params.dataPath, params.tgtLang);

  fprintf('# Reading dictionary from files "%s" and "%s"...', pathsFile, wordsFile);
  wordsID = fopen(wordsFile, 'r');
  words = textscan(wordsID, '%s');
  words = words{1};
  fclose(wordsID);
  
  vocabSize = length(words);
  
  pathID = fopen(pathsFile, 'r');
  paths = cell(vocabSize, 1);
  for i = 1 : vocabSize
    paths{i} = sscanf(fgetl(pathID), '%d');
  end
  fclose(pathID);
  
  fprintf('done\n');
