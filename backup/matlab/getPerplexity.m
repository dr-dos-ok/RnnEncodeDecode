function [cost] = getPerplexity(params, model)
  validFile = sprintf('%s/%s', params.dataPath, params.tgtValid);
  fileID = fopen(validFile, 'r');
  
  cost = 0;
  wordCounts = 0;
  while ~feof(fileID)
    [sents, ~] = loadBatchData(fileID, 1, 1);
    sentence = sents{1};
    cost = cost + sentencePerplexity(model, sentence);
    wordCounts = wordCounts + length(sentence);
  end
  
  cost = exp(cost / wordCounts);
  
  fclose(fileID);