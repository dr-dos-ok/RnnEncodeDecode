function [srcCost, tgtCost] = getPerplexity(params, model)
  %% source
  srcValidFile = sprintf('%s/%s', params.dataPath, params.srcValid);
  srcID = fopen(srcValidFile, 'r');
  
  tgtValidFile = sprintf('%s/%s', params.dataPath, params.tgtValid);
  tgtID = fopen(tgtValidFile, 'r');
  
  srcCost = 0;
  tgtCost = 0;
  srcWordCounts = 0;
  tgtWordCounts = 0;
  while ~feof(srcID)
    [srcSents, ~] = loadBatchData(srcID, 1, 1);
    [tgtSents, ~] = loadBatchData(tgtID, 1, 1);
    srcSentence = srcSents{1};
    tgtSentence = tgtSents{1};
    [currSrc, currTgt] = sentencePerplexity(model, srcSentence, tgtSentence);
    srcCost = srcCost + currSrc;
    tgtCost = tgtCost + currTgt;
    srcWordCounts = srcWordCounts + length(srcSentence);
    tgtWordCounts = tgtWordCounts + length(tgtSentence);
  end
  
  srcCost = exp(srcCost / srcWordCounts);
  tgtCost = exp(tgtCost / tgtWordCounts);
  
  fclose(tgtID);
  fclose(srcID);