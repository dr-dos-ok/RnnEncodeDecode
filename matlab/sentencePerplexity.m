function [srcCost, tgtCost] = sentencePerplexity(model, srcSentence, tgtSentence)
  %% source
  srcHiddenSize = model.srcHiddenSize;    tgtHiddenSize = model.tgtHiddenSize;
  srcFeatureSize = model.srcFeatureSize;  tgtFeatureSize = model.tgtFeatureSize;
  srcU = model.srcU;                      tgtU = model.tgtU;
  srcR = model.srcR;                      tgtR = model.tgtR;
  srcW = model.srcW;                      tgtW = model.tgtW;
  srcWe = model.srcWe;                    tgtWe = model.tgtWe;
  srcWordPaths = model.srcWordPaths;      tgtWordPaths = model.tgtWordPaths;
  srcTree = model.srcTree;                tgtTree = model.tgtTree;
                                          tgtS = model.tgtS;
  
  m = length(srcSentence);
  n = length(tgtSentence);
  
  srcHidden = zeros(srcHiddenSize, m);
  tgtHidden = zeros(tgtHiddenSize, n);
  
  srcCost = 0;
  srcHidden(:,1) = sigmoid(srcU * srcWe(:,srcSentence(1)));
  for i = 2 : m
    srcHidden(:,i) = sigmoid(srcU * srcWe(:,srcSentence(i)) + srcR * srcHidden(:,i-1));
  end
  
  srcFeats = srcW * srcHidden(:,1:m-1);
  
  for i = 1 : m-1
    path = srcWordPaths{srcSentence(i+1)};
    len = length(path);               % path length
    dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    
    probs = sigmoid((srcTree(:, path(1 : len-1))' * srcFeats(:,i)) .* dir);
    srcCost = srcCost - sum(log(probs));
  end
  
  b = tgtS * srcHidden(:,m);

  %% target
  tgtHidden(:,1) = sigmoid(b);
  for i = 2 : n
    tgtHidden(:,i) = sigmoid(tgtU * tgtWe(:,tgtSentence(i-1)) + tgtR * tgtHidden(:,i-1) + b);
  end
  
  tgtFeats = tgtW * tgtHidden;
  
  tgtCost = 0;
  for i = 1 : n
    path = tgtWordPaths{tgtSentence(i)};
    len = length(path);               % path length
    dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    
    probs = sigmoid((tgtTree(:, path(1 : len-1))' * tgtFeats(:,i)) .* dir);
    tgtCost = tgtCost - sum(log(probs));
  end
