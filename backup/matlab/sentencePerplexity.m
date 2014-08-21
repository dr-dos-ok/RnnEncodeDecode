function [cost] = sentencePerplexity(model, sentence)
  hiddenSize = model.hiddenSize;
  
  n = length(sentence);
  
  hidden = zeros(hiddenSize, n-1);
  U = model.U;
  R = model.R;
  W = model.W;
  tgtWe = model.tgtWe;
  tgtWordsPath = model.tgtWordPaths;
  tgtTree = model.tgtTree;
  
  cost = 0;
  
  for i = 1 : n-1
    if i == 1
      hidden = sigmoid(U * tgtWe(:,sentence(1)));
    else
      hidden = sigmoid(U * tgtWe(:,sentence(i)) + R * hidden);
    end
    
    feats = W * hidden;
    path = tgtWordsPath{sentence(i+1)};
    len = length(path);               % path length
    dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    
    probs = sigmoid((tgtTree(:, path(1 : len-1))' * feats) .* dir);
    cost = cost - sum(log(probs));
  end