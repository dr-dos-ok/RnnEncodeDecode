function [cost, grad] = trainOneSent(model, sentence)
  hiddenSize = model.hiddenSize;
  tgtFeatureSize = model.tgtFeatureSize;
  
  n = length(sentence);
  
  hidden = zeros(hiddenSize, n-1);
  U = model.U;
  R = model.R;
  W = model.W;
  tgtWe = model.tgtWe;
  tgtWordPaths = model.tgtWordPaths;
  tgtTree = model.tgtTree;
  
  dPaths = cell(n-1,1);
  dTrees = cell(n-1,1);
  
  cost = 0;
  for i = 1 : n-1
    if i == 1
      hidden(:,i) = sigmoid(U * tgtWe(:,sentence(i)));
    else
      hidden(:,i) = sigmoid(U * tgtWe(:,sentence(i)) + R * hidden(:,i-1));
    end
  end
  
  feats = W * hidden;
  dFeats = zeros(tgtFeatureSize,n-1);
  
  for i = 1 : n-1
    path = tgtWordPaths{sentence(i+1)};
    len = length(path);               % path length
    dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    
%     size(tgtTree(:, path(1 : len-1)))
%     size(feats(:,i))
%     size(dir)
%     if size(dir,2) == 0
%       fprintf('word = %s, wordIndex = %d\n', model.tgtWords{sentence(i+1)}, sentence(i+1));
%       tgtWordPaths{sentence(i+1)}
%     end
    probs = sigmoid((tgtTree(:, path(1 : len-1))' * feats(:,i)) .* dir);
    cost = cost - sum(log(probs));
    
    dFeats(:,i) = tgtTree(:, path(1 : len-1)) * (dir .* (1 - probs));
    dTrees{i} = feats(:,i) * (dir .* (1 - probs))';
    dPaths{i} = path(1 : len-1);
  end
  
  dHidden = -W' * dFeats;
  grad.W = dFeats * hidden';
  
  % back propagation through time
  for i = n-2 : -1 : 1
    dHidden(:,i) = dHidden(:,i) - R' * (dHidden(:,i+1) .* hidden(:,i+1) .* (1 - hidden(:,i+1)));
  end
  
  dHidden = dHidden .* hidden .* (1 - hidden);
  grad.R = dHidden(:,2:n-1) * hidden(:,1:n-2)';       % R
  grad.U = dHidden * tgtWe(:,sentence(1:n-1))';       % U
  grad.tgtWe = U' * dHidden;                          % tgtWe
  
  grad.allTrees = horzcat(dTrees{:});
  grad.allPaths = vertcat(dPaths{:});