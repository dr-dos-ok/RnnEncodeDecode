function [cost, grad] = trainOneSent(model, srcSentence, tgtSentence)
  %% source
  srcHiddenSize = model.srcHiddenSize;    tgtHiddenSize = model.tgtHiddenSize;
  srcFeatureSize = model.srcFeatureSize;  tgtFeatureSize = model.tgtFeatureSize;
  srcU = model.srcU;                      tgtU = model.tgtU;
  srcR = model.srcR;                      tgtR = model.tgtR;
  srcW = model.srcW;                      tgtW = model.tgtW;
  srcWe = model.srcWe;                    tgtWe = model.tgtWe;
  srcWordPaths = model.srcWordPaths;      tgtWordPaths = model.tgtWordPaths;
  srcWordDirs = model.srcWordDirs;        tgtWordDirs = model.tgtWordDirs;
  srcTree = model.srcTree;                tgtTree = model.tgtTree;
                                          tgtS = model.tgtS;
  
  m = length(srcSentence);
  n = length(tgtSentence);
  
  srcHidden = zeros(srcHiddenSize, m);
  tgtHidden = zeros(tgtHiddenSize, n);
  
  dSrcPaths = cell(m,1);
  dSrcTrees = cell(m,1);
  dTgtPaths = cell(n,1);
  dTgtTrees = cell(n,1);
  
  cost = 0;
  srcHidden(:,1) = sigmoid(srcU * srcWe(:,srcSentence(1)));
  for i = 2 : m
    srcHidden(:,i) = sigmoid(srcU * srcWe(:,srcSentence(i)) + srcR * srcHidden(:,i-1));
  end
  
  srcFeats = srcW * srcHidden(:,1:m-1);
  dSrcFeats = zeros(srcFeatureSize,m-1);
  
  for i = 1 : m-1
    path = srcWordPaths{srcSentence(i+1)};
    len = length(path);               % path length
%     dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    dir = (-1) .^ srcWordDirs{srcSentence(i+1)};
    
    probs = sigmoid((srcTree(:, path)' * srcFeats(:,i)) .* dir);
    cost = cost - sum(log(probs));
    
    dSrcFeats(:,i) = srcTree(:, path) * (dir .* (1 - probs));
    dSrcTrees{i} = srcFeats(:,i) * (dir .* (1 - probs))';
    dSrcPaths{i} = path;
  end
  
  b = tgtS * srcHidden(:,m);
%   b = zeros(srcHiddenSize,1);

  %% target
  tgtHidden(:,1) = sigmoid(b);
  for i = 2 : n
    tgtHidden(:,i) = sigmoid(tgtU * tgtWe(:,tgtSentence(i-1)) + tgtR * tgtHidden(:,i-1) + b);
  end
  
  tgtFeats = tgtW * tgtHidden;
  dTgtFeats = zeros(tgtFeatureSize,n-1);
  
  for i = 1 : n
    path = tgtWordPaths{tgtSentence(i)};
    len = length(path);               % path length
%     dir = (-1) .^ mod(path(2:len),2); % direction, 0 means go left, 1 means go right
    dir = (-1) .^ tgtWordDirs{tgtSentence(i)};
    
    probs = sigmoid((tgtTree(:, path)' * tgtFeats(:,i)) .* dir);
    cost = cost - sum(log(probs));
    
    dTgtFeats(:,i) = tgtTree(:, path) * (dir .* (1 - probs));
    dTgtTrees{i} = tgtFeats(:,i) * (dir .* (1 - probs))';
    dTgtPaths{i} = path;
  end
  
  dTgtHidden = -tgtW' * dTgtFeats;
  grad.tgtW = dTgtFeats * tgtHidden';
  
  % back propagation through time
  for i = n-1 : -1 : 1
    dTgtHidden(:,i) = dTgtHidden(:,i) - tgtR' * (dTgtHidden(:,i+1) .* tgtHidden(:,i+1) .* (1 - tgtHidden(:,i+1)));
  end
  
  dTgtHidden = dTgtHidden .* tgtHidden .* (1 - tgtHidden);
  dB = sum(dTgtHidden,2);
%   dB = zeros(tgtHiddenSize,1);
  grad.tgtU = dTgtHidden(:,2:n) * tgtWe(:,tgtSentence(1:n-1))';       % tgtU
  grad.tgtR = dTgtHidden(:,2:n) * tgtHidden(:,1:n-1)';                % tgtR
  grad.tgtS = dB * srcHidden(:,m)';                                   % tgtS
  grad.tgtWe = tgtU' * dTgtHidden(:,2:n);                             % tgtWe
  
  dSrcHidden = -[srcW' * dSrcFeats, tgtS' * dB];
  grad.srcW = dSrcFeats * srcHidden(:,1:m-1)';
  
  % back propagation through time
  for i = m-1 : -1 : 1
    dSrcHidden(:,i) = dSrcHidden(:,i) - srcR' * (dSrcHidden(:,i+1) .* srcHidden(:,i+1) .* (1 - srcHidden(:,i+1)));
  end
  
  dSrcHidden = dSrcHidden .* srcHidden .* (1 - srcHidden);
  grad.srcU = dSrcHidden * srcWe(:,srcSentence(:))';         % srcU
  grad.srcR = dSrcHidden(:,2:m) * srcHidden(:,1:m-1)';       % srcR
  grad.srcWe = srcU' * dSrcHidden;                           % srcWe
  
  grad.allSrcTrees = horzcat(dSrcTrees{:});
  grad.allSrcPaths = vertcat(dSrcPaths{:});
  grad.allTgtTrees = horzcat(dTgtTrees{:});
  grad.allTgtPaths = vertcat(dTgtPaths{:});
