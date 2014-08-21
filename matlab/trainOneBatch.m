function [cost, grad] = trainOneBatch(model, srcSents, tgtSents)
  srcTreeSize = model.srcTreeSize;
  tgtTreeSize = model.tgtTreeSize;
  srcVocabSize = model.srcVocabSize;
  tgtVocabSize = model.tgtVocabSize;
  
  numSents = length(tgtSents);
  if numSents ~= length(srcSents)
    error('fuck %d %d.\n', length(srcSents), length(tgtSents));
  end
  cost = 0;
  
  grad.srcU = zeros(size(model.srcU));
  grad.srcR = zeros(size(model.srcR));
  grad.srcW = zeros(size(model.srcW));
  
  allSrcWords = cell(numSents,1);
  allSrcVects = cell(numSents,1);
  allSrcTrees = cell(numSents,1);
  allSrcPaths = cell(numSents,1);
  
  grad.tgtS = zeros(size(model.tgtS));
  
  grad.tgtU = zeros(size(model.tgtU));
  grad.tgtR = zeros(size(model.tgtR));
  grad.tgtW = zeros(size(model.tgtW));
  
  allTgtWords = cell(numSents,1);
  allTgtVects = cell(numSents,1);
  allTgtTrees = cell(numSents,1);
  allTgtPaths = cell(numSents,1);
  
  for i = 1 : numSents
%     [costSents, gradSents] = trainOneSent(model, srcSents{i}, tgtSents{i});
%     if length(srcSents{i}) < 2 || length(tgtSents{i}) < 2
%       continue;
%     end
%     cost = cost + costSents;
%     
%     grad.srcU = grad.srcU + gradSents.srcU;
%     grad.srcR = grad.srcR + gradSents.srcR;
%     grad.srcW = grad.srcW + gradSents.srcW;
%     
%     grad.tgtU = grad.tgtU + gradSents.tgtU;
%     grad.tgtR = grad.tgtR + gradSents.tgtR;
%     grad.tgtW = grad.tgtW + gradSents.tgtW;
%     
%     grad.tgtS = grad.tgtS + gradSents.tgtS;
%     
%     allSrcWords{i} = srcSents{i}(1 : length(srcSents{i}));
%     allSrcVects{i} = gradSents.srcWe;
%     allSrcTrees{i} = gradSents.allSrcTrees;
%     allSrcPaths{i} = gradSents.allSrcPaths;
%     
%     allTgtWords{i} = tgtSents{i}(1 : length(tgtSents{i})-1);
%     allTgtVects{i} = gradSents.tgtWe;
%     allTgtTrees{i} = gradSents.allTgtTrees;
%     allTgtPaths{i} = gradSents.allTgtPaths;
    gradientCheck(model, srcSents{i}, tgtSents{i});
  end
  
  cost = cost / numSents;
  
  grad.tgtU = grad.tgtU / numSents;
  grad.tgtR = grad.tgtR / numSents;
  grad.tgtW = grad.tgtW / numSents;
  
  grad.srcU = grad.srcU / numSents;
  grad.srcR = grad.srcR / numSents;
  grad.srcW = grad.srcW / numSents;
  
  grad.tgtWe = full(aggregateMatrix(horzcat(allTgtVects{:}), vertcat(allTgtWords{:}), tgtVocabSize) / numSents);
  grad.tgtTree = full(aggregateMatrix(horzcat(allTgtTrees{:}), vertcat(allTgtPaths{:}), tgtTreeSize) / numSents);
  
%   size(horzcat(allSrcVects{:}))
%   size(vertcat(allSrcWords{:}))
  grad.srcWe = full(aggregateMatrix(horzcat(allSrcVects{:}), vertcat(allSrcWords{:}), srcVocabSize) / numSents);
  grad.srcTree = full(aggregateMatrix(horzcat(allSrcTrees{:}), vertcat(allSrcPaths{:}), srcTreeSize) / numSents);
  
%   allGrad = [grad.tgtU(:) ; grad.tgtR(:) ; grad.W(:) ; grad.tgtWe(:) ; grad.tgtTree(:)];
%   max(allGrad)
