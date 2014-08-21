function [cost, grad] = trainOneBatch(model, tgtSents)
  tgtTreeSize = model.tgtTreeSize;
  tgtVocabSize = model.tgtVocabSize;
  numSents = length(tgtSents);
  cost = 0;
  grad.U = zeros(size(model.U));
  grad.R = zeros(size(model.R));
  grad.W = zeros(size(model.W));
  
  allTgtWords = cell(numSents,1);
  allTgtVects = cell(numSents,1);
  allTgtTrees = cell(numSents,1);
  allTgtPaths = cell(numSents,1);
  
  for i = 1 : numSents
    [costSents, gradSents] = trainOneSent(model, tgtSents{i});
    cost = cost + costSents;
    
    grad.U = grad.U + gradSents.U;
    grad.R = grad.R + gradSents.R;
    grad.W = grad.W + gradSents.W;
    
    allTgtWords{i} = tgtSents{i}(1 : length(tgtSents{i})-1);
    allTgtVects{i} = gradSents.tgtWe;
    allTgtTrees{i} = gradSents.allTrees;
    allTgtPaths{i} = gradSents.allPaths;
%     gradientCheck(model, tgtSents{i});
  end
  
  cost = cost / numSents;
  
  grad.U = grad.U / numSents;
  grad.R = grad.R / numSents;
  grad.W = grad.W / numSents;
  grad.tgtWe = full(aggregateMatrix(horzcat(allTgtVects{:}), vertcat(allTgtWords{:}), tgtVocabSize) / numSents);
  grad.tgtTree = full(aggregateMatrix(horzcat(allTgtTrees{:}), vertcat(allTgtPaths{:}), tgtTreeSize) / numSents);
  
%   allGrad = [grad.U(:) ; grad.R(:) ; grad.W(:) ; grad.tgtWe(:) ; grad.tgtTree(:)];
%   max(allGrad)