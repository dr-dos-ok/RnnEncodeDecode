function gradientCheck(model, srcSentence, tgtSentence)
  delta = 1e-7;
  EPS = 1e-4;
  
  [computedCost, computedGrad] = trainOneSent(model, srcSentence, tgtSentence);
  computedGrad.srcWe = full(aggregateMatrix(computedGrad.srcWe, srcSentence(1:length(srcSentence)), size(model.srcWe, 2)));
  computedGrad.tgtWe = full(aggregateMatrix(computedGrad.tgtWe, tgtSentence(1:length(tgtSentence)-1), size(model.tgtWe, 2)));
  computedGrad.srcTree = full(aggregateMatrix(computedGrad.allSrcTrees,computedGrad.allSrcPaths,size(model.srcTree,2)));
  computedGrad.tgtTree = full(aggregateMatrix(computedGrad.allTgtTrees,computedGrad.allTgtPaths,size(model.tgtTree,2)));
  
  trueGrad.srcU = zeros(size(model.srcU));
  trueGrad.srcR = zeros(size(model.srcR));
  trueGrad.srcW = zeros(size(model.srcW));
  trueGrad.srcWe = zeros(size(model.srcWe));
  trueGrad.srcTree = zeros(size(model.srcTree));
  trueGrad.tgtU = zeros(size(model.tgtU));
  trueGrad.tgtR = zeros(size(model.tgtR));
  trueGrad.tgtW = zeros(size(model.tgtW));
  trueGrad.tgtS = zeros(size(model.tgtS));
  trueGrad.tgtWe = zeros(size(model.tgtWe));
  trueGrad.tgtTree = zeros(size(model.tgtTree));
  
  for i = 1 : size(model.srcU,1)
    for j = 1 : size(model.srcU,2)
      model.srcU(i,j) = model.srcU(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.srcU(i,j) = (newCost - computedCost) / delta;
      model.srcU(i,j) = model.srcU(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.srcR,1)
    for j = 1 : size(model.srcR,2)
      model.srcR(i,j) = model.srcR(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.srcR(i,j) = (newCost - computedCost) / delta;
      model.srcR(i,j) = model.srcR(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.srcWe,1)
    for j = 1 : size(model.srcWe,2)
      model.srcWe(i,j) = model.srcWe(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.srcWe(i,j) = (newCost - computedCost) / delta;
      model.srcWe(i,j) = model.srcWe(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.srcTree,1)
    for j = 1 : size(model.srcTree,2)
      model.srcTree(i,j) = model.srcTree(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.srcTree(i,j) = (newCost - computedCost) / delta;
      model.srcTree(i,j) = model.srcTree(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtU,1)
    for j = 1 : size(model.tgtU,2)
      model.tgtU(i,j) = model.tgtU(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtU(i,j) = (newCost - computedCost) / delta;
      model.tgtU(i,j) = model.tgtU(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtR,1)
    for j = 1 : size(model.tgtR,2)
      model.tgtR(i,j) = model.tgtR(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtR(i,j) = (newCost - computedCost) / delta;
      model.tgtR(i,j) = model.tgtR(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtS,1)
    for j = 1 : size(model.tgtS,2)
      model.tgtS(i,j) = model.tgtS(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtS(i,j) = (newCost - computedCost) / delta;
      model.tgtS(i,j) = model.tgtS(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtW,1)
    for j = 1 : size(model.tgtW,2)
      model.tgtW(i,j) = model.tgtW(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtW(i,j) = (newCost - computedCost) / delta;
      model.tgtW(i,j) = model.tgtW(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtWe,1)
    for j = 1 : size(model.tgtWe,2)
      model.tgtWe(i,j) = model.tgtWe(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtWe(i,j) = (newCost - computedCost) / delta;
      model.tgtWe(i,j) = model.tgtWe(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtTree,1)
    for j = 1 : size(model.tgtTree,2)
      model.tgtTree(i,j) = model.tgtTree(i,j) + delta;
      [newCost, ~] = trainOneSent(model, srcSentence, tgtSentence);
      trueGrad.tgtTree(i,j) = (newCost - computedCost) / delta;
      model.tgtTree(i,j) = model.tgtTree(i,j) - delta;
    end
  end
  
%   allTrueGrad = [trueGrad.srcU(:)];
%   allComputedGrad = [computedGrad.srcU(:)];
  
  allTrueGrad = [trueGrad.srcU(:) ; trueGrad.srcR(:) ; trueGrad.srcWe(:) ; trueGrad.srcTree(:) ; ...
                 trueGrad.tgtU(:) ; trueGrad.tgtR(:) ; trueGrad.tgtW(:) ; trueGrad.tgtS(:) ; trueGrad.tgtWe(:) ; trueGrad.tgtTree(:)];
  allComputedGrad = [computedGrad.srcU(:) ; computedGrad.srcR(:) ; computedGrad.srcWe(:) ; computedGrad.srcTree(:) ; ...
                     computedGrad.tgtU(:) ; computedGrad.tgtR(:) ; computedGrad.tgtW(:) ; computedGrad.tgtS(:) ; computedGrad.tgtWe(:) ; computedGrad.tgtTree(:)];
  
  all = [allTrueGrad allComputedGrad];
  disp(all(any(all,2),:));
  [maxVal, maxIndex] = max(abs(all(:,1) - all(:,2)));
  if maxVal > EPS
    error('Gradient check failed at %i, where trueGrad=%f but got %f, diff=%f\n', maxIndex, all(maxIndex,1), all(maxIndex,2), maxVal);
  else
    fprintf('# Gradient check passed\n');
  end