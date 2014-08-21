function gradientCheck(model, sentence)
  delta = 1e-7;
  EPS = 1e-4;
  
  [computedCost, computedGrad] = trainOneSent(model, sentence);
  computedGrad.tgtWe = full(aggregateMatrix(computedGrad.tgtWe, sentence(1:length(sentence)-1), size(model.tgtWe, 2)));
  computedGrad.tgtTree = full(aggregateMatrix(computedGrad.allTrees,computedGrad.allPaths,size(model.tgtTree,2)));
  
  trueGrad.U = zeros(size(model.U));
  trueGrad.R = zeros(size(model.R));
  trueGrad.W = zeros(size(model.W));
  trueGrad.tgtWe = zeros(size(model.tgtWe));
  trueGrad.tgtTree = zeros(size(model.tgtTree));
  
  for i = 1 : size(model.U,1)
    for j = 1 : size(model.U,2)
      model.U(i,j) = model.U(i,j) + delta;
      [newCost, ~] = trainOneSent(model, sentence);
      trueGrad.U(i,j) = (newCost - computedCost) / delta;
      model.U(i,j) = model.U(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.R,1)
    for j = 1 : size(model.R,2)
      model.R(i,j) = model.R(i,j) + delta;
      [newCost, ~] = trainOneSent(model, sentence);
      trueGrad.R(i,j) = (newCost - computedCost) / delta;
      model.R(i,j) = model.R(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.W,1)
    for j = 1 : size(model.W,2)
      model.W(i,j) = model.W(i,j) + delta;
      [newCost, ~] = trainOneSent(model, sentence);
      trueGrad.W(i,j) = (newCost - computedCost) / delta;
      model.W(i,j) = model.W(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtWe,1)
    for j = 1 : size(model.tgtWe,2)
      model.tgtWe(i,j) = model.tgtWe(i,j) + delta;
      [newCost, ~] = trainOneSent(model, sentence);
      trueGrad.tgtWe(i,j) = (newCost - computedCost) / delta;
      model.tgtWe(i,j) = model.tgtWe(i,j) - delta;
    end
  end
  
  for i = 1 : size(model.tgtTree,1)
    for j = 1 : size(model.tgtTree,2)
      model.tgtTree(i,j) = model.tgtTree(i,j) + delta;
      [newCost, ~] = trainOneSent(model, sentence);
      trueGrad.tgtTree(i,j) = (newCost - computedCost) / delta;
      model.tgtTree(i,j) = model.tgtTree(i,j) - delta;
    end
  end
  
  allTrueGrad = [trueGrad.U(:) ; trueGrad.R(:) ; trueGrad.W(:) ; trueGrad.tgtWe(:) ; trueGrad.tgtTree(:)];
  allComputedGrad = [computedGrad.U(:) ; computedGrad.R(:) ; computedGrad.W(:) ; computedGrad.tgtWe(:) ; computedGrad.tgtTree(:)];
  
  all = [allTrueGrad allComputedGrad];
  disp(all(any(all,2),:));
  [maxVal, maxIndex] = max(abs(all(:,1) - all(:,2)));
  if maxVal > EPS
    error('Gradient check failed at %i, where trueGrad=%f but got %f, diff=%f\n', maxIndex, all(maxIndex,1), all(maxIndex,2), maxVal);
  else
    fprintf('# Gradient check passed\n');
  end