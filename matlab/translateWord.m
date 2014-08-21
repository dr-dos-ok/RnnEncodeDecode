function [output] = translateWord(model, srcWord)
  srcIndex = find(strcmp(model.srcWords, srcWord));
  srcHidden = sigmoid(model.srcU * model.srcWe(:,srcIndex));
  tgtHidden = sigmoid(model.tgtS * sigmoid(srcHidden));
  feats = model.tgtW * tgtHidden;
  
  probs = sigmoid(model.tgtTree' * feats);
  leftProbs = log(probs);
  rightProbs = log(1 - probs);
  
  accumProbs = zeros(3*length(probs)+1,1);
  
  for i = 2 : length(accumProbs)
    parent = floor(i / 2 + 0.01);
    if parent <= length(probs)
      if mod(i,2) == 0
        accumProbs(i) = accumProbs(parent) + leftProbs(parent);
      else
        accumProbs(i) = accumProbs(parent) + rightProbs(parent);
      end
    else
      break;
    end
  end
  
  [tgtIndex, bestValue] = maxProbs(accumProbs, 1, 1, length(model.tgtWords));
  fprintf('index = %d\tword = %f\n', tgtIndex, bestValue);
  output = model.tgtWords(tgtIndex);
  
function [wordIndex, bestValue] = maxProbs(accumProbs, node, tLeft, tRight)
  if tLeft >= tRight
    wordIndex = tLeft;
    bestValue = accumProbs(node);
    return;
  end
  leftChild = 2 * node;
  rightChild = 2 * node + 1;
  tMid = floor((tLeft + tRight) / 2 + 0.01);
  [leftIndex, leftBest] = maxProbs(accumProbs, leftChild, tLeft, tMid);
  [rightIndex, rightBest] = maxProbs(accumProbs, rightChild, tMid+1, tRight);
  if leftBest >= rightBest
    wordIndex = leftIndex;
    bestValue = leftBest;
  else
    wordIndex = rightIndex;
    bestValue = rightBest;
  end