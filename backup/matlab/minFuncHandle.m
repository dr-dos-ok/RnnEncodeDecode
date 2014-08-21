function [cost, grad] = minFuncHandle(p, tgtSents, model)
  actualModel = vector2model(p, model);
  [cost, modelGrad] = trainOneBatch(actualModel, tgtSents);
  grad = model2vector(modelGrad);