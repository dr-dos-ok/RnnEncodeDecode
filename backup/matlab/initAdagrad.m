function [adagrad] = initAdagrad(model)
  adagrad.EPS = 0.01;
  adagrad.learningRate = 1.0;
  adagrad.U = zeros(size(model.U));
  adagrad.R = zeros(size(model.R));
  adagrad.W = zeros(size(model.W));
  adagrad.tgtWe = zeros(size(model.tgtWe));
  adagrad.tgtTree = zeros(size(model.tgtTree));
