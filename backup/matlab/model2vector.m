function [p] = model2vector(model)
  p = [model.U(:) ; model.R(:) ; model.W(:) ; model.tgtWe(:) ; model.tgtTree(:)];