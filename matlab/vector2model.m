function [actualModel] = vector2model(p, model)
  embeddingSize = model.embeddingSize;
  hiddenSize = model.hiddenSize;
  tgtVocabSize = model.tgtVocabSize;
  tgtFeatureSize = model.tgtFeatureSize;
  tgtTreeSize = model.tgtTreeSize;
  
  lower = 1;
  upper = hiddenSize * embeddingSize;
  
  actualModel.U = reshape(p(lower : upper), hiddenSize, embeddingSize);
  lower = upper + 1;
  upper = upper + hiddenSize * hiddenSize;
  actualModel.R = reshape(p(lower : upper), hiddenSize, hiddenSize);
  lower = upper + 1;
  upper = upper + tgtFeatureSize * hiddenSize;
  actualModel.W = reshape(p(lower : upper), tgtFeatureSize, hiddenSize);
  lower = upper + 1;
  upper = upper + embeddingSize * tgtVocabSize;
  actualModel.tgtWe = reshape(p(lower : upper), embeddingSize, tgtVocabSize);
  lower = upper + 1;
  upper = upper + tgtFeatureSize * tgtTreeSize;
  actualModel.tgtTree = reshape(p(lower : upper), tgtFeatureSize, tgtTreeSize);
  
  actualModel.embeddingSize = model.embeddingSize;
  actualModel.hiddenSize = model.hiddenSize;
  actualModel.tgtVocabSize = model.tgtVocabSize;
  actualModel.tgtFeatureSize = model.tgtFeatureSize;
  actualModel.tgtTreeSize = model.tgtTreeSize;
  actualModel.tgtWordPaths = model.tgtWordPaths;