function [model] = initParameter(params)
  fprintf('# Initializing parameters...\n');
  
  model.srcEmbeddingSize = params.srcEmbeddingSize;
  model.srcHiddenSize = params.srcHiddenSize;
  model.tgtEmbeddingSize = params.tgtEmbeddingSize;
  model.tgtHiddenSize = params.tgtHiddenSize;
  
  [model.srcWords, model.srcWordPaths, model.srcWordDirs] = loadDictionary(params, 'src', 0);
  model.srcFeatureSize = params.srcFeatureSize;
  model.srcVocabSize = length(model.srcWords);
  model.srcTreeSize = max(vertcat(model.srcWordPaths{:}));
  
  [model.tgtWords, model.tgtWordPaths, model.tgtWordDirs] = loadDictionary(params, 'tgt', 0);
  model.tgtFeatureSize = params.tgtFeatureSize;
  model.tgtVocabSize = length(model.tgtWords);
  model.tgtTreeSize = max(vertcat(model.tgtWordPaths{:}));
  
  initRange = 1.0;
  model.srcU = randomMatrix(initRange, [model.srcHiddenSize, model.srcEmbeddingSize]);
  model.srcR = randomMatrix(initRange, [model.srcHiddenSize]);
  model.srcW = randomMatrix(initRange, [model.srcFeatureSize, model.srcHiddenSize]);
  model.srcTree= randomMatrix(initRange, [model.srcFeatureSize, model.srcTreeSize]);
  model.srcWe = randomMatrix(initRange, [model.srcEmbeddingSize, model.srcVocabSize]);
  
  model.tgtU = randomMatrix(initRange, [model.tgtHiddenSize, model.tgtEmbeddingSize]);
  model.tgtR = randomMatrix(initRange, [model.tgtHiddenSize]);
  model.tgtS = randomMatrix(initRange, [model.tgtHiddenSize, model.srcHiddenSize]);
  model.tgtW = randomMatrix(initRange, [model.tgtFeatureSize, model.tgtHiddenSize]);
  model.tgtTree= randomMatrix(initRange, [model.tgtFeatureSize, model.tgtTreeSize]);
  model.tgtWe = randomMatrix(initRange, [model.tgtEmbeddingSize, model.tgtVocabSize]);
  
  fprintf('done\n');
