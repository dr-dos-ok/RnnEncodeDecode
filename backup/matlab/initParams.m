function [model] = initParameter(params)
  fprintf('# Initializing parameters...\n');
  
  model.embeddingSize = params.embeddingSize;
  model.hiddenSize = params.hiddenSize;
  
  [model.tgtWords, model.tgtWordPaths] = loadDictionary(params);
  model.tgtFeatureSize = params.tgtFeatureSize;
  model.tgtVocabSize= length(model.tgtWords);
  model.tgtTreeSize = max(vertcat(model.tgtWordPaths{:}));
  
  model.U = rand(model.hiddenSize, model.embeddingSize) - 0.5;
  model.R = rand(model.hiddenSize) - 0.5;
  model.W = rand(model.tgtFeatureSize, model.hiddenSize) - 0.5;
  model.tgtTree= rand(model.tgtFeatureSize, model.tgtTreeSize) - 0.5;
  model.tgtWe = rand(model.embeddingSize, model.tgtVocabSize) - 0.5;
  
  fileID = fopen('temp', 'w');
  for i = 1 : length(model.tgtWordPaths)
    fprintf(fileID, '%s ', model.tgtWords{i});
    for j = 1 : length(model.tgtWordPaths{i})
      fprintf(fileID, '%d ', model.tgtWordPaths{i}(j));
    end
    fprintf(fileID, '\n');
  end
  fclose(fileID);
  
  fprintf('done\n');