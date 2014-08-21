function saveModel(params, model)
  path = sprintf('%s/model.mat', params.dataPath);
  save(path, 'model');