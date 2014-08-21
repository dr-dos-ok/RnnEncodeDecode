function train(opts)
  addpath(genpath(pwd));
  rand('seed', 21260063);
  params = parseOpts(opts);
  model = initParams(params);
  
  params
  model
  if params.isProfile
    profile on;
  end
  if params.numThreads ~= 0
    matlabpool('open', params.numThreads);
  end
  
  beta = params.regularization;
  srcFile = sprintf('%s/%s.int', params.dataPath, params.srcFile);
  tgtFile = sprintf('%s/%s.int', params.dataPath, params.tgtFile);
  fprintf('# Start training using srcFile "%s" and tgtFile "%s"', srcFile, tgtFile);
  for epoch = 1 : params.numEpoches
    srcID = fopen(srcFile, 'r');
    tgtID = fopen(tgtFile, 'r');

    lineCounts = 0;
    totalCost = 0;
    startTime = clock;
    while ~feof(srcID)
      alpha = params.learningRate / (1 + params.learningRate * (epoch-1));
      [srcSents, srcNumSents] = loadBatchData(srcID, 1, params.minibatch);
      [tgtSents, tgtNumSents] = loadBatchData(tgtID, 1, params.minibatch);
      
      [cost, grad] = trainOneBatch(model, srcSents, tgtSents);
      totalCost = totalCost + cost;
      
%       theta = model2vector(model);
%       options.Method = 'lbfgs';
%       options.DerivativeCheck = 'off';
%       options.display = 'off'; % 'on'; %
%       options.maxIter = 3;
%       [theta, cost, ~, ~] = minFunc( @(p) minFuncHandle(p, tgtSents, model), theta, options);
%       model = vector2model(theta, model);
%       totalCost = totalCost + cost;
      
      % gradient descent update
      model.srcU = (1 - beta) * model.srcU - alpha * grad.srcU;
      model.srcR = (1 - beta) * model.srcR - alpha * grad.srcR;
      model.srcW = (1 - beta) * model.srcW - alpha * grad.srcW;
      model.srcWe = (1 - beta) * model.srcWe - alpha * grad.srcWe;
      model.srcTree = (1 - beta) * model.srcTree - alpha * grad.srcTree;
      
      model.tgtS = (1 - beta) * model.tgtS - alpha * grad.tgtS;
      
      model.tgtU = (1 - beta) * model.tgtU - alpha * grad.tgtU;
      model.tgtR = (1 - beta) * model.tgtR - alpha * grad.tgtR;
      model.tgtW = (1 - beta) * model.tgtW - alpha * grad.tgtW;
      model.tgtWe = (1 - beta) * model.tgtWe - alpha * grad.tgtWe;
      model.tgtTree = (1 - beta) * model.tgtTree - alpha * grad.tgtTree;
    
%       if epoch >= 5
%       % adagrad
%         adagrad.U = adagrad.U + (grad.U) .^ 2;
%         adagrad.R = adagrad.R + (grad.R) .^ 2;
%         adagrad.W = adagrad.W + (grad.W) .^ 2;
%         adagrad.tgtWe = adagrad.tgtWe + (grad.tgtWe) .^ 2;
%         adagrad.tgtTree = adagrad.tgtTree + (grad.tgtTree) .^ 2;
%         
%         model.U = model.U - adagrad.learningRate * (grad.U ./ (sqrt(adagrad.U) + adagrad.EPS));
%         model.R = model.R - adagrad.learningRate * (grad.R ./ (sqrt(adagrad.R) + adagrad.EPS));
%         model.W = model.W - adagrad.learningRate * (grad.W ./ (sqrt(adagrad.W) + adagrad.EPS));
%         model.tgtWe = model.tgtWe - adagrad.learningRate * (grad.tgtWe ./ (sqrt(adagrad.tgtWe) + adagrad.EPS));
%         model.tgtTree = model.tgtTree - adagrad.learningRate * (grad.tgtTree ./ (sqrt(adagrad.tgtTree) + adagrad.EPS));
% %       [max(grad.U(:)) max(grad.R(:)) max(grad.W(:)) max(grad.tgtWe(:)) max(grad.tgtTree(:))]
%       else
%       % gradient descent
%         model.U = model.U - alpha * grad.U;
%         model.R = model.R - alpha * grad.R;
%         model.W = model.W - alpha * grad.W;
%         model.tgtWe = model.tgtWe - alpha * grad.tgtWe;
%         model.tgtTree = model.tgtTree - alpha * grad.tgtTree;
%       end

      lineCounts = lineCounts + tgtNumSents;
      if lineCounts > params.logEvery
        if params.isProfile
          profile viewer
        end
        fprintf('.');
        lineCounts = 0;
      end
      
      filename = sprintf('%s/model.%d.%d.%d.%d.%d.mat', params.dataPath, params.srcEmbeddingSize, params.srcHiddenSize, ...
                                                                         params.tgtEmbeddingSize, params.tgtHiddenSize, epoch);
      saveModel(filename, model);
    end
    
    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    [srcPPL, tgtPPL] = getPerplexity(params, model);
    fprintf('\n## Epoch %d:\tObjective: %f\tElapsed: %.2f seconds\tValid %s PPL: %.2f\tValid %s PPL: %.2f', ...
                                epoch, totalCost, timeElapsed, params.srcLang, srcPPL, params.tgtLang, tgtPPL);
    
    fclose(tgtID);
    fclose(srcID);
  end
  
  fprintf('\ndone\n');

function [params] = parseOpts(opts)
  params.dataPath = '';
  params.srcFile = '';
  params.srcLang = 'fr';
  
  params.tgtFile = '';
  params.tgtLang = 'en';
  
  params.srcValid = '';
  params.tgtValid = '';
  params.srcFeatureSize = 25;
  params.tgtFeatureSize = 25;

  params.numThreads = 0;
  params.isProfile = 0;
  params.numEpoches = 7;
  params.learningRate = 1e-2;
  params.regularization = 1e-2;
  params.minibatch = 100;
  params.logEvery = 1000;
  
  params.srcEmbeddingSize = 30;
  params.srcHiddenSize = 60;
  params.tgtEmbeddingSize = 30;
  params.tgtHiddenSize = 60;
  
  tokens = strsplit(opts, ',');
  if length(tokens) == 1
    tokens = tokens{1};
  end
  
  for i = 1 : length(tokens)
    [key, value] = strtok(tokens{i}, '=');
    if isempty(value)
      params.(key) = 1;
    else
      value = value(2 : end);
      switch key
        case 'dataPath'
          params.dataPath = value;
        case 'srcFile'
          params.srcFile = value;
        case 'srcLang'
          params.srcLang = value;
        case 'tgtFile'
          params.tgtFile = value;
        case 'tgtLang'
          params.tgtLang = value;
        case 'srcValid'
          params.srcValid = value;
        case 'tgtValid'
          params.tgtValid = value;
        case 'srcFeatureSize'
          params.srcFeatureSize = str2double(value);
        case 'tgtFeatureSize'
          params.tgtFeatureSize = str2double(value);
        case 'numEpoches'
          params.numEpoches = str2double(value);
        case 'numThreads'
          params.numThreads = str2double(value);
        case 'learningRate'
          params.learningRate = str2double(value);
        case 'regularization'
          params.regularization = str2double(value);
        case 'minibatch'
          params.minibatch = str2double(value);
        case 'logEvery'
          params.logEvery = str2double(value);
        case 'srcEmbeddingSize'
          params.srcEmbeddingSize = str2double(value);
        case 'srcHiddenSize'
          params.srcHiddenSize = str2double(value);
        case 'tgtEmbeddingSize'
          params.tgtEmbeddingSize = str2double(value);
        case 'tgtHiddenSize'
          params.tgtHiddenSize = str2double(value);
        case 'numClasses'
          params.numClasses = str2double(value);
        otherwise
          errorMessage = sprintf('Unknown argument %s', key);
          error(errorMessage);
      end
    end
  end
