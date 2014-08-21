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
  tgtFile = sprintf('%s/%s.int', params.dataPath, params.tgtFile);
  fprintf('# Start training using file "%s"', tgtFile);
  for epoch = 1 : params.numEpoches
    tgtID = fopen(tgtFile, 'r');

    lineCounts = 0;
    totalCost = 0;
    startTime = clock;
    while ~feof(tgtID)
      alpha = params.learningRate / (1 + params.learningRate * (epoch-1));
      [tgtSents, tgtNumSents] = loadBatchData(tgtID, 1, params.minibatch);
      if length(tgtSents{1}) < 2
        continue;
      end
      [cost, grad] = trainOneBatch(model, tgtSents);
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
      model.U = (1 - beta) * model.U - alpha * grad.U;
      model.R = (1 - beta) * model.R - alpha * grad.R;
      model.W = (1 - beta) * model.W - alpha * grad.W;
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
      
      saveModel(params, model);
    end
    
    endTime = clock;
    timeElapsed = etime(endTime, startTime);
    ppl = getPerplexity(params, model);
    fprintf('\n## Epoch %d:\tObjective: %f\tElapsed: %.2f seconds\tValid perplexity: %.2f', epoch, totalCost, timeElapsed, ppl);
  end
  
  fprintf('\ndone\n');
  
  fclose(tgtID);

function [params] = parseOpts(opts)
  params.dataPath = '';
  params.tgtFile = '';
  params.tgtLang = 'en';
  params.tgtValid = '';
  params.tgtFeatureSize = 25;

  params.numThreads = 0;
  params.isProfile = 0;
  params.numEpoches = 7;
  params.learningRate = 1e-2;
  params.regularization = 1e-2;
  params.minibatch = 100;
  params.logEvery = 1000;
  
  params.embeddingSize = 30;
  params.hiddenSize = 60;
  params.numClasses = 100;
  
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
        case 'tgtFile'
          params.tgtFile = value;
        case 'tgtLang'
          params.tgtLang = value;
        case 'tgtValid'
          params.tgtValid = value;
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
        case 'embeddingSize'
          params.embeddingSize = str2double(value);
        case 'hiddenSize'
          params.hiddenSize = str2double(value);
        case 'numClasses'
          params.numClasses = str2double(value);
        otherwise
          errorMessage = sprintf('Unknown argument %s', key);
          error(errorMessage);
      end
    end
  end
