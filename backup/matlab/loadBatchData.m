function [sents, numSents] = loadBatchData(fid, baseIndex, batchSize)
  sents = cell(batchSize, 1);
  numSents = 0;
  while (~feof(fid) && numSents<batchSize)
    indices = sscanf(fgetl(fid), '%d') + (1-baseIndex);
    if ~isempty(indices) % ignore empty lines
      numSents = numSents + 1;
      sents{numSents} = indices;
    end
  end
  sents((numSents+1):end) = []; % delete empty cells
end
