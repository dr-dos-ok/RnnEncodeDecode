function [startIndex, endIndex] = getClassRange(model, classIndex)
  if classIndex ~= 1
    startIndex = model.auxV(classIndex - 1) + 1;
  else
    startIndex = 1;
  end
  endIndex = model.auxV(classIndex);