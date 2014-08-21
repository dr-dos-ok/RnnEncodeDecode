function [result] = randomMatrix(rangeSize, size)
  result = 2*rangeSize * (rand(size) - 0.5);