function [treeIndex] = buildTree(node, left, right, wordIndex)
  if left >= right
    treeIndex = node;
  else
    leftChild = 2 * node;
    rightChild = 2 * node + 1;
    mid = (left + right) / 2;
    if wordIndex <= mid
      treeIndex = buildTree(leftChild, left, mid, wordIndex);
    else
      treeIndex = buildTree(rightChild, mid+1, right, wordIndex);
    end
  end