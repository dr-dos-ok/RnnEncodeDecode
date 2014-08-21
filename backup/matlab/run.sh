#!/bin/bash

if [[ $# -ne 9 ]]; then
  echo "`basename $0` trainSize validSize embeddingSize hiddenSize numEpoches learningRate numThreads minibatchSize tgtFeatureSize"
  exit
fi

trainSize=$1
validSize=$2
embeddingSize=$3
hiddenSize=$4
numEpoches=$5
learningRate=$6
numThreads=$7
minibatch=$8
tgtFeatureSize=$9

dataPath="../data"
tgtLang="en"
tgtFile="data.$trainSize.$tgtLang"

head -$trainSize $dataPath/data.full.$tgtLang > $dataPath/data.$trainSize.$tgtLang

javac PrepareData.java
java PrepareData -dataPath $dataPath -corpusFile $tgtFile -language $tgtLang 
rm *.class

tail -$validSize $dataPath/data.$trainSize.$tgtLang.int > $dataPath/data.$trainSize.valid.$tgtLang.int
head -$((trainSize - validSize)) $dataPath/data.$trainSize.$tgtLang.int > $dataPath/temp
rm $dataPath/data.$trainSize.$tgtLang.int
mv $dataPath/temp $dataPath/data.$trainSize.$tgtLang.int

echo "train('dataPath=$dataPath,tgtFile=$tgtFile,tgtLang=$tgtLang,embeddingSize=$embeddingSize,hiddenSize=$hiddenSize,numEpoches=$numEpoches,learningRate=$learningRate,tgtValid=data.$trainSize.valid.$tgtLang.int,numThreads=$numThreads,minibatch=$minibatch,tgtFeatureSize=$tgtFeatureSize')"
matlab -nodesktop -nodisplay -nosplash -r "train('dataPath=$dataPath,tgtFile=$tgtFile,tgtLang=$tgtLang,embeddingSize=$embeddingSize,hiddenSize=$hiddenSize,numEpoches=$numEpoches,learningRate=$learningRate,tgtValid=data.$trainSize.valid.$tgtLang.int,numThreads=$numThreads,minibatch=$minibatch,tgtFeatureSize=$tgtFeatureSize'); exit();"
# NLP cluster versions
# ~/./matlab_r2013b -nodesktop -nodisplay -nosplash -r "train('dataPath=$dataPath,tgtFile=$tgtFile,tgtLang=$tgtLang,embeddingSize=$embeddingSize,hiddenSize=$hiddenSize,numEpoches=$numEpoches,learningRate=$learningRate,tgtValid=data.$trainSize.valid.$tgtLang.int,numThreads=$numThreads,minibatch=$minibatch,tgtFeatureSize=$tgtFeatureSize'); exit();"

rm $dataPath/data.$trainSize*
