#!/bin/bash

if [[ $# -ne 12 ]]; then
  echo "`basename $0` trainSize validSize srcEmbeddingSize srcHiddenSize tgtEmbeddingSize tgtHiddenSize numEpoches learningRate numThreads minibatchSize srcFeatureSize tgtFeatureSize"
  exit
fi

trainSize=$1
validSize=$2
srcEmbeddingSize=$3
srcHiddenSize=$4
tgtEmbeddingSize=$5
tgtHiddenSize=$6
numEpoches=$7
learningRate=$8
numThreads=$9
minibatch=${10}
srcFeatureSize=${11}
tgtFeatureSize=${12}

dataPath="../data"
srcLang="fr"
srcFile="data.$trainSize.$srcLang"
tgtLang="en"
tgtFile="data.$trainSize.$tgtLang"

head -$trainSize $dataPath/data.full.$srcLang > $dataPath/$srcFile
head -$trainSize $dataPath/data.full.$tgtLang > $dataPath/$tgtFile

javac PrepareData.java
java PrepareData -dataPath $dataPath -corpusFile $srcFile -language $srcLang
java PrepareData -dataPath $dataPath -corpusFile $tgtFile -language $tgtLang 
rm *.class

tail -$validSize $dataPath/data.$trainSize.$srcLang.int > $dataPath/data.$trainSize.valid.$srcLang.int
head -$((trainSize - validSize)) $dataPath/data.$trainSize.$srcLang.int > $dataPath/temp
rm $dataPath/data.$trainSize.$srcLang.int
mv $dataPath/temp $dataPath/data.$trainSize.$srcLang.int


tail -$validSize $dataPath/data.$trainSize.$tgtLang.int > $dataPath/data.$trainSize.valid.$tgtLang.int
head -$((trainSize - validSize)) $dataPath/data.$trainSize.$tgtLang.int > $dataPath/temp
rm $dataPath/data.$trainSize.$tgtLang.int
mv $dataPath/temp $dataPath/data.$trainSize.$tgtLang.int

matlabCommand="train('dataPath=$dataPath,srcFile=$srcFile,tgtFile=$tgtFile,srcLang=$srcLang,tgtLang=$tgtLang,srcEmbeddingSize=$srcEmbeddingSize,tgtEmbeddingSize=$tgtEmbeddingSize,srcHiddenSize=$srcHiddenSize,tgtHiddenSize=$tgtHiddenSize,numEpoches=$numEpoches,learningRate=$learningRate,srcValid=data.$trainSize.valid.$srcLang.int,tgtValid=data.$trainSize.valid.$tgtLang.int,numThreads=$numThreads,minibatch=$minibatch,srcFeatureSize=$srcFeatureSize,tgtFeatureSize=$tgtFeatureSize')"

echo "$matlabCommand"
matlab -nodesktop -nodisplay -nosplash -r "$matlabCommand ; exit()"
# NLP cluster versions
# ~/./matlab_r2013b -nodesktop -nodisplay -nosplash -r "$matlabCommand ; exit()"

rm $dataPath/data.$trainSize*
