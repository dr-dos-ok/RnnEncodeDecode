#!bin/bash

if [[ $# -ne 3 ]]; then
  echo "`basename $0` dataPath corpusFile language"
  exit
fi

dataPath=$1
corpusFile=$2
language=$3

javac PrepareData.java
java PrepareData -dataPath $dataPath -corpusFile $corpusFile -language $language
rm *.class
