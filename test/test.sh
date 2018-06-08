#!/bin/bash
# echo "###############################"

for f in ../data/*.txt
do
  python ../prg/xs.py $f
  echo ""
done
