#!/bin/bash
# echo "###############################"

for f in ../data/*.txt
do
  python3 ../prg/xs.py $f
  echo ""
done

python3 ../prg/gna.py --lrs ../data/gfn1-4.txt 15
echo ""
python3 ../prg/gna.py ../data/gfn1-4.txt 15