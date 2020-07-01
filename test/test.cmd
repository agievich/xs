@echo off
for %%f in (..\data\*.txt) do (
  python ..\prg\xs.py %%f
  echo.
)

python ..\prg\gna.py --F2 ..\data\gfn1-4.txt 15
python ..\prg\gna.py ..\data\gfn1-4.txt 15