@echo off
for %%f in (..\data\*.txt) do (
  python ..\prg\xs.py %%f
  echo(
)