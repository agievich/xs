XS-circuits
===========

What is XS-circuits?
--------------------

XS-circuits describe block ciphers that utilize 2 operations: 
X) bitwise modulo 2 addition of binary words and 
S) substitution of words using key-dependent S-boxes 
with possibly complicated internal structure.

In the paper [XS-circuits in block ciphers](https://eprint.iacr.org/2018/592) 
we propose a model of XS-circuits which, despite the simplicity, covers 
rather wide range of block ciphers. In this model, several instances of a 
simple round circuit, which contains only one S operation, are linked 
together and form a compound circuit called a cascade. S-operations of a 
cascade are interpreted as independent round oracles. A round circuit is 
described by a binary matrix, called an extended matrix of the circuit.

What is this repo?
------------------

This repo supports the mentioned paper. 

First, we present extended matrices of several well-known circuits (see [data](data)). 

Second, we provide the Python script [xs.py](prg/xs.py) which calculates 
various characteristics of a given round XS-circuit specified by its extended matrix.

Third, the script [gna.py](prg/gna.py) implements an algorithm for calculating 
the guaranteed number of activations in a given cascade. This number relates 
to security against differential and linear attacks. Details will be provided 
in the forthcoming paper.
