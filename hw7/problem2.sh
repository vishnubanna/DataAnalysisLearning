#!/usr/bin/env bash
OUTFILE1=$OUTFILE.out 
ERRFILE=$OUTFILE.err
INFILE=$INFILE
echo $OUTFILE1
echo $ERRFILE

chmod +x cmd1
chmod +x cmd3
touch $(pwd)/$OUTFILE1
touch $(pwd)/$ERRILE
./cmd3 < ./cmd1 < $INFILE 1> $OUTFILE 2> $ERRFILE 