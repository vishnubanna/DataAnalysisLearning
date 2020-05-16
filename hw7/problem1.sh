#!/usr/bin/env bash
INFILE=$(pwd)/test.in
HOLDFILE=$(pwd)/hold
CMD1FILE=$(pwd)/cmd1
CMD2FILE=$(pwd)/cmd2

python3 $CMD1FILE < $INFILE > $HOLDFILE
python3 $CMD2FILE < $HOLDFILE
rm $HOLDFILE



