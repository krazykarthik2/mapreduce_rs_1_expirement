#!/bin/bash

INPUT=$1
PARTITIONS=$2
OUTDIR=output

echo "=============================="
echo "MapReduce Benchmark Runner"
echo "Input File: $INPUT"
echo "Partitions: $PARTITIONS"
echo "=============================="

rm -rf $OUTDIR
mkdir -p $OUTDIR

START=$(date +%s%N)

cargo run -p mapreduce_cli --release -- \
    wordcount \
    --input $INPUT \
    --output $OUTDIR \
    --partitions $PARTITIONS

END=$(date +%s%N)

DIFF=$((END - START))

MS=$((DIFF / 1000000))
SEC=$((DIFF / 1000000000))

echo "=============================="
echo "Execution Time:"
echo "$SEC sec  ($MS ms)"
echo "=============================="