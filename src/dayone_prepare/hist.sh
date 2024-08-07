#!/usr/bin/env bash

cat all.txt \
| awk '{h[length($0)]+=1} END{for (i in h) print i, ", ", h[i]}' \
> hist.csv
