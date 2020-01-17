#!/bin/bash

find ~/c487/data/features/arffs/ -name "*\.arff" > tmp

while read arff; do
    arff=${arff##*/}
    csv="${arff%%\.*}.csv"
    echo $csv
    proj_code/arff2csv.py -i ~/c487/data/features/arffs/$arff -o ~/c487/data/features/csvs/$csv
done < tmp | sort

rm tmp
