#!/bin/bash

find IEMOCAP_full_release/Session* -name "Categorical" | grep "dialog/EmoEvaluation" > tmp

while read labels; do
    echo $labels
    cp -r "${labels}/" labels/
done < tmp

rm tmp

