#!/bin/bash

while read wav_file_name; do
    arff=${wav_file_name##*/}
    arff="${arff%%\.*}.arff"
    echo $arff
    #opensmile-2.3.0/SMILExtract -C opensmile-2.3.0/config/IS13_ComParE.conf \
    #                            -I $wav_file_name \
    #                            -O /home/kelvin/c487/data/features/$arff
done < ~/c487/data/labels/wav_files.txt
