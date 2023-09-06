#!/bin/bash

pip install -r requirements.txt

mkdir "experiments/"
cp -r "dataset/" "experiments/"
cp -r "route_info/" "experiments/"
cd "experiments/"
for (( n=1 ; n<=30; n++ ))
do
    cp -r "../GA" "./GA_${n}"
    echo "EXECUTION ${n}"
    cd "./GA_${n}"
    python3 LSTMGAMain.py
    cd "../"
done
