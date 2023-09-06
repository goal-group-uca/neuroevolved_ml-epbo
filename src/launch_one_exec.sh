#!/bin/bash

pip install -r requirements.txt
cd "./GA"
python3 LSTMGAMain.py
cd "../"
