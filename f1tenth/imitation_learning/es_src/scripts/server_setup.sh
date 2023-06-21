#!/usr/bin/env bash

# path: the same directory as es_src
python3 ./es_src/configs/config_generator.py
pip install -e ./es_src/
cd ./es_src/scripts && python3 ./experiments.py with default