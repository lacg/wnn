#!/usr/bin/env zsh

source wnn/bin/activate
pip install -e src/wnn
export PYTHONPATH="$(pwd)/src/wnn:$PYTHONPATH"