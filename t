#!/bin/sh

# python init.py

# for i in `seq  100` ; do
#     python self_play_1.py
#     python train.py
#     python evaluate.py
# done

# ls data/play | xargs -I {} rm data/play/{}

for i in `seq 10000` ; do
    python ./trainer/self_play_2.py
    python ./trainer/train.py
    python ./trainer/evaluate.py
done
