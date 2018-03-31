#!/usr/bin/env bash
nvidia-docker run \
    -v $PWD:/tmp/working \
    -v $PWD:/tmp/ \
    -v $PWD/input:/tmp/input \
    -w=/tmp/working \
    --rm -it kaggle-python-gpu \
    bash -c "python setup.py develop && python $*"
