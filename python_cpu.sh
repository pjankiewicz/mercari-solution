#!/usr/bin/env bash
docker run \
    -v $PWD:/tmp/working \
    -v $PWD:/tmp/ \
    -v $PWD/input:/tmp/input \
    -w=/tmp/working \
    --rm -it --memory 16g --cpus 4 kaggle-python-cpu \
    bash -c "python setup.py develop && /usr/bin/time -v python $*"
