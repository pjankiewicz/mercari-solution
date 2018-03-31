#!/usr/bin/env bash
docker run -v $PWD:/tmp/working -w=/tmp/working -p 8890:8888 \
    --memory 16g --cpus 4 --rm -it kaggle-python-cpu \
    jupyter notebook --no-browser --ip="0.0.0.0" --notebook-dir=/tmp/working --allow-root
