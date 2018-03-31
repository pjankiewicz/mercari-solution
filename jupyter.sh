#!/usr/bin/env bash
nvidia-docker run -v $PWD:/tmp/working -w=/tmp/working -p 8889:8888 --rm -it \
    kaggle-python-gpu \
    jupyter notebook --no-browser --ip="0.0.0.0" \
    --notebook-dir=/tmp/working --allow-root
