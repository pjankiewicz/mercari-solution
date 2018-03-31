#!/usr/bin/env bash
docker run -v $PWD:/tmp/working -w=/tmp/working --rm -it \
    --memory 16g --cpus 4 kaggle/python ipython
