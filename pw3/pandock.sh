#!/bin/bash

FILE_NAME=report.md
OUT_FILE=report.pdf

docker run --rm \
        --volume "$(pwd):/data" \
        --user $(id -u):$(id -g) \
        pandoc/extra $FILE_NAME -o $OUT_FILE --template eisvogel