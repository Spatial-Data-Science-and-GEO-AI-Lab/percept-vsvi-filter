#!/bin/bash

IMAGE=percept_vsvi_filter

docker run --gpus 1 --net=host -it --rm \
       -e XAUTHORITY=$HOME/.Xauthority \
       -e DISPLAY="$DISPLAY" \
       -v $PWD:/work \
       -v $PWD/.huggingface:/home/user/.cache/huggingface \
       -v "$HOME":"$HOME":ro \
       -u `id -u`:`id -g` \
       "$IMAGE" $*
