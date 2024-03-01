#!/bin/bash

IMAGE=percept_vsvi_filter

docker build -t "$IMAGE" \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  .
