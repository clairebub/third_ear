#!/bin/sh

TOOLS_DIR=/Users/rjtang/_hack/tensorflow/tensorflow/python/tools

cd ${TOOLS_DIR}
pwd
python freeze_graph.py \
  --input_graph=/tmp/tfdriod/tfdroid.pbtxt \
  --input_binary=false \
  --input_checkpoint=/tmp/tfdriod/tfdroid.ckpt \
  --output_graph=/tmp/frozen_graph.pb \
  --output_node_names=W
