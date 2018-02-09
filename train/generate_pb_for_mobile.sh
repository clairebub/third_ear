#!/bin/sh

TOOLS_DIR=/Users/rjtang/_hack/tensorflow/tensorflow/python/tools

model_dir=/tmp/stem/model-20180208-211356
echo "model_dir=$model_dir"

cd ${TOOLS_DIR}
pwd
python freeze_graph.py \
  --input_graph=$model_dir/model.pbtxt \
  --input_binary=false \
  --input_checkpoint=/tmp/stem/model-20180208-211356/ckpt \
  --output_graph=/tmp/frozen_graph.pb \
  --output_node_names=W

#python freeze_graph.py \
#  --input_saved_model_dir=$model_dir \
#  --output_graph=sound_model.pb \
#  --output_node_names=W
