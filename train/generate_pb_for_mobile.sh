#!/bin/sh

TOOLS_DIR=${HOME}/Downloads/tensorflow/tensorflow/python/tools

model_dir="/tmp/stem/export-current"
echo "model_dir=${model_dir}"

cd ${TOOLS_DIR}
pwd
#python freeze_graph.py \
#  --input_graph=$model_dir/model.pbtxt \
#  --input_binary=false \
#  --input_checkpoint=/tmp/stem/model-20180208-211356/ckpt \
#  --output_graph=/tmp/frozen_graph.pb \
#  --output_node_names=W

python freeze_graph.py \
  --input_saved_model_dir=${model_dir} \
  --saved_model_tags="train" \
  --output_graph=${model_dir}/sound_model.pb \
  --output_node_names="Variable_4"
