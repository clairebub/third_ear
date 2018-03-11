#!/bin/sh

TOOLS_DIR=${HOME}/Downloads/tensorflow/tensorflow/python/tools

model_dir="/tmp/v_ear/export-20180310-190433"
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
  --output_graph=${HOME}/frozen_vear_model.pb \
  --output_node_names="X,Y,w,b,w_1,b_1,w_2,b_2,pred"
