#!/usr/bin/env python

# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf

ckpt_path = "./model/urban_sound/model.ckpt-20"

# print all tensors in checkpoint file
chkp.print_tensors_in_checkpoint_file(
    ckpt_path,
    tensor_name='',
    all_tensors=False,
    all_tensor_names=True)

# tensor_name:  v1
# [ 1.  1.  1.]
# tensor_name:  v2
# [-1. -1. -1. -1. -1.]

# print only tensor v1 in checkpoint file
#chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v1', all_tensors=False)

# tensor_name:  v1
# [ 1.  1.  1.]

# print only tensor v2 in checkpoint file
#chkp.print_tensors_in_checkpoint_file("/tmp/model.ckpt", tensor_name='v2', all_tensors=False)

# tensor_name:  v2
# [-1. -1. -1. -1. -1.]
