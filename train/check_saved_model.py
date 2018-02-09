#!/usr/bin/env python
import sys
import tensorflow as tf
from tensorflow.python.framework import meta_graph
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import meta_graph_pb2

def display_saved_model():
    #meta_graph.read_meta_graph_file("/tmp/stem/model-20180208-211356/ckpt.meta")
    #g = tf.MetaGraphDef()
    #g.ParseFromString(open("/tmp/stem/model-20180208-211356/ckpt.meta", "rb").read())
    #print("GraphDef from meta_graph_file:", g.graph_def)
    #g = tf.GraphDef()
    #g.ParseFromString(open("/tmp/stem/model-20180208-211356/model.pbtxt", "rb").read())
    #print("GraphDef: ", g)
    #[n for n in g.node if n.name.find("input") != -1] # same for output or any other node you want to make sure is ok

    saved_model = saved_model_pb2.SavedModel()
    saved_model.ParseFromString(open("/tmp/stem/export-current/saved_model.pb", "rb").read())
    print("saved_model parsed", saved_model.saved_model_schema_version, len(saved_model.meta_graphs))
    print("GraphDef from SavedModel file:", saved_model.meta_graphs[0].graph_def)

def display_frozen_model():
    g = tf.GraphDef()
    g.ParseFromString(open("/tmp/stem/export-current/sound_model.pb", "rb").read())
    print("GraphDef: ", g)

def main():
#    display_saved_model()
    display_frozen_model()

if __name__ == "__main__":
    main()
