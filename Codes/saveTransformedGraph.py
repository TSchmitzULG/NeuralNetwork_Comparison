#!/usr/bin/python
# -*- coding: latin-1 -*-
"""freeze optimize and transform graph"""

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
import tensorflow as tf


def optimizeGraph(pathTemp,input_nodes,output_nodes):
    # freeze graph
    input_graph_path = "{}/".format(pathTemp)+'myFinalGraph.pbtxt'
    checkpoint_path = "{}/".format(pathTemp)+'myFinalModel.ckpt'
    input_saver_def_path = ""
    input_binary = False
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_frozen_graph_name = "{}/".format(pathTemp)+'frozenModel.pb'
    # output_optimized_graph_name = 'optimized_'+MODEL_NAME+'.pb'
    clear_devices = True
    output_optimized_graph_name ="{}/".format(pathTemp)+'optimizedModel.pb'
    output_transformed_graph_name ="{}/".format(pathTemp)+'transformedModel.pb'
    
    
    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_nodes[0],
                          restore_op_name, filename_tensor_name,
                          output_frozen_graph_name, clear_devices, "")
    # Optimize for inference    
    with tf.gfile.Open(output_frozen_graph_name, "rb") as f:
        input_graph_def = tf.GraphDef()
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            input_nodes, # an array of the input node(s)
            output_nodes, # an array of output nodes
            tf.float32.as_datatype_enum)
    # Save the optimized graph
    f = tf.gfile.FastGFile(output_optimized_graph_name, "wb")
    f.write(output_graph_def.SerializeToString())
    
    #graph transform
    with tf.gfile.Open(output_optimized_graph_name, "rb") as f:
        input_graph_def = tf.GraphDef()
        input_graph_def.ParseFromString(f.read())
    transforms = ["strip_unused_nodes"]
    transformed_graph_def = TransformGraph(
        input_graph_def,
        input_nodes,
        output_nodes,
        transforms)
    # Save the transformed graph
    f = tf.gfile.FastGFile(output_transformed_graph_name, "wb")
    f.write(transformed_graph_def.SerializeToString())
    return 



