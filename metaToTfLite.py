__author__ = "Majaja <r97922513@gmail.com>"
__license__ = 'MIT License. See LICENSE.'

import tensorflow as tf
import sys
from subprocess import call
from tensorflow.python.platform import gfile
import json

fixPbName = '_fix.pb'

def fixPb(pb_path):
# read graph definition
    f = gfile.FastGFile(pb_path, "rb")
    gd = graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    for node in graph_def.node:
        if node.op == 'RefSwitch':
            print(node.name, "\n")
            node.op = 'Switch'
            for index in xrange(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            print(node.name, "\n")
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

    tf.import_graph_def(graph_def, name='')
	
    tf.train.write_graph(graph_def, './', pb_path, as_text=False)

def toPb(meta_path, model_name, pb_path):
    with tf.Session() as sess:
        print(meta_path, ' ', model_name)
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        #saver.restore(sess,tf.train.latest_checkpoint('.'))
        saver.restore(sess, model_name)

        # Output nodes
        output_node_names =[n.name for n in tf.get_default_graph().as_graph_def().node]

        # Freeze the graph, 
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # Save the frozen graph
        with open(pb_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

def makeOpt(string, config):
    value = config.get(string)
    if value != None:
        r = (string + "=" + str(config[string]))     
    else:
        r = None
    return r 

def main(argv):
    config = json.load(open("config.json"))

    output_file        = makeOpt("--output_file", config)
    graph_def_file     = makeOpt("--graph_def_file", config)
    meta_file          = str(config['--meta_file'])
    meta_prefix        = str(config['--meta_prefix'])
    inference_type     = makeOpt("--inference_type", config)
    input_arrays       = makeOpt("--input_arrays", config)
    output_arrays      = makeOpt("--output_arrays", config) 
    default_ranges_max = makeOpt("--default_ranges_max", config)
    default_ranges_min = makeOpt("--default_ranges_min", config)
    mean_values        = makeOpt("--mean_values", config)
    std_dev_values     = makeOpt("--std_dev_values", config)

    if graph_def_file == None:
        pb_path = 'output.pb'
        toPb(meta_file, meta_prefix, pb_path)
        fixPb(pb_path)
        graph_def_file = "--graph_def_file" + "=" + pb_path

    print("graph_def_file = ", graph_def_file)
    call(["tflite_convert", 
          output_file,
          graph_def_file,
          inference_type, 
          input_arrays ,
          output_arrays,
          default_ranges_max,
          default_ranges_min,
          mean_values,
          std_dev_values])

if __name__ == "__main__":
    main(sys.argv[1:])