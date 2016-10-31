#!/usr/bin/env python

##########################################
#
# label_image.py: Simple parametized python script to use a fine trained Inception V3 model to classify images
#          Based on:
#            *  Tensorflow example https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html#training-on-your-own-categories
#            *  This great article on codeLabs https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
#
# NOTE: This version works with TensorFlow-0.9.0-devel!
#
# Author: Cosimo Iaia <cosimo.iaia@gmail.com>
# Date: 26/10/2016
#
# This file is distribuited under the terms of GNU General Public
#
#########################################



import tensorflow as tf
import sys
import argparse


FLAGS = None

def main():
     image_path = FLAGS.image_path
     
     # Read in the image_data
     image_data = tf.gfile.FastGFile(image_path, 'rb').read()
     
     # Loads label file, strips off carriage return
     label_lines = [line.rstrip() for line 
                        in tf.gfile.GFile(FLAGS.datafolder+"/retrained_labels.txt")]
     
     # Unpersists graph from file
     with tf.gfile.FastGFile(FLAGS.datafolder"/retrained_graph.pb", 'rb') as f:
         graph_def = tf.GraphDef()
         graph_def.ParseFromString(f.read())
         _ = tf.import_graph_def(graph_def, name='')
     
     with tf.Session() as sess:
         # Feed the image_data as input to the graph and get first prediction
         softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
         
         predictions = sess.run(softmax_tensor, \
                  {'DecodeJpeg/contents:0': image_data})
         
         # Sort to show labels of first prediction in order of confidence
         top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
         
         for node_id in top_k:
             human_string = label_lines[node_id]
             score = predictions[0][node_id]
             print('%s (score = %.5f)' % (human_string, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image classifier with custom fine trained Inception V3")
    parser.add_argument('--datafolder', type=str, required=True, default='tf_files', help='Path to trained tensorflow model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the images to be classified')
    FLAGS = parser.parse_args()
    main()
