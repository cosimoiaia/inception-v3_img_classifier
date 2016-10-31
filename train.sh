#!/bin/bash

if [ $# -lt 1 ]
then 
    echo "Usage: `basename ${0} .sh` --tf_bin=[tensorflow_installation_path] --tf_data=<tensorflow_data_folder>"
    exit -1;
fi

TF_BIN=/tensorflow

for i in "$@"
do 
    case $i in
        --tf_bin=*)
            TF_BIN="${i#*=}"
        shift
        ;;
        --tf_data=*)
            TF_DATA="${i#*=}"
        shift
        ;;
        *)
        ;;
    esac
done

echo "Binary:" $TF_BIN
echo "Data:" $TF_DATA

/usr/bin/env python $TF_BIN/tensorflow/examples/image_retraining/retrain.py \
--bottleneck_dir=$TF_DATA/bottlenecks \
--how_many_training_steps 4000 \
--model_dir=$TF_DATA/inception \
--output_graph=$TF_DATA/graph \
--output_labels=$TF_DATA/labels \
--image_dir $TF_DATA/dataset
