# char-rnn-tensorflow

> Multi-layer Recurrent Neural Networks (LSTM, RNN) for character-level language models in Python using TensorflowOnSpark.


TensorFlowOnSpark implementation of Char RNN inspired by [sherjilozair's implementation](https://github.com//char-rnn-tensorflow).


## Usage

### Training
```
${SPARK_HOME}/bin/spark-submit \
--master yarn --deploy-mode cluster \
--queue gpu \
--num-executors 3 \
--executor-memory 5G \
--py-files /media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/utils.py,/media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/model.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.executor.cores=1 \
/media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/train.py \
--data_dir /tfos/rnn_data/sherlock_small \
--epochs 50 \
--steps 10000 \
--save_dir /tfos/rnn_model
```

### Sample

```
${SPARK_HOME}/bin/spark-submit \
--master yarn --deploy-mode cluster \
--queue default \
--num-executors 1 \
--executor-memory 5G \
--py-files /media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/utils.py,/media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/model.py \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.executor.cores=1 \
/media/ephemeral0/tfos/TensorFlowOnSpark/examples/rnn/sample.py \
--output_dir /tfos/rnn_output \
--save_dir /tfos/rnn_model
```
