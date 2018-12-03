# bpm-predictor
A bidirectional recurrent neural network (BRNN) with LSTM cells for predicting music beats using ~12512 'osu!' beatmaps as a dataset. Implementation is based on a thesis and several papers by Sebastian Bock.

A custom Tensorflow Estimator is used to implement the network. The data is preprocessing is multithreaded to incur high system utilization. The data is eventually ends up sharded into multiple TFRecord files that are consumed by a TF Dataset object.
<div align="center">
  <img src="https://i.imgur.com/775ElJa.png"><br><br>
</div>
