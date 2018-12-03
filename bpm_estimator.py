import tensorflow as tf
import os
import glob

import utils


def reshape_stft(stfts, num_mel_bins):
    """
    Reshapes the audio data to be a more meanful of the original stft.

    :param stfts:
        The tensor of stfts.
    :param num_mel_bins:
        The resolution of the output, should be based on steps per octave.
    :return:
    """
    magnitude_spectrograms = tf.abs(stfts)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # scale frequency to mel scale and put into bins to reduce dimensionality
    lower_edge_hertz, upper_edge_hertz = 30.0, 17000.0

    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, utils.sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # log scale the mel bins to better represent human loudness perception
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # compute first order differential and concat. "It indicates a raise or reduction of the energy for each
    # frequency bin at a frame relative to its predecessor"
    first_order_diff = tf.abs(
        tf.subtract(log_mel_spectrograms, tf.manip.roll(log_mel_spectrograms, shift=1, axis=1)))
    mel_fod = tf.concat([log_mel_spectrograms, first_order_diff], 1)

    return mel_fod


def transform_data(audio, beatmap=None):
    """
    Transforms the audio data into a meanful representation. Also handles beatmap data, which is just reshaped
    and casted.

    1. Calculate STFT of audio signal for various resolutions.
    2. Magnitude of spectrograms.
    3. Truncate frequency range between 30-17000 Hz.
    3. Rescale to Mel scale.
    4. Reduce dimensionality by combining frequency bands.
    5. Log scale spectrograms.
    6. Calculate first order differences.
    7. Stack all three STFTs and their first order differences.

    :param audio:
        1d Tensor of type float16
    :param beatmap:
        1d Tensor of type int64
    :return:
        tuple of tensors of shape (timesteps, mel_bins_base * 14) and (timesteps, 1)
    """
    # transform audio
    stfts = [tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=1024),
             tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=2048),
             tf.contrib.signal.stft(audio, frame_length=utils.frame_length, frame_step=utils.frame_step, fft_length=4096)]

    # one hot encode the 'binary' values for beats
    # beatmap = tf.one_hot(beatmap, utils.num_classes)
    if beatmap is not None:
        beatmap = tf.reshape(beatmap, [int(utils.timesteps), 1])
        beatmap = tf.cast(beatmap, tf.float32)

    # apply transforms to stfts, this includes the first order difference data
    mel_bins_base = utils.mel_bins_base
    log_mel_spectrograms = [reshape_stft(stfts[0], mel_bins_base),
                            reshape_stft(stfts[1], mel_bins_base*2),
                            reshape_stft(stfts[2], mel_bins_base*4)]

    # concat specs together
    stacked_log_mel_specs = tf.concat([log_mel_spectrograms[0], log_mel_spectrograms[1]], 1)
    stacked_log_mel_specs = tf.concat([stacked_log_mel_specs, log_mel_spectrograms[2]], 1)

    return stacked_log_mel_specs, beatmap


def parse_tfrecord(example_proto):
    """
    Parsing function for when creating a TF Datset from TFRecords.
    :param example_proto:
        A single TF Example that is passed to this function when the Dataset reads the TFRecords.
    :return:
        A tuple that contains a dictionary of the transformed audio data (features) and beatmap data (labels).
    """
    keys_to_features = {'audio': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                        'beatmap': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    audio, beatmap = transform_data(parsed_features['audio'], parsed_features['beatmap'])
    return {'transformed_audio': audio}, beatmap


def build_dataset_from_tfrecords(records, tag, num_repeat):
    """
    Builds a TF Dataset from TFRecord shards.
    :param records:
        List of TFRecord shard filepaths to create the dataset from.
    :param tag:
        A string for naming the tranformation cache correctly.
    :param num_repeat:
        number of times to loop the dataset, required for training with the TF Estimator specifically.
    :return:
        A TF Dataset object.
    """
    dataset = tf.data.TFRecordDataset(records)

    dataset = dataset.apply(tf.contrib.data.map_and_batch(
        map_func=parse_tfrecord, batch_size=utils.batch_size, drop_remainder=True))
    # os.path.join(utils.working_dir, 'transformation_cache.dat')
    dataset = dataset.cache(os.path.join(utils.trans_cache_dir, '{}_transformation_cache.dat'.format(tag)))
    dataset = dataset.repeat(num_repeat)
    dataset = dataset.prefetch(4)

    return dataset


def serving_input_fn():
    """
    Function used to format data for inferences.
    :return:
        Returns a serving input reciever, to be consumed by an Exporter.
    """
    feature_placeholders = {
        'raw_audio_data': tf.placeholder(tf.float32, [int(utils.audio_clip_len*utils.sample_rate)])
    }
    transformed_audio, _ = transform_data(feature_placeholders['raw_audio_data'], None)
    transformed_audio = tf.reshape(transformed_audio, [1, transformed_audio.shape[0], transformed_audio.shape[1]])
    zeros = tf.zeros([utils.batch_size-1, transformed_audio.shape[1], transformed_audio.shape[2]])
    transformed_audio = tf.concat([transformed_audio, zeros], 0)
    features = {
        'transformed_audio': transformed_audio
    }

    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


def train_input_fn():
    """
    Training input function that feeds the estimator data. The dataset will repeat itself.
    :return:
        a TFDataset object to be consumed by an estimator
    """
    return build_dataset_from_tfrecords(
        glob.glob(os.path.join(utils.working_dir, 'train_fragment_*.tfrecords')), 'train', num_repeat=None)


def validation_input_fn():
    """
        Validation input function that feeds the estimator data.
        :return:
            a TFDataset object to be consumed by an estimator
        """
    return build_dataset_from_tfrecords(
        glob.glob(os.path.join(utils.working_dir, 'val_fragment_*.tfrecords')), 'val', num_repeat=1)


def test_input_fn():
    """
    Testing input function that feeds the estimator data.
    :return:
        a TFDataset object to be consumed by an estimator
    """
    return build_dataset_from_tfrecords(
        glob.glob(os.path.join(utils.working_dir, 'test_fragment_*.tfrecords')), 'test', num_repeat=1)


def thresholding(inputs):
    """
    Thresholding function that takes the probabilities of the output of the network and thresholds them based on the
    thresholding function from Sebastian Bock's thesis: http://mir.minimoog.org/sb-diploma-thesis

    The implementation is more complicated than expected because the data is in batched form, so the mean has to be
    computed batchwise. Also the use of bitfields and rolls complicate things further.
    :param inputs:
        A tensor of shape (batch_size, timesteps, 1)
    :return:
        A tensor that represents thresholded values of shape (timesteps, 1)
    """
    # find the mean for each example in the batch
    mean_output = tf.reduce_mean(inputs, axis=1)

    # scale each mean based on a factor
    threshold_scalar = tf.Variable(utils.threshold_scalar, tf.float32)
    scaled_mean = tf.scalar_mul(threshold_scalar, mean_output)
    scaled_mean = tf.reshape(scaled_mean, [utils.batch_size])

    # setup matrix for
    min_thresh_for_max = tf.fill([utils.batch_size], 0.05)
    max_thresh_for_min = tf.fill([utils.batch_size], 0.15)   #0.4
    thresholds = tf.maximum(min_thresh_for_max, scaled_mean)
    thresholds = tf.minimum(max_thresh_for_min, thresholds)

    # zero values under the thresholds using bitmask
    thresholds = tf.reshape(thresholds, [128, 1, 1])

    threshold_mask = tf.cast(tf.greater(inputs, thresholds), tf.float32)
    thresholded_input = tf.multiply(inputs, threshold_mask)

    # peak picking
    # select beats by x[i-1] < x[i] > x[i+1] (local maximum)
    x_minus_1 = tf.cast(tf.greater(thresholded_input, tf.manip.roll(thresholded_input, shift=-1, axis=1)), tf.float32)
    x_plus_1 = tf.cast(tf.greater(thresholded_input, tf.manip.roll(thresholded_input, shift=1, axis=1)), tf.float32)
    output = tf.multiply(x_minus_1, x_plus_1)

    return output


def model_fn(features, labels, mode):
    """
    The model function defines the neural network, output logits (read probabilites), predictions, loss mechanism,
    metrics, and results that are returned after a prediction.
    :param features:
        The input (transformed audio) tensors that come from the input_fn
    :param labels:
        The output (beat labels) tensors that come from the input_fn
    :param mode:
        A enum that describes if the model_fn is being used for training, validation (evaluation),
        or testing (inference).
    :return:
        Returns a EstimatorSpec that is ready to be used to create a custom Estimator
    """
    # turn inputs into time-major instead of batch-
    inputs = tf.transpose(features['transformed_audio'], perm=[1, 0, 2])
    brnn = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=3,
                                          input_mode='linear_input',
                                          num_units=50,
                                          direction='bidirectional')
    brnn_output, hidden = brnn(inputs)
    # TODO: might want to normalize inputs because the sigmoid will fk it?
    logits = tf.layers.dense(inputs=brnn_output, units=1, activation=tf.nn.sigmoid)
    logits = tf.transpose(logits, perm=[1, 0, 2])
    predictions = thresholding(logits)

    loss = None
    train_op = None
    metrics = None

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_pred=logits, y_true=labels))

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=utils.learning_rate)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
        # yay for metrics
        acc = tf.metrics.accuracy(predictions=predictions, labels=labels)
        pre = tf.metrics.precision(predictions=predictions, labels=labels)
        rec = tf.metrics.recall(predictions=predictions, labels=labels)

        metrics = {'accuracy': acc, 'precision': pre, 'recall': rec}
        tf.summary.scalar('accuracy', acc[1])
        tf.summary.scalar('precision', pre[1])
        tf.summary.scalar('recall', rec[1])

    result_dict = {
        'predictions': predictions,
        'probabilities': logits,
        'features': features['transformed_audio'],
        'brnn_output': tf.transpose(brnn_output, perm=[1, 0, 2])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        predictions=result_dict,
        train_op=train_op,
        eval_metric_ops=metrics,
        export_outputs={'predictions': tf.estimator.export.PredictOutput(result_dict)}
    )


def train_sequential():
    """
    Trains the network, evaluates, and tests the dataset, displaying the results. Serves as a function for testing the
    dataset until I have more time to export the model and do a bunch of cool stuff with it.
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_interval = 300

    # strategy = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=1)
    # config = tf.estimator.RunConfig(train_distribute=strategy, beval_distribute=strategy, model_dir=utils.nfs_dir)
    config = tf.estimator.RunConfig(model_dir=os.path.join(utils.working_dir, 'logs'), save_checkpoints_secs=eval_interval, keep_checkpoint_max=10)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
    # bidirectional_input
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=(utils.train_samples_to_use//utils.batch_size)*utils.num_epochs) #(utils.train_samples_to_use//utils.batch_size)*1) # utils.num_epochs

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec = tf.estimator.EvalSpec(input_fn=validation_input_fn,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=60,
                                      throttle_secs=eval_interval)

    tf.train.create_global_step()
    # TODO: train and evaluate isn't running evaluation step. not sure why.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    estimator.export_saved_model(export_dir_base=os.path.join(utils.working_dir, 'logs/exports'),
                                 serving_input_receiver_fn=serving_input_fn)

    estimator.evaluate(input_fn=validation_input_fn, steps=None)
    predictoo = estimator.predict(input_fn=test_input_fn)
    while True:
        predictions = next(predictoo)
        utils.plot_output(predictions['predictions'], predictions['probabilities'],
                          predictions['features'], predictions['brnn_output'])


'''
def get_predict_dataset_from_np():
    import soundfile as sf
    audio_file = sf.SoundFile('/mnt/mirrored-storage/tf-workdir/extracted/209651_Afilia_Saga_-_S.M.L_(TV_size_ver.).osz/sml (tv size).mp3.wav')
    batch_x = audio_file.read(dtype='float32')
    batch_x = batch_x[0:int(utils.audio_clip_len * utils.sample_rate) + 1]
    #batch_x = np.reshape(batch_x, (1, batch_x.shape[0]))
    dataset = tf.data.Dataset.from_tensor_slices(batch_x)
    dataset.apply(transform_data)
    return dataset


def predict_sequential(export_dir):
    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        eval_interval = 300
        config = tf.estimator.RunConfig(model_dir=os.path.join(utils.working_dir, 'logs'),
                                        save_checkpoints_secs=eval_interval, keep_checkpoint_max=10)
        estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)

        predictoo = estimator.predict(input_fn=test_input_fn)
        while True:
            predictions = next(predictoo)
            utils.plot_output(predictions['predictions'], predictions['probabilities'],
                              predictions['features'], predictions['brnn_output'])
'''


def parse_tfrecord_raw(example_proto):
    """
    Returns a dataset for the audio transformation visualizations. Not important to the function of the network.
    """
    keys_to_features = {'audio': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
                        'beatmap': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)}
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    return parsed_features['audio'], parsed_features['beatmap']


def visualize_data_transformations():
    """
    A function for visualizing the audio transformations. Not important to the function of the network.
    """
    records = glob.glob(os.path.join(utils.working_dir, 'train_fragment_*.tfrecords'))
    dataset = tf.data.TFRecordDataset(records)
    dataset = dataset.map(parse_tfrecord_raw)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.prefetch(2)
    it = dataset.make_one_shot_iterator()

    data_x = tf.placeholder(tf.float32, shape=(utils.sample_rate * utils.audio_clip_len,))
    data_y = tf.placeholder(tf.float32, shape=(utils.timesteps,))
    stfts = tf.contrib.signal.stft(data_x, frame_length=utils.frame_length, frame_step=utils.frame_step,
                                   fft_length=4096)
    power_stfts = tf.real(stfts * tf.conj(stfts))
    magnitude_spectrograms = tf.abs(stfts)
    power_magnitude_spectrograms = tf.abs(power_stfts)

    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    # scale frequency to mel scale and put into bins to reduce dimensionality
    lower_edge_hertz, upper_edge_hertz = 30.0, 17000.0
    num_mel_bins = utils.mel_bins_base * 4
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, utils.sample_rate, lower_edge_hertz,
        upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(
        magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # log scale the mel bins to better represent human loudness perception
    log_offset = 1e-6
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset)

    # compute first order differential and concat. "It indicates a raise or reduction of the energy for each
    # frequency bin at a frame relative to its predecessor"
    first_order_diff = tf.abs(
        tf.subtract(log_mel_spectrograms, tf.manip.roll(log_mel_spectrograms, shift=1, axis=1)))
    mel_fod = tf.concat([log_mel_spectrograms, first_order_diff], 1)

    with tf.Session() as sess:
        while True:
            try:
                raw_x, raw_y = sess.run(it.get_next())
                np_stfts = sess.run(power_stfts, feed_dict={data_x: raw_x})
                np_magnitude_spectrograms = sess.run(power_magnitude_spectrograms, feed_dict={data_x: raw_x})
                np_mel_spectrograms = sess.run(mel_spectrograms, feed_dict={data_x: raw_x})
                np_log_mel_spectrograms = sess.run(log_mel_spectrograms, feed_dict={data_x: raw_x})
                np_mel_fod = sess.run(mel_fod, feed_dict={data_x: raw_x})

                utils.plot_signal_transforms(raw_x,
                                             np_stfts,
                                             np_magnitude_spectrograms,
                                             np_mel_spectrograms,
                                             np_log_mel_spectrograms,
                                             np_mel_fod)
                print('wank')

            except tf.errors.OutOfRangeError:
                break
