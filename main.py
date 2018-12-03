import tensorflow as tf
import os
import glob
import utils
import bpm_estimator

tf.app.flags.DEFINE_string("mode", "s", "Either 's' for single, 'd' for distributed', 'g' for graph, 'v' for visuals")
flags = tf.app.flags.FLAGS

with open('tf_config.json', 'r') as config:
    data = config.read()

# this config is for setting up distributed training
os.environ["TF_CONFIG"] = data

if flags.mode == 'd':
    pass
elif flags.mode == 's':
    bpm_estimator.train_sequential()
elif flags.mode == 'g':
    # For data visualizations.
    train_iterator = bpm_estimator.build_dataset_from_tfrecords(glob.glob(os.path.join(utils.working_dir, 'test_fragment_*.tfrecords')), 'train', num_repeat=1).make_one_shot_iterator()
    with tf.Session() as sess:
        while True:
            try:
                epoch_x, epoch_y = sess.run(train_iterator.get_next())
                for i in range(utils.batch_size):
                    utils.plot_stft(epoch_x['transformed_audio'][i], epoch_y[i])
                    print('...')

            except tf.errors.OutOfRangeError:
                break
elif flags.mode == 'v':
    bpm_estimator.visualize_data_transformations()
