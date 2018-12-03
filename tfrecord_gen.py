from random import shuffle
import math
import os
import numpy as np
import tensorflow as tf
from multiprocessing import Process, Queue, Array

import soundfile as sf
import pyttanko

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from data_prep import Base, BeatmapMetadata

import utils


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _read_audio(audio_path):
    """Reads an audio file
    :return:
        a numpy 1d numpy array of floats and a boolean value
        describing if the sample rate of the audio file was 44.1kHz
    """
    audio_file = sf.SoundFile(audio_path)
    return audio_file.read(dtype='float32'), audio_file.samplerate == utils.sample_rate


def process_beatmap(bmap, total_frames, frame_rate, offset_ms, mode=0, beat_divison=1):
    """
    :param bmap:
        pytanko beatmap object
    :param total_frames:
        total amount of frames in the beatmap
    :param frame_rate:
        rate of frames in the beatmap
    :param offset_ms:
        offset of the audiofile in milliseconds. used for skipping the intro section of the beatmap
    :param mode:
        mode=0, label beats based on bpm of timing sections
        mode=1, label beats based on highly subjectively human placed beats
        mode=2, use both methods above
    :param beat_divison:
        an integer describing the division of beats for mode 0 and mode 2.
         1 == quarter notes, 2 == eighth, 4 == sixteenth
    :return:
        a 1d numpy array of int64 that contains 0 for no beat and 1 for beat at a paticular timestep.
    """
    bm_hitobject_data = np.zeros(shape=total_frames, dtype=np.int64)

    # label beats based on bpm of timing sections
    if mode == 0 or mode == 2:
        timing_pts = list()
        for i in range(len(bmap.timing_points)):
            if bmap.timing_points[i].ms_per_beat > 1:
                timing_pts.append(bmap.timing_points[i])

        for i in range(0, len(timing_pts)):
            cur_beat_len = timing_pts[i].ms_per_beat / beat_divison
            cur_time = max(timing_pts[i].time, bmap.hitobjects[0].time)
            end_time = None

            if i < len(timing_pts) - 1:
                end_time = timing_pts[i+1].time
            else:
                end_time = bmap.hitobjects[-1].time
            while cur_time < end_time:
                j = math.floor((cur_time - offset_ms) / ((1/frame_rate)*1000))
                if j < total_frames:
                    bm_hitobject_data[j] = 1
                cur_time += cur_beat_len

    # label beats based on highly subjectively human placed beats
    if mode == 1 or mode == 2:
        # transform beatmap into frames
        for hitObject in bmap.hitobjects:
            i = math.floor((hitObject.time - offset_ms) / ((1/frame_rate)*1000))
            bm_hitobject_data[i] = 1

    return bm_hitobject_data


def _write_data_to_tfrecord_worker(start, end, data_set, label, path, output):
    """
    Processes audio and beatmap files, writing a tfrecord shard
    :param start:
        an integer describing the start index of the data_set to process (inclusive).
    :param end:
        an integer describing the end index of the data_set to process (exclusive).
    :param data_set:
        a list of tuples, (audio_filepath, beatmap_filepath)
    :param label:
        a string that represents the type of data (train, val, test)
    :param path:
        the directory to write the tfrecord file
    :param output:
        a Queue object from the python multiprocesing library
    """
    i = start
    total_writes = 0
    filename = '{}_fragment_{}-{}.tfrecords'.format(label, str(start), str(end))

    fullpath = os.path.join(path, filename)
    writer = tf.python_io.TFRecordWriter(fullpath)

    while i < end:
        try:
            # if not i % 50:
            print('{}: {}/{}'.format(filename, i-start, end-start))

            # read beatmap file
            file = open(data_set[i][1])
            p = pyttanko.parser()
            bmap = p.map(file)
            audio_data, is44khz = _read_audio(data_set[i][0])
            if is44khz:
                # trim audio based on first beat and last beat
                first_beat_time = bmap.hitobjects[0].time
                last_beat_time = bmap.hitobjects[-1].time
                first_frame = math.floor(first_beat_time / 1000 * utils.sample_rate)
                last_frame = math.floor(last_beat_time / 1000 * utils.sample_rate)
                audio_data = audio_data[first_frame:last_frame]

                total_frames = math.ceil(audio_data.size / utils.frame_step) + 1

                beatmap_data = process_beatmap(bmap, total_frames, utils.frame_rate, first_beat_time,
                                               beat_divison=4,
                                               mode=0)

                # audio clip length in seconds
                audio_clip_len = utils.audio_clip_len
                for j in range(math.floor(len(audio_data) / utils.sample_rate / audio_clip_len)):
                    start_sample = math.floor(j * utils.sample_rate * audio_clip_len)
                    end_sample = math.floor((j + 1) * utils.sample_rate * audio_clip_len)
                    audio_clip = audio_data[start_sample:end_sample]

                    start_frame = math.floor(start_sample / utils.frame_step)
                    end_frame = math.floor(end_sample / utils.frame_step) - 1
                    beatmap_clip = beatmap_data[start_frame:end_frame]

                    # print(len(audio_clip), len(beatmap_clip))
                    feature = {'audio': _float_feature(audio_clip),
                               'beatmap': _int64_feature(beatmap_clip)}

                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    total_writes += 1
                    writer.write(example.SerializeToString())
        except Exception:
            pass
        i += 1
    output.put(total_writes)


class TfRecordGen(object):
    def __init__(self, num_processes):
        """
        :param num_processes:
            The number of processes to run concurrently.
        """
        try:
            os.remove(os.path.join(utils.working_dir, 'tfrecord_data.txt'))
        except OSError:
            pass

        # format query for data
        engine = create_engine(utils.db_path)
        Base.metadata.bind = engine
        DBSession = sessionmaker(bind=engine)
        session = DBSession()

        query_data = session.query(BeatmapMetadata).\
            filter(BeatmapMetadata.gamemodeType == 0)

        data_paths = []
        seen_beatmaps = set()
        for bmmd in query_data:
            if bmmd.audioFilePath not in seen_beatmaps:
                data_paths.append((bmmd.audioFilePath, bmmd.bmFilePath))
                seen_beatmaps.add(bmmd.audioFilePath)
        shuffle(data_paths)

        train_data = data_paths[0:int(0.6 * len(data_paths))]
        val_data = data_paths[int(0.6 * len(data_paths)):int(0.8 * len(data_paths))]
        test_data = data_paths[int(0.8 * len(data_paths)):]

        self.write_data_to_tfrecord(train_data, 'train', num_processes, utils.working_dir)
        self.write_data_to_tfrecord(val_data, 'val', num_processes, utils.working_dir)
        self.write_data_to_tfrecord(test_data, 'test', num_processes, utils.working_dir)

    def write_data_to_tfrecord(self, data_set, label, num_processes, path=''):
        """
        :param data_set:
            a list of tuples, (audio_filepath, beatmap_filepath)
        :param label:
            a string that represents the type of data (train, val, test)
        :param num_processes:
            The number of processes to run concurrently.
        :param path:
            the directory to write the tfrecord files
        """
        output = Queue()
        processes = []
        for i in range(num_processes):
            start = i*(len(data_set)//num_processes)
            # python leaks scope, so this line does nothing. i hate python.
            end = None
            if i != num_processes-1:
                end = (i+1)*(len(data_set)//num_processes)
            else:
                end = len(data_set)
            processes.append(Process(target=_write_data_to_tfrecord_worker,
                                     args=(start, end, data_set, label, path, output)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        results = [output.get() for p in range(len(processes))]
        total_writes = sum(results)
        print(total_writes)
        tfd = open(os.path.join(utils.working_dir, 'tfrecord_data.txt'), 'a')
        tfd.write(str(total_writes)+'\n')


def main():
    TfRecordGen(num_processes=8)


if __name__ == "__main__":
    main()
