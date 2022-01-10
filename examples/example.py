#!/usr/bin/env python

import time
import sys
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from fast5fetch import fast5data


def parse_args(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--dirs", required=True, nargs='+',
                        help="fast5 directory containing samples")
    return parser.parse_args(args)


def main():
    args = parse_args(sys.argv[1:])

    fast5s = fast5data.get_all_fast5s(args.dirs)

    batch = 32
    buffer_size = 1000
    window = 1000
    CPUS = 5  # CPUs

    # Count time to consume the generator
    start = time.time()
    g = fast5data.xy_generator_many(fast5s, 1, window, par=CPUS)
    for i in g:
        if g.count % 1000 == 0:
            print(g.count)
    print('reads: {:d} in {:f}secs'.format(g.count, time.time() - start))

    ####################
    # Benchmark datasets

    # Parallelized (preferred)
    ds = tf.data.Dataset.from_generator(
            fast5data.xy_generator_many,
            args=[fast5s, 1, window, True, CPUS],
            output_signature=(
                tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int16)))
    tfds.benchmark(ds.batch(batch).prefetch(buffer_size=buffer_size))

    # Interleaved files parallelized (painfully slow)
    ds = tf.data.Dataset.from_tensor_slices(fast5s)
    ds = ds.interleave(
            lambda filename: tf.data.Dataset.from_generator(
                fast5data.xy_generator_single,
                args=(filename, 1, window),
                output_signature=(
                    tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))),
            CPUS, 1, num_parallel_calls=CPUS, deterministic=False)
    tfds.benchmark(ds.batch(batch).prefetch(buffer_size=buffer_size))

    # Our previous approach (painfully slow)
    ds = tf.data.Dataset.from_generator(
            fast5data.data_and_label_generation,
            args=[fast5s, 1, window],
            output_signature=(
                tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int16)))
    tfds.benchmark(ds.batch(batch).prefetch(buffer_size=buffer_size))

    # Dataset from skewed generators.
    lessCPUS = int(CPUS/2)
    split = int(len(fast5s)/2)
    ds1_fast5s = fast5s[0:split]
    ds1 = tf.data.Dataset.from_generator(
            fast5data.xy_generator_many,
            args=[ds1_fast5s, 1, window, True, lessCPUS],
            output_signature=(
                tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int16)))
    ds2_fast5s = fast5s[split:]
    ds2 = tf.data.Dataset.from_generator(
            fast5data.xy_generator_many,
            args=[ds2_fast5s, 1, window, True, lessCPUS],
            output_signature=(
                tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int16)))
    ds = tf.data.experimental.sample_from_datasets(
            [ds1, ds2], weights=[0.5, 0.5])
    tfds.benchmark(ds.batch(batch).prefetch(buffer_size=buffer_size))

    # # An alternative to the above. Has the same performance but is less
    # # flexible.
    # split = int(len(fast5s)/2)
    # ds1_fast5s = fast5s[0:split]
    # ds2_fast5s = fast5s[split:]
    # ds = tf.data.Dataset.from_generator(
    #         fast5data.skew_generators,
    #         args=[ds1_fast5s, ds2_fast5s, 1, 0, window, 0.3, CPUS],
    #         output_signature=(
    #             tf.TensorSpec(shape=(window, 1), dtype=tf.float32),
    #             tf.TensorSpec(shape=(), dtype=tf.int16)))
    # tfds.benchmark(ds.batch(batch).prefetch(buffer_size=buffer_size))


if __name__ == "__main__":
    main()
