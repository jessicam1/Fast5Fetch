Fast5Fetch
This package was created primarily for training data generation from nanopore sequencing data. It includes two generators (one that generates x and y for training/testing/validation and one that generates x alone for predictions). 

How to use for training, testing, and validation data from multiple ONT libraries for a neural network:
1) Get a list of files for training, testing, and validation for each class.
train_list, test_list, val_list = train_test_val_split(library_dirs)
2) Create a tf.data.Dataset from the generator for each train, test, val file list for each class.
set = tf.data.Dataset.from_generator(data_and_label_generation, args[list, label, window], output_signature=(tf.TensorSpec(shape=window,1), dtype=tf.float32), tf.TensorSpec(shape-(), dtype=tf.int16)))
3) Combine datasets from negative and positive class at desired ratio using tf.data.experimental.sample_from_dataset.
dataset = tf.data.experimental.sample_from_datasets([pos_set, neg_set], weights=[ratio, (q-ratio)]).take(num).batch(batch_size).repeat()

How to use for getting predictions from a model:
1) Get list of files and list of reads for single ONT library.
file_list, read_list = lib_file_list(library)
2) Create a tf.data.Dataset from the generator with the file list.
set = tf.data.Dataset.from_generator(data_generation, args[list, window], output_signature=(tf.TensorSpec(shape=window,1), dtype=tf.float32)).take(num)



