import os
import csv
import random
import numpy as np
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
import tensorflow as tf
from tensorflow import keras
from scipy import stats

def lib_file_list(file_dirs):
    file_list = []
    read_list = []
    for directory in file_dirs:
        path = Path(directory)
        for afile in path.rglob("*.fast5"):
            file_list.append(str(afile))
    random.shuffle(file_list)
    for i in range(len(file_list)):
        with get_fast5_file(sample_file, mode='r') as f5:
            sample_file = file_list[i]
            for read in f5.get_reads():
                read_list.append(str(read))
    return file_list, read_list

def train_test_val_split(file_dirs): #, num_train, num_val):
    total_list = []
    train_list = []
    test_list = []
    val_list = []
    for directory in file_dirs:
        path = Path(directory)
        for afile in path.rglob("*.fast5"):
            # total_list.append(str(afile))
            rand_num = random.random()
            if rand_num < 0.1:
                val_list.append(str(afile))
            if rand_num >= 0.1 and rand_num < 0.2:
                test_list.append(str(afile))
            else:
                train_list.append(str(afile))
    random.shuffle(train_list)
    random.shuffle(test_list)
    random.shuffle(val_list)
    # random.shuffle(total_list)
    # train_list = total_list[0 : num_train : 1]
    # val_list = total_list[num_train+1 : num_train+num_val+1 : 1] 
    return train_list, test_list, val_list

def data_generation(file_list, window, shuffle=True):
    # for generating x (data) without labels -- for predicting
    if shuffle==True:
        random.shuffle(file_list)
    for i in range(len(file_list)):
        sample_file = file_list[i]
        with get_fast5_file(sample_file, mode='r') as f5:
            for read in f5.get_reads():
                whole_sample = np.asarray(read.get_raw_data(scale=True))
                start_col = random.randint(1000, (len(whole_sample)-window))
                sample = whole_sample[start_col:start_col+window:1]
                norm_sample = stats.zscore(sample) # .asarray()
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                yield x  

def data_and_label_generation(file_list, label, window, shuffle=True):
    # for generating x,y (data and labels) -- for training, testing
    if shuffle==True:
        random.shuffle(file_list)
    for i in range(len(file_list)):
        sample_file = file_list[i]
        with get_fast5_file(sample_file, mode='r') as f5:
            for read in f5.get_reads():
                whole_sample = np.asarray(read.get_raw_data(scale=True))
                start_col = random.randint(1000, (len(whole_sample)-window))
                sample = whole_sample[start_col:start_col+window:1]
                norm_sample = stats.zscore(sample) # .asarray()
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                y = np.array(label)
                yield (x, y)        

# def data_from_csv_generator(csv_file):
#     reader = csv.reader(csv_file)
#     for line in reader:
#         wholeline = 

# class TrainValSplit:
#     def __init__(self, file_dirs):
#         self.file_dirs = file_dirs
#         self.train_val_split() #file_dirs, train_reads, val_reads)
#
#     def train_val_split(self):
#         train_list = []
#         val_list = []
#         for directory in self.file_dirs:
#             path = Path(directory)
#             for afile in path.rglob("*.fast5"):
#                 rand_num = random.random()
#                 if rand_num < 0.1:
#                     val_list.append(str(afile))
#                 else:
#                     train_list.append(str(afile))
#         random.shuffle(train_list)
#         random.shuffle(val_list)
#         return train_list, val_list
#
# class SampleGeneratorFromFiles:
#     def __init__(self, file_list, label, shuffle=True):
#         self.file_list = file_list
#         self.label = label
#         self.shuffle = shuffle
#
#     # def __iter__(self):
#     #     return self
#
#     def data_generation(self): #__next__(self): #, file_list, label, numreads, shuffle):
#         window = 8000
#         if self.shuffle==True:
#             random.shuffle(self.file_list)
#         for i in range(len(self.file_list)):
#             sample_file = self.file_list[i]
#             with get_fast5_file(sample_file, mode='r') as f5:
#                 for read in f5.get_reads():
#                     whole_sample = np.asarray(read.get_raw_data(scale=True))
#                     start_col = random.randint(1000, (len(whole_sample)-window))
#                     sample = whole_sample[start_col:start_col+window:1]
#                     norm_sample = stats.zscore(sample) # .asarray()
#                     # sig_output = norm_sample.reshape((sample_array.shape[0], 1))
#                     # x = tf.convert_to_tensor(sig_output, dtype=tf.float32)
#                     x = norm_sample.reshape((norm_sample.shape[0], 1))
#                     # label_output = np.array(self.label)
#                     # y = tf.convert_to_tensor(label_output, dtype=tf.int16)
#                     y = np.array(self.label)
#                     # if i == self.numreads:
#                     #     break
#                     yield (x, y)                
#
# class BatchGenerator:
#     """
#     Spitting out batches of positives and negatives at the proper ratio
#     """
#     def __init__(self, pos_files, neg_files, window=8000, batchsize=32, ratio=0.5, shuffle=True):
#         self.pos_gen = SampleGeneratorFromFiles(pos_files, window, shuffle, 1)
#         self.neg_gen = SampleGeneratorFromFiles(neg_files, window, shuffle, 0)
#         self.window = window
#         self.batchsize = batchsize
#         self.ratio = ratio
#
#     def __next__(self):
#         batch = []
#         for i in range(0, self.batchsize):
#             random_num = random.random()
#             if random_num < self.ratio:
#                 batch += [next(self.pos_gen)]
#             else:
#                 batch += [next(self.neg_gen)]
#         yield batch
#
