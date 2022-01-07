import sys
import random
import numpy as np
import multiprocessing as mp
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def get_all_fast5s(dirs):
    fast5s = []
    for d in dirs:
        fast5s += [str(x) for x in Path(d).rglob("*.fast5")]

    return fast5s


def train_test_val_split(dirs, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    elems = get_all_fast5s(dirs)

    if len(elems) < 3:
        print("error: at least 3 elements are required", file=sys.stderr)
        return None, None, None

    if shuffle:
        random.shuffle(elems)

    # Assures that no list is returned empty
    train_list, test_list, val_list = [elems[0]], [elems[1]], [elems[2]]

    for e in elems[3:]:
        rand_num = random.random()
        if rand_num < val_ratio:
            val_list.append(e)
        elif rand_num < val_ratio + test_ratio:
            test_list.append(e)
        else:
            train_list.append(e)

    return train_list, test_list, val_list


def data_generation(file_list, window, shuffle=True):
    # for generating x (data) without labels -- for predicting
    if shuffle:
        random.shuffle(file_list)

    for i in range(len(file_list)):
        sample_file = file_list[i]
        with get_fast5_file(sample_file, mode='r') as f5:
            for read in f5.get_reads():
                norm_sample = process_fast5_read(read, window)
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                yield x


def data_and_label_generation(file_list, label, window, shuffle=True):
    # for generating x,y (data and labels) -- for training, testing
    if shuffle:
        random.shuffle(file_list)

    for i in range(len(file_list)):
        sample_file = file_list[i]
        with get_fast5_file(sample_file, mode='r') as f5:
            for read in f5.get_reads():
                norm_sample = process_fast5_read(read, window)
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                y = np.array(label)
                yield (x, y)


def process_fast5_read(read, window, skip=1000):
    whole_sample = np.asarray(read.get_raw_data(scale=True))
    start_col = random.randint(skip, (len(whole_sample)-window))
    sample = whole_sample[start_col:start_col+window:1]
    norm_sample = stats.zscore(sample)
    return norm_sample


def fast5_generator(fast5, label, window):
    with get_fast5_file(fast5, mode='r') as f5:
        for read in f5.get_reads():
            s = process_fast5_read(read, window)
            x = s.reshape((s.shape[0], 1))
            y = np.array(label)
            yield (x, y)


def fast5_read_generator(file_list, shuffle=True):
    if shuffle:
        random.shuffle(file_list)

    for f in file_list:
        with get_fast5_file(f, mode='r') as f5:
            for read in f5.get_reads():
                yield read.reshape((s.shape[0], 1))


def fast5_read_to_example(read, label, window):
    r = process_fast5_read(read, window)
    x = s.reshape((s.shape[0], 1))
    y = np.array(label)
    return (x, y)


def get_examples_from_fast5(fast5, label, window):
    with get_fast5_file(fast5, mode='r') as f5:
        for read in f5.get_reads():
            return fast5_read_to_example(read, label, window)


def worker(input, output, label, window):
    """ Processes fast5 files and puts individual reads in queue"""

    for f in iter(input.get, 'STOP'):
        print(f)
        with get_fast5_file(f, mode='r') as f5:
            for read in f5.get_reads():
                norm_sample = process_fast5_read(read, window)
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                y = np.array(label)
                output.put((x, y))
    return


class parallel_xy_generator():

    def __init__(self, files, label, window, shuffle=True, par=1):
        self.files = files
        self.label = label
        self.window = window
        self.shuffle = shuffle
        self.par = par

        self.pool = mp.Pool(self.par)
        self.it = self.pool.imap_unordered(self.worker, self.files)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.it)

    def worker(self, fast5):
        """ Processes a fast5 with single read """

        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads():
                norm_sample = process_fast5_read(read, self.window)
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                y = np.array(self.label)
                return(x, y)

    #  https://stackoverflow.com/questions/25382455/
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['it']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class parallel_xy_generator2():

    # def __init__(self, files, label, window, shuffle=True, par=1):
    #     self.files = files
    #     self.label = label
    #     self.window = window
    #     self.shuffle = shuffle
    #     self.par = par
    #
    #     # Create queues
    #     self.task_q = Queue()
    #     self.done_q = Queue()
    #
    #     # Submit tasks:
    #     for f in files:
    #         self.task_q.put(f)
    #
    #     # Start worker processes
    #     for i in range(par):
    #         Process(target=worker,
    #                 args=(self.task_q, self.done_q, label, window)).start()
    #
    #     # Tell child processes to stop when all tasks are done
    #     for i in range(par):
    #         self.task_q.put('STOP')
    #
    def __init__(self, files, label, window, shuffle=True, par=1):
        self.files = files
        self.label = label
        self.window = window
        self.shuffle = shuffle
        self.par = par
        self.done = False

        # Create queue for workers to put results
        m = mp.Manager()
        self.results = m.Queue()

        # Create a pool of workers and submit jobs
        self.pool = mp.Pool(self.par)
        self.it = self.pool.imap_unordered(self.worker, self.files)

        self.pool.close()
        self.pool.join()
        self.done = True

    def __iter__(self):
        return self

    def __next__(self):
        if self.results.empty() and self.done:
            raise StopIteration
        return self.results.get()

    def worker(self, fast5):
        """ Processes fast5 files and puts individual reads in queue"""

        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads():
                norm_sample = process_fast5_read(read, self.window)
                x = norm_sample.reshape((norm_sample.shape[0], 1))
                y = np.array(self.label)
                self.results.put((x, y))

    #  https://stackoverflow.com/questions/25382455/
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['it']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
