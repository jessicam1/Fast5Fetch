import time
import sys
import random
import numpy as np
import multiprocessing as mp
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from scipy import stats


def get_all_fast5s(dirs):
    """ returns all fast5 files in a given directory list, recursively """

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
                x = process_fast5_read(read, window)
                yield x


def data_and_label_generation(file_list, label, window, shuffle=True):
    # for generating x,y (data and labels) -- for training, testing
    if shuffle:
        random.shuffle(file_list)

    for i in range(len(file_list)):
        sample_file = file_list[i]
        with get_fast5_file(sample_file, mode='r') as f5:
            for read in f5.get_reads():
                x = process_fast5_read(read, window)
                y = np.array(label)
                yield (x, y)


def process_fast5_read(read, window, skip=1000, zscore=True):
    """ Normalizes and extracts specified region from raw signal """

    s = read.get_raw_data(scale=True)  # Expensive

    if zscore:
        s = stats.zscore(s)

    pos = random.randint(skip, len(s)-window)

    return s[pos:pos+window].reshape((window, 1))


def xy_generator_single(fast5, label, window):
    """ Generator that yields training examples from one fast5 file  """

    with get_fast5_file(fast5, mode='r') as f5:
        for read in f5.get_reads():
            x = process_fast5_read(read, window)
            y = np.array(label)
            yield (x, y)

def x_generator_single(fast5, window):
    """ Generator that yields training examples from one fast5 file  """

    with get_fast5_file(fast5, mode='r') as f5:
        for read in f5.get_reads():
            x = process_fast5_read(read, window)
            yield (x)

class xy_generator_many_wrapper():

    def __init__(self, files, label, window, shuffle=True, par=1, buff=100):
        self.gen = xy_generator_many(
                files, label, window, shuffle=shuffle, par=par, buff=buff)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def __del__(self):
        self.gen.pool.terminate()


class xy_generator_many():
    """ Generator that yields training examples from multiple fast5 files

    Processing of fast5 files can be parallelized to achieve higher throughput.
    """

    def __init__(self, files, label, window, shuffle=True, par=1, buff=100):
        if shuffle:
            random.shuffle(files)

        self.files = files
        self.label = label
        self.window = window
        self.shuffle = shuffle
        self.par = par
        self.count = 0  # counts number of reads processed
        self.buff = buff

        # Create queue for workers to put results
        m = mp.Manager()
        self.results = m.Queue(self.buff)

        # Create a pool of workers and submit jobs
        chunksize = int(len(files)/(2*par))
        if chunksize == 0:
            chunksize = 1
        self.pool = mp.Pool(self.par)
        self.async_res = self.pool.map_async(self.worker, self.files,
                                             chunksize=chunksize)

        # Close the pool, as no more jobs will be submitted
        self.pool.close()

    def __iter__(self):
        return self

    def __next__(self):
        # Logic to terminate iteration when jobs are finished and all results
        # have been consumed. The empty() and ready() methods used below have
        # infinitesimally small delays until they return True. Therefore, we
        # need to check them again after a small delay. Perhaps not the best
        # solution.
        if self.results.empty():
            time.sleep(1)
            if self.results.empty() and self.async_res.ready():
                raise StopIteration

        self.count += 1
        return self.results.get()

    def worker(self, fast5):
        """ Processes a fast5 file and adds training examples in a queue"""

        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads():
                x = process_fast5_read(read, self.window)
                y = np.array(self.label)
                self.results.put((x, y), block=True)

    #  https://stackoverflow.com/questions/25382455/
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['async_res']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)


class batch_readid_x_generator_many_wrapper():

    def __init__(self, files, window, par=1, buff=100):
        self.gen = readid_x_generator_many(
                files, window, par=par, buff=buff)

    def __iter__(self):
        return self

    def __next__(self):
        i = 0
        bs = 32
        batch = np.empty([1, 4000, 1])
        # add window and bs to params
        while next(self.gen) != None:
            i +=1 
            batch = np.append(batch, 
                next(self.gen).reshape(1, 4001, 1), axis=0)
            if i == bs:
                i ==0
                # how to deal with < 32 samples at end
                # foo 0 0 0 ... 0
                return batch
                # how to clear batch after returning
                

    def __del__(self):
        self.gen.pool.terminate()

class readid_x_generator_many_wrapper():

    def __init__(self, files, window, par=1, buff=100):
        self.gen = readid_x_generator_many(
                files, window, par=par, buff=buff)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.gen)

    def __del__(self):
        self.gen.pool.terminate()


class readid_x_generator_many():
    """ Generator that yields training examples from multiple fast5 files

    Processing of fast5 files can be parallelized to achieve higher throughput.
    """

    def __init__(self, files, window, par=1, buff=100):
        self.files = files
        self.window = window
        self.par = par
        self.count = 0  # counts number of reads processed
        self.buff = buff

        # Create queue for workers to put results
        m = mp.Manager()
        self.results = m.Queue(self.buff)

        # Create a pool of workers and submit jobs
        self.pool = mp.Pool(self.par)
        self.async_res = self.pool.map_async(self.worker, self.files,
                                             chunksize=int(len(files)/(2*par)))

        # Close the pool, as no more jobs will be submitted
        self.pool.close()

    def __iter__(self):
        return self

    def __next__(self):
        # Logic to terminate iteration when jobs are finished and all results
        # have been consumed. The empty() and ready() methods used below have
        # infinitesimally small delays until they return True. Therefore, we
        # need to check them again after a small delay. Perhaps not the best
        # solution.
        if self.results.empty():
            time.sleep(1)
            if self.results.empty() and self.async_res.ready():
                raise StopIteration

        self.count += 1
        return self.results.get()

    def worker(self, fast5):
        """ Processes a fast5 file and adds training examples in a queue"""

        with get_fast5_file(fast5, mode='r') as f5:
            for read in f5.get_reads():
                readid = read.get_read_id()
                x = process_fast5_read(read, self.window)
                self.results.put((readid, x), block=True)

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        del self_dict['async_res']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

def skew_generators(files1, files2, label1, label2, window, skew=0.5, par=2):
    """ Interleaves 2 generators. skew defines proportion of the first

    Not extensively test, especially with respect to StopIteration exceptions.
    """

    par1 = int(skew * par) + 1
    par2 = int((1-skew) * par) + 1
    g1 = xy_generator_many(files1, label1, window, True, par1)
    g2 = xy_generator_many(files2, label2, window, True, par2)

    # TODO: The code below will probably lead to unexpected StopIteration
    while True:
        if random.random() < skew:
            yield next(g1)

        yield next(g2)
