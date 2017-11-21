import codecs
import os
import collections
import numpy as np
from tensorflowonspark import TFNode
from tensorflowonspark import TFSparkNode
import pydoop.hdfs as hdfs

class TextLoader():
    def __init__(self, sc, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding

        defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
        working_dir = os.getcwd()

        input_file = TFNode.hdfs_path(os.path.join(data_dir, "input.txt"), defaultFS, working_dir)
        
        print("reading text file")
        self.preprocess(input_file)
        
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file):
        with hdfs.open(input_file, 'r') as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = zip(*count_pairs)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.array(list(map(self.vocab.get, data)))

    def create_batches(self):
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

        # When the data (tensor) is too small,
        # let's give them a better error message
        if self.num_batches == 0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1),
                                  self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0

    def get_data_for_feeder(self):
        xdata = np.copy(self.tensor)
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]

        x = xdata.reshape(-1, self.seq_length)
        y = ydata.reshape(-1, self.seq_length)

        return zip(x, y)
