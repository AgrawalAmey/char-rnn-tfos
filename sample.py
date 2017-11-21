from __future__ import print_function
import tensorflow as tf
from tensorflowonspark import TFNode
import argparse
import os
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
import pydoop.hdfs as hdfs
from model import Model

from six import text_type
  
def main():
    sc=SparkContext(conf=SparkConf().setAppName("rnn_tf_sample"))

    parser = argparse.ArgumentParser(
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save_dir', type=str, default='save',
                        help='model directory to store checkpointed models')
    parser.add_argument('--output_dir', type=str, default='save',
                        help='directory to store genrated output')
    parser.add_argument('-n', type=int, default=500,
                        help='number of characters to sample')
    parser.add_argument('--prime', type=text_type, default=u' ',
                        help='prime text')
    parser.add_argument('--sample', type=int, default=1,
                        help='0 to use max at each timestep, 1 to sample at '
                             'each timestep, 2 to sample on spaces')

    args = parser.parse_args()
    sample(args, sc)


def sample(args, sc):
    defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
    working_dir = os.getcwd()
    
    config_file = TFNode.hdfs_path(os.path.join(args.save_dir, 'config.p'), defaultFS, working_dir)
    saved_args = sc.pickleFile(config_file).collect()[0]
    chars_vocab_file = TFNode.hdfs_path(os.path.join(args.save_dir, 'chars_vocab.p'), defaultFS, working_dir)
    chars, vocab = sc.pickleFile(chars_vocab_file).collect()
    model = Model(saved_args, training=False)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        save_dir  = TFNode.hdfs_path(os.path.join(args.save_dir, ''), defaultFS, working_dir)
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
 	    sample_ = model.sample(sess, chars, vocab, args.n, args.prime, args.sample)
	    with hdfs.open(TFNode.hdfs_path(os.path.join(args.output_dir, 'output.txt'), defaultFS, working_dir), 'w') as f:
                f.write(sample_)

if __name__ == '__main__':
    main()