from tensorflowonspark import TFCluster, TFNode
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from datetime import datetime
import argparse
from utils import TextLoader
import os
import pydoop.hdfs as hdfs


def main_fun(args, ctx):
    import tensorflow as tf
    import argparse
    import time
    import os
    from six.moves import cPickle
    from model import Model
    from tensorflowonspark import TFNode
    from datetime import datetime
    import numpy as np

    worker_num = ctx.worker_num
    job_name = ctx.job_name
    task_index = ctx.task_index
    cluster_spec = ctx.cluster_spec
    num_workers = len(cluster_spec['worker'])

    # Delay PS nodes a bit, since workers seem to reserve GPUs more quickly/reliably (w/o conflict)
    if job_name == "ps":
        time.sleep((worker_num + 1) * 5)

    # Get TF cluster and server instances
    cluster, server = TFNode.start_cluster_server(ctx, 1, args.rdma)

    if job_name == "ps":
        server.join()
    else:
        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % task_index,
                                                    cluster=cluster)):
            model = Model(args)
            # instrument for tensorboard
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()

        logdir = TFNode.hdfs_path(args.save_dir, ctx.defaultFS, ctx.working_dir)

        print("tensorflow model path: {0}".format(logdir))

        summary_writer = TFNode.get_summary_writer(ctx)

        sv = tf.train.Supervisor(is_chief=(task_index == 0),
                                logdir=logdir,
                                init_op=init_op,
                                summary_op=None,
                                saver=saver,
                                global_step=model.global_step,
                                stop_grace_secs=300, save_model_secs=10)

        # The supervisor takes care of session initialization, restoring from
        # a checkpoint, and closing when done or an error occurs.
        with sv.managed_session(server.target) as sess:
            print("{0} session ready".format(
                datetime.now().isoformat()))

            state=sess.run(model.initial_state)

            # Loop until the supervisor shuts down or 1000000 steps have completed.
            step=0
            tf_feed=TFNode.DataFeed(ctx.mgr, True)
            while not sv.should_stop() and not tf_feed.should_stop() and step < args.steps:
                # Run a training step asynchronously.
                # See `tf.train.SyncReplicasOptimizer` for additional details on how to
                # perform *synchronous* training.

                # using feed_dict
                batch = tf_feed.next_batch(args.batch_size)
                batch_xs = np.asarray([data[0] for data in batch])
                batch_ys = np.asarray([data[1] for data in batch])

                feed={model.input_data: batch_xs, model.targets: batch_ys}

                for i, (c, h) in enumerate(model.initial_state):
                    feed[c]=state[i].c
                    feed[h]=state[i].h

                if len(batch_xs) > 0:
                    # instrument for tensorboard
                    summ, train_loss, state, _, step = sess.run(
                        [summary_op, model.cost, model.final_state, model.train_op, model.global_step], feed_dict=feed)

                    # print loss
                    print("Step: {}, train_loss: {}".format(step, train_loss))

                if sv.is_chief:
                    summary_writer.add_summary(summ, step)

            if sv.should_stop() or step >= args.steps:
                tf_feed.terminate()

        # Ask for all the services to stop.
        print("{0} stopping supervisor".format(datetime.now().isoformat()))
        sv.stop()


if __name__ == '__main__':
    sc=SparkContext(conf=SparkConf().setAppName("rnn_tf_train"))

    executors=sc._conf.get("spark.executor.instances")
    num_executors=int(executors) if executors is not None else 1

    parser=argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_ps_tasks', type=int, default=1,
                        help='number of ps tasks')
    parser.add_argument('--num_clones', default=1, type=int,
                        help='Number of model clones to deploy.')
    parser.add_argument('--clone_on_cpu', default=False,
                       help='Use CPUs to deploy clones.')
    parser.add_argument("--rdma", help="use rdma connection", default=False)
    parser.add_argument('--data_dir', type=str, default='data/tinyshakespeare',
                        help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='save',
                        help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=128,
                        help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                        help='RNN sequence length')
    parser.add_argument('--grad_clip', type=float, default=5.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.97,
                        help='decay rate for rmsprop')
    parser.add_argument('--output_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the hidden layer')
    parser.add_argument('--input_keep_prob', type=float, default=1.0,
                        help='probability of keeping weights in the input layer')
    parser.add_argument("--epochs", help="number of epochs",
                        type=int, default=1)
    parser.add_argument(
        "--steps", help="maximum number of steps", type=int, default=1000)

    args=parser.parse_args()

    data_loader=TextLoader(
        sc, args.data_dir, args.batch_size, args.seq_length)

    args.vocab_size = data_loader.vocab_size

    defaultFS = sc._jsc.hadoopConfiguration().get("fs.defaultFS")
    working_dir = os.getcwd()

    config_file = TFNode.hdfs_path(os.path.join(args.save_dir, 'config.p'), defaultFS, working_dir)
    sc.parallelize([args]).saveAsPickleFile(config_file)

    chars_vocab_file = TFNode.hdfs_path(os.path.join(args.save_dir, 'chars_vocab.p'), defaultFS, working_dir)
    sc.parallelize([data_loader.chars, data_loader.vocab]).saveAsPickleFile(chars_vocab_file)

    dataRDD=sc.parallelize(data_loader.get_data_for_feeder())

    cluster=TFCluster.run(sc, main_fun, args, num_executors,
                            args.num_ps_tasks, TFCluster.InputMode.SPARK)

    cluster.train(dataRDD, args.epochs)

    cluster.shutdown()

    print("{0} ===== Stop".format(datetime.now().isoformat()))