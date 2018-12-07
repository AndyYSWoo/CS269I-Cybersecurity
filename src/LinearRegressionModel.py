from os import mkdir
from os.path import join, exists
from shutil import rmtree
import logging
import json

import numpy as np
from sklearn.metrics import r2_score
import tensorflow as tf

class LinearRegressionModel(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def build(self):
        # Build compuation graph
        self.add_placeholders_op()
        self.build_prediction_op()
        self.add_loss_op()
        self.add_optimizer_op()

        # Model savers and tensorboard summaries
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self.best_model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
        tf.summary.histogram('prices/predictions', self.predictions)
        tf.summary.histogram('prices/labels', self.label_placeholder)
        self.summaries = tf.summary.merge_all()

    def initialize(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholders_op(self):
        self.input_placeholder = tf.placeholder(tf.float32, (None, self.num_features))
        self.label_placeholder = tf.placeholder(tf.float32, (None, 1))
        self.keep_prob = tf.placeholder_with_default(1.0, shape=())

    def build_prediction_op(self):
        features = self.input_placeholder

        with tf.variable_scope('LinearRegression', reuse=tf.AUTO_REUSE):
            out = tf.contrib.layers.fully_connected(
                features,
                1,
                activation_fn=None,
                weights_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength)
            )
        self.predictions = out

    def add_loss_op(self):
        self.loss = tf.losses.mean_squared_error(
            labels=self.label_placeholder,
            predictions=self.predictions,
            reduction=tf.losses.Reduction.MEAN
        )

    def add_optimizer_op(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = tf.contrib.layers.optimize_loss(
            self.loss, self.global_step, self.FLAGS.learning_rate, 'Adam', summaries=['gradients'])

    def train(self, X_train, y_train, X_dev, y_dev):
        epoch = 0
        self.best_dev_r2 = None
        data_size = X_train.shape[0]
        indices = np.arange(data_size)
        while self.FLAGS.num_epochs == 0 or epoch < self.FLAGS.num_epochs:
            np.random.shuffle(indices)  # Shuffle batches inside file
            for batch_start in np.arange(0, data_size, self.FLAGS.batch_size):
                batch_indices = indices[batch_start: batch_start + self.FLAGS.batch_size]
                X_train_batch = X_train[batch_indices]
                y_train_batch = y_train[batch_indices]
                input_feed = {
                    self.input_placeholder: X_train_batch,
                    self.label_placeholder: y_train_batch,
                    self.keep_prob: self.FLAGS.dropout
                }
                output_feed = [self.summaries, self.train_op, self.loss, self.global_step]
                summaries, _, train_loss, global_step = self.sess.run(output_feed, feed_dict=input_feed)
                self.summary_writer.add_summary(summaries, global_step)
                self.write_summary(train_loss, 'loss/train', global_step)
                # Sometimes evaluate model on loss/r2
                if global_step % self.FLAGS.eval_every == 0:
                    train_r2 = self.eval(X_train_batch, y_train_batch)
                    self.write_summary(train_r2, 'r2/train', global_step)

                    # Eval on whole dev set
                    dev_loss = self.sess.run(self.loss, feed_dict={self.input_placeholder: X_dev, self.label_placeholder: y_dev})
                    dev_r2 = self.eval(X_dev, y_dev)
                    self.write_summary(dev_loss, 'loss/dev', global_step)
                    self.write_summary(dev_r2, 'r2/dev', global_step)

                    logging.info('Epoch: %d, Iter: %d, train loss: %3.4f, train r2: %1.3f, '
                                 'dev loss: %3.4f, dev r2: %1.3f' % (epoch, global_step, train_loss, train_r2,
                                                                     dev_loss, dev_r2))

                    if self.best_dev_r2 is None or dev_r2 > self.best_dev_r2:
                        self.best_dev_r2 = dev_r2
                        logging.info('\tBest dev r2 so far: %1.3f! Saving best checkpoint to %s...'
                                     % (self.best_dev_r2, self.bestmodel_ckpt_path))
                        self.best_model_saver.save(self.sess, self.bestmodel_ckpt_path, global_step=global_step)
                        self.write_weights()

                # Sometimes save model checkpoint
                if self.FLAGS.save_every != 0 and global_step % self.FLAGS.save_every == 0:
                    logging.info('Saving checkpoint to %s...' % self.checkpoint_path)
                    self.saver.save(self.sess, self.checkpoint_path, global_step=global_step)
            epoch += 1

    def eval(self, X, y):
        predictions = self.sess.run(self.predictions, feed_dict={self.input_placeholder: X})
        r2 = r2_score(y, predictions)
        return r2

    def load_data_npy(self, path):
        data = np.load(path)
        X, y = data[:, 1:], data[:, 0].reshape(-1, 1)
        self.num_features = X.shape[1]
        return X, y

    def run(self):
        # Create a session
        config = tf.ConfigProto()
        self.sess = tf.Session(config=config)

        # Model saving related
        latest_dir = join(self.FLAGS.train_dir, 'latest_checkpoint')
        self.checkpoint_path = join(latest_dir, 'latest.ckpt')
        best_dir = join(self.FLAGS.train_dir, 'best_checkpoint')
        self.bestmodel_ckpt_path = join(best_dir, 'best.ckpt')

        # Load data
        X_train, y_train = self.load_data_npy(self.FLAGS.train_file)
        # Overfit a small subset for sanity check
        # X_train = X_train[:100,:]
        # y_train = y_train[:100,:]
        X_dev, y_dev = self.load_data_npy(self.FLAGS.dev_file)

        # Build the compute graph
        self.build()

        if self.FLAGS.mode == 'train':
            for d in [self.FLAGS.train_dir, latest_dir, best_dir]: mkdir(d)
            self.initialize()

            # Set up logger
            file_handler = logging.FileHandler(join(self.FLAGS.train_dir, 'log.txt'))
            logging.getLogger().addHandler(file_handler)

            # Save a record of flags as a .json file in train_dir
            with open(join(self.FLAGS.train_dir, "flags.json"), 'w') as fout:
                fout.write(json.dumps(self.FLAGS.flag_values_dict(), indent=4))

            # Summary file writter
            self.summary_writer = tf.summary.FileWriter(self.FLAGS.train_dir)

            # Start training
            self.train(X_train, y_train, X_dev, y_dev)

        elif self.FLAGS.mode == 'load':
            # Load best checkpoint
            ckpt = tf.train.get_checkpoint_state(best_dir)
            ckpt_path = ckpt.model_checkpoint_path + '.index' if ckpt else ''
            if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(ckpt_path)):
                self.best_model_saver.restore(self.sess, ckpt.model_checkpoint_path)

    def write_summary(self, value, tag, global_step):
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)


    def write_weights(self):
        weight_dir = join(self.FLAGS.train_dir, 'weights')
        if exists(weight_dir): rmtree(weight_dir)
        mkdir(weight_dir)
        names = [v.name for v in tf.trainable_variables()]
        weights = self.sess.run(names)
        for name, weight in zip(names, weights):
            np.save(join(weight_dir, name.replace('/', '_').replace(':', '_')), weight)
        logging.info('\tAll weights written out to %s.' % weight_dir)
