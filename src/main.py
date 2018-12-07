import time
import os
import logging

import tensorflow as tf

from LinearRegressionModel import LinearRegressionModel
from MLPModel import MLPModel

logging.basicConfig(level=logging.INFO)
REPO_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

tf.app.flags.DEFINE_string('mode', 'train', 'Available modes: train / load')
tf.app.flags.DEFINE_string('model', 'LR', 'LR / MLP')

tf.app.flags.DEFINE_string('data_dir', os.path.join(REPO_DIR, "data/npy"), 'Main directory of data')
tf.app.flags.DEFINE_string('train_file', os.path.join(REPO_DIR, 'data/npy/train.npy'), 'Train file path')
tf.app.flags.DEFINE_string('dev_file', os.path.join(REPO_DIR, 'data/npy/dev.npy'), 'Train file path')

# Training related paths
tf.app.flags.DEFINE_string('experiment_name', '', 'Unique name for experiment.')
tf.app.flags.DEFINE_string('tb_dir', os.path.join(REPO_DIR, 'experiments'), 'Directory of experiments, default to experiments/')
tf.app.flags.DEFINE_string('train_dir', '', 'Training directory')

# Debugging related
tf.app.flags.DEFINE_integer('eval_every', 10, 'How many iterations to do per calculating loss/auc')
tf.app.flags.DEFINE_integer('save_every', 0, 'How many iterations to save a model checkpoint')

# Training Hyperparameters
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
tf.app.flags.DEFINE_float('reg_strength', 0, 'L2 Regularization strength')
tf.app.flags.DEFINE_integer('batch_size', 500, 'Batch size')
tf.app.flags.DEFINE_integer('num_epochs', 0, 'Number of training epochs, 0 for indefinitely.')

# MLP Hyperparameters
tf.app.flags.DEFINE_integer('layers', 5, 'Number of hidden layers of MLP')
tf.app.flags.DEFINE_integer('hidden', 100, 'Number of hidden neurons in each layer of MLP')
tf.app.flags.DEFINE_float('dropout', 0.8, 'Drop out reg')

FLAGS = tf.app.flags.FLAGS

def main(unused_argv):
    FLAGS.experiment_name = FLAGS.experiment_name or str(int(time.time()))
    FLAGS.train_dir = os.path.join(FLAGS.tb_dir, FLAGS.experiment_name)

    models = {
        "LR": LinearRegressionModel(FLAGS),
        "MLP": MLPModel(FLAGS)
    }
    model = models[FLAGS.model]
    model.run()


if __name__ == "__main__":
    tf.app.run()
