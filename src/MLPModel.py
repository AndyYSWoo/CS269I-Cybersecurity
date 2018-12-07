import tensorflow as tf
from LinearRegressionModel import LinearRegressionModel

class MLPModel(LinearRegressionModel):
    def build_prediction_op(self):
        features = self.input_placeholder
        with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
            for layer in range(self.FLAGS.layers):
                out = tf.layers.dense(features, self.FLAGS.hidden,
                                      activation=tf.nn.relu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength))
                out = tf.nn.dropout(out, self.keep_prob)
        out = tf.layers.dense(out, 1, activation=None,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(self.FLAGS.reg_strength),)
        self.predictions = out