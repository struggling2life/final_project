#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from common import config
import tensorflow.contrib as tf_contrib
import tf_metrics


class BiLSTM:
    def __init__(self):
        # set the initializer of conv_weight and conv_bias
        self.weight_init = tf_contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                          mode='FAN_IN', uniform=False)
        self.bias_init = tf.zeros_initializer()
        self.reg = tf_contrib.layers.l2_regularizer(config.weight_decay)

    def _embed_layer(self, name, inp):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable(initializer=tf.random_uniform([config.vocab_size, config.embeddingsize], -1.0, 1.0),
                                name='W')
            embedded_chars = tf.nn.embedding_lookup(W, tf.cast(inp, tf.int32))
        return tf.cast(embedded_chars, tf.float32)

    def _lstm_layer(self, name, inp, rnn_size):
        with tf.variable_scope(name) as scope:
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size, forget_bias=1.0, name="fw")
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=rnn_size, forget_bias=1.0, name="bw")
            hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                             cell_bw=lstm_bw_cell,
                                                             inputs=inp,
                                                             dtype=tf.float32)
        return tf.concat(hiddens, 2)

    def build(self):
        data = tf.placeholder(tf.float32, shape=(None, config.padding_size), name='data')
        label = tf.placeholder(tf.int32, shape=(None, config.padding_size), name='label')
        sequence_lengths = tf.placeholder(tf.int32, shape=[None], name='sequence_lengths')
        is_training = tf.placeholder(tf.bool, name='is_training')

        # embeding
        x = self._embed_layer(name='embed_input', inp=data)  # [batch_size, padding_length, embeding_length]

        # bilstm
        x = self._lstm_layer(name="lstm1", inp=x, rnn_size=16)

        placeholders = {
            'data': data,
            'label': label,
            'is_training': is_training,
            'sequence_lengths': sequence_lengths
        }

        log_likelihood, self.transition_params = tf_contrib.crf.crf_log_likelihood(inputs=x,
                                                                                   tag_indices=label,
                                                                                   sequence_lengths=sequence_lengths)
        viterbi_sequence, viterbi_score = tf_contrib.crf.crf_decode(potentials=x,
                                                                    transition_params=self.transition_params,
                                                                    sequence_length=sequence_lengths)
        loss = -tf.reduce_mean(log_likelihood)

        # tf_metric, tf_metric_update = tf_metrics.precision(label, viterbi_sequence, 14, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        #                                  average="macro")
        #  = tf_metrics.recall(label, viterbi_sequence, 14, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average="macro")
        # f = tf_metrics.f1(label, viterbi_sequence, 14, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average="macro")
        # tf_metric_p, tf_metric_update = tf_metrics.precision(label, viterbi_sequence, 14,
        #                                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        #                                                      average="macro")
        # tf_metric_r, tf_metric_update = tf_metrics.recall(label, viterbi_sequence, 14,
        #                                                   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average="macro")
        # tf_metric_f, tf_metric_update = tf_metrics.f1(label, viterbi_sequence, 14,
        #                                               [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], average="macro")
        return placeholders, loss, viterbi_sequence, label
