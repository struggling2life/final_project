import tensorflow as tf
from common import config
from dataforbilstm import Dataset
import argparse
from bilstm import BiLSTM
import os
import pandas as pd
import numpy as np
import tf_metrics


def main():
    d = Dataset()
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue', dest='continue_path', required=False)
    args = parser.parse_args()

    ## build graph
    network = BiLSTM()
    placeholders, loss, viterbi_sequence, label = network.build()

    # loss_reg = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    ## train config
    global_steps = tf.Variable(0, trainable=False)
    boundaries = [config.train_size // config.batch_size * 15, config.train_size // config.batch_size * 40]
    values = [0.01, 0.001, 0.0005]
    lr = tf.train.piecewise_constant(global_steps, boundaries, values)
    opt = tf.train.AdamOptimizer(lr)
    # in order to update BN in every iter
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train = opt.minimize(loss)

    ## init tensorboard
    # tf.summary.scalar('loss_regularization', loss_reg)
    # tf.summary.scalar('loss_crossEntropy', loss - loss_reg)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('learning_rate', lr)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'train'),
                                         tf.get_default_graph())
    test_writer = tf.summary.FileWriter(os.path.join(config.log_dir, 'tf_log', 'validation'),
                                        tf.get_default_graph())

    ## create a session
    tf.set_random_seed(12345)  # ensure consistent results
    global_cnt = 0
    epoch_start = 0
    g_list = tf.global_variables()
    saver = tf.train.Saver(var_list=g_list)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())  # init all variables
        if args.continue_path:  # load a model snapshot
            ckpt = tf.train.get_checkpoint_state(args.continue_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            epoch_start = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[1])
            global_cnt = epoch_start * config.train_size // config.batch_size

        ## training
        for epoch in range(epoch_start + 1, config.nr_epoch + 1):
            for _ in range(config.train_size // config.batch_size):
                global_cnt += 1
                images, labels, seq_len = d.one_batch_train().__next__()
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: labels,
                    global_steps: global_cnt,
                    placeholders['is_training']: True,
                    placeholders['sequence_lengths']: seq_len
                }
                _, loss_v, lr_v, summary, v, y = sess.run([train, loss,
                                                           lr, merged, viterbi_sequence, label],
                                                          feed_dict=feed_dict)

                if global_cnt % config.show_interval == 0:
                    precision, recall = cal(v, y)
                    train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{},{}/{}".format(epoch, (global_cnt % config.train_size) // config.batch_size,
                                            config.train_size // config.batch_size),
                        'loss: {:.3f}'.format(loss_v),
                        'precision: {:.3f}'.format(precision),
                        'recall: {:.3f}'.format(recall)

                    )

            ## validation
            if epoch % config.test_interval == 0:
                loss_sum = 0
                for i in range(config.val_size // config.batch_size):
                    images, labels, seq_len = d.one_batch_val().__next__()
                    feed_dict = {
                        placeholders['data']: images,
                        placeholders['label']: labels,
                        global_steps: global_cnt,
                        placeholders['is_training']: False,
                        placeholders['sequence_lengths']: seq_len
                    }
                    loss_v, summary, v, y = sess.run(
                        [loss, merged, viterbi_sequence, label],
                        feed_dict=feed_dict)
                    loss_sum += loss_v

                    precision, recall = cal(v, y)
                test_writer.add_summary(summary, global_cnt)
                print("\n**************Validation results****************")
                print('loss_avg: {:.3f}'.format(loss_sum / (config.val_size // config.batch_size)),
                      'precision: {:.3f}'.format(precision),
                      'recall: {:.3f}'.format(recall),
                      )
                print("************************************************\n")

            ## save model
            if epoch % config.snapshot_interval == 0:
                saver.save(sess, os.path.join(config.log_model_dir, 'epoch-{}'.format(epoch)),
                           global_step=global_cnt)

        print('Training is done, exit.')
        # prediction_out = np.array(sess.run(preds, feed_dict={placeholders['data']: d.test, global_steps: global_cnt,
        #                                 placeholders['is_training']: True}))[:, 1]
        # pd.DataFrame({'ID': pd.read_csv(config.test_path)['ID'], 'Pred': prediction_out}).to_csv(os.path.join(config.submission_path, "{}.csv".format("bilstm_1")), index=False)
        # print('Prediction is done, out')


def metric(pred, true):
    entity_true, entity_pred = [], []
    start = 0
    while start < len(true):
        if 0 < true[start] < 13:
            end = start + 1
            while end < len(true) and true[end] == true[start] + 1:
                end += 1
            entity_true.append((start, end - 1, true[start] // 2))
            start = end
        else:
            start += 1
    start = 0
    while start < len(pred):
        if 0 < pred[start] < 13:
            end = start + 1
            while end < len(pred) and pred[end] == pred[start] + 1:
                end += 1
            entity_pred.append((start, end - 1, pred[start] // 2))
            start = end
        else:
            start += 1
    tp = len(set(entity_pred).intersection(set(entity_true)))
    return tp / max(entity_pred.__len__(), 1), tp / max(entity_true.__len__(), 1)


def cal(preds, trues):
    p, r = 0, 0
    for i in range(preds.shape[0]):
        p1, r1 = metric(preds[i], trues[i])
        p += p1
        r += r1
    return p / preds.shape[0], r / preds.shape[0]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        os._exit(1)
