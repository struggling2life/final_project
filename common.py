import os


class config:

    path = "./"
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")
    word2vecmodel_path = os.path.join(path, "model")
    data_classic_train_path = os.path.join(path, "train_classic.pickle")
    data_classic_test_path = os.path.join(path, "test_classic.pickle")
    submission_path = os.path.join(path, "submission/")
    log_dir = os.path.join(path, 'train_log')

    ensemble_path = os.path.join(submission_path, 'ensemble.csv')
    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    embeddingsize = 100
    word2vec_inter = 80000
    padding_size = 512
    components = 1000

    train_size = 589
    val_size = 80
    test_size = 2712
    batch_size = 32
    weight_decay = 0.2

    shape = (padding_size, embeddingsize)
    nr_epoch = 60

    vocab_size = 17780
    lr = 0.001
    show_interval = 30
    test_interval = 5
    snapshot_interval = 5
