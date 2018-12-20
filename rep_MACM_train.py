import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import numpy as np
import glob, os, sys
import pickle
import math
import signal
import configparser
import ast

from rep_MACM_builder import MultiMatch
from lib.torch_utils import non_neg_normalize, np_softmax
from lib.data_utils import list_shuffle, pad_batch_list
from lib.eval import write_run, compute_ndcg, compute_map
from functools import partial

def load_dataset(path=None):
    '''load the train and test datasets'''
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def max_len(D):
    maxlen = 0
    for doc in D:
        current_len = len(doc)
        if current_len > maxlen:
            maxlen = current_len
    return maxlen


def prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, max_q_len):
    """sample randomly a query
    for a given query, sample doc according to the distro softmax(BM25_scores)
    q_sample_size: num of queries sampled from one data pkl file
    docpair_sample_size: for each q, how many pairs of (D+, D-) sampled
    """
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    Q_counter = 0
    while Q_counter < q_sample_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_idx = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_idx = q_idx[0]  # idx from 0 to len(q_list)
        topic_num = q_list[q_idx]
        query = data[topic_num]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[topic_num]['docs']
            scores = data[topic_num]['scores']
            if len(docs) >=2:
                # calcuate BM25 score softmax distribution
                np_scores = np.asarray(scores)
                BM25_distro = np_softmax(np_scores)
                idx = np.random.choice(range(len(docs)), size=(docpair_sample_size, 2), replace=True, p=BM25_distro)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
                Q_counter += 1
    return [Q, D_pos, D_neg, label]


def train(config_path, resume_training=False):
    '''training process'''
    # parse config
    config = configparser.ConfigParser()
    config.read(config_path)
    '''hyper params'''
    model_name_str = config['hyperparams']['model_name_str']
    batch_size = ast.literal_eval(config['hyperparams']['batch_size'])  # batch_size
    vocab_size = ast.literal_eval(config['hyperparams']['vocab_size'])  # vocab_size
    emb_size = ast.literal_eval(config['hyperparams']['emb_size'])  # embedding dimension
    hidden_size = ast.literal_eval(config['hyperparams']['hidden_size'])
    dropout = ast.literal_eval(config['hyperparams']['dropout'])
    q_filt1 = ast.literal_eval(config['hyperparams']['q_filt1'])
    q_filt2 = ast.literal_eval(config['hyperparams']['q_filt2'])
    q_filt3 = ast.literal_eval(config['hyperparams']['q_filt3'])
    q_stride1 = ast.literal_eval(config['hyperparams']['q_stride1'])
    q_stride2 = ast.literal_eval(config['hyperparams']['q_stride2'])
    q_stride3 = ast.literal_eval(config['hyperparams']['q_stride3'])
    d_filt1 = ast.literal_eval(config['hyperparams']['d_filt1'])
    d_filt2 = ast.literal_eval(config['hyperparams']['d_filt2'])
    d_filt3 = ast.literal_eval(config['hyperparams']['d_filt3'])
    d_stride1 = ast.literal_eval(config['hyperparams']['d_stride1'])
    d_stride2 = ast.literal_eval(config['hyperparams']['d_stride2'])
    d_stride3 = ast.literal_eval(config['hyperparams']['d_stride3'])
    Mtopk = ast.literal_eval(config['hyperparams']['Mtopk'])
    preemb = ast.literal_eval(config['hyperparams']['preemb'])
    preemb_path = config['hyperparams']['preemb_path']
    sim_type = config['hyperparams']['sim_type']
    hinge_margin = ast.literal_eval(config['hyperparams']['hinge_margin'])
    train_datablock_size = ast.literal_eval(config['hyperparams']['train_datablock_size'])
    # for sampleQ sampleD prepare_data()
    q_sample_size = ast.literal_eval(config['hyperparams']['q_sample_size'])
    docpair_sample_size = ast.literal_eval(config['hyperparams']['docpair_sample_size'])
    n_epoch = ast.literal_eval(config['hyperparams']['n_epoch']) # num of epochs
    alpha = ast.literal_eval(config['hyperparams']['alpha'])  # weight decay
    learning_rate = ast.literal_eval(config['hyperparams']['learning_rate'])
    # q and doc cuts
    q_len = ast.literal_eval(config['hyperparams']['q_len'])
    d_len = ast.literal_eval(config['hyperparams']['d_len'])
    # base_path
    data_base_path = config['hyperparams']['data_base_path']  #'/scratch/nyfbb'
    model_base_path = config['hyperparams']['model_base_path']  #'/home/nyfbb/exp'
    # use cuda flag
    use_cuda = True

    '''TRAINING DIR'''
    TRAIN_DIR = '{}/train/'.format(data_base_path)
    train_files = glob.glob("{}/data*.pkl".format(TRAIN_DIR))

    '''build model'''
    model = MultiMatch(BS=batch_size, q_len=q_len, d_len=d_len,
                       q_filt1=q_filt1, q_filt2=q_filt2, q_filt3=q_filt3,
                       q_stride1=q_stride1, q_stride2=q_stride2, q_stride3=q_stride3,
                       d_filt1=d_filt1, d_filt2=d_filt2, d_filt3=d_filt3,
                       d_stride1=d_stride1, d_stride2=d_stride2, d_stride3=d_stride3,
                       intermat_topk=Mtopk, vocab_size=vocab_size,
                       emb_size=emb_size, hidden_size=hidden_size,
                       dropout=dropout, sim_type=sim_type, preemb=preemb,
                       preemb_path=preemb_path)
    if use_cuda:
        model.cuda()
    # optimizer
    #optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
    # loss func
    loss = nn.MarginRankingLoss(margin=hinge_margin, size_average=True)
    # experiment
    print("Experiment")
    if resume_training == False:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
    else:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
    # model_file
    model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
    """TRAINING"""
    best_MAP = 0.0
    # restore saved parameter if resume_training is true
    if resume_training == True:
        model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
        model.load_state_dict(torch.load(model_file))
        with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'rb') as f_MAP:
            best_MAP = pickle.load(f_MAP)
        print("loaded model, and resume training now")
    # preparing batch data
    file_count = 0
    # gen Hingeloss Y labels always 1 (S+ > S-)
    Y = Variable(torch.ones(batch_size), requires_grad=False).cuda()
    for epoch in range(1, n_epoch + 1):
        train_files = list_shuffle(train_files)
        for f in train_files:
            '''load_data'''
            data = load_dataset(f)
            print("loaded {}".format(f))
            '''prepare_data'''
            [Q, D_pos, D_neg, L] = prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, q_len)
            ''' shuffle data'''
            train_data = list_shuffle(Q, D_pos, D_neg, L)
            '''training func'''
            batch_count_tr = 0
            num_batch = len(train_data[0]) // batch_size

            for batch_count in range(num_batch):
                Q = train_data[0][batch_size * batch_count: batch_size * (batch_count + 1)]
                D_pos = train_data[1][batch_size * batch_count: batch_size * (batch_count + 1)]
                D_neg = train_data[2][batch_size * batch_count: batch_size * (batch_count + 1)]
                L = train_data[3][batch_size * batch_count: batch_size * (batch_count + 1)]
                Q = Variable(torch.LongTensor(pad_batch_list(Q, max_len=q_len, padding_id=0)), requires_grad=False).cuda()
                D_pos = Variable(torch.LongTensor(pad_batch_list(D_pos, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
                D_neg = Variable(torch.LongTensor(pad_batch_list(D_neg, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
                L = Variable(torch.FloatTensor(L), requires_grad=False).cuda()
                # run on this batch
                optimizer.zero_grad()
                t1 = time.time()
                S_pos, S_neg = model(Q, D_pos, D_neg)
                Loss = loss(S_pos, S_neg, Y)
                Loss.backward()
                optimizer.step()
                t2 = time.time()
                batch_count_tr +=1
                print("epoch {} batch {} training cost: {} using {}s" \
                .format(epoch, batch_count_tr, Loss.data[0], t2-t1))
                f_log.write("epoch {} batch {} training cost: {}, using {}s".format(epoch, batch_count_tr, Loss.data[0], t2 - t1) + '\n')
            file_count += 1
            if file_count % 4 == 0:
                # do rapid validation
                ndcg_list, mapvalue = validation(model, model_name_str,
                                                 data_base_path, model_base_path)  # pass the training compuational graph placeholder to valid function to evaluate with the same set of parameters
                print("epoch :{}, pkl count: {}, NDCG".format(epoch, file_count), ndcg_list)
                print("MAP: {}".format(mapvalue))
                # check if this valid period is the best and update best_MAP, save model to disk
                if mapvalue > best_MAP:
                    best_MAP = mapvalue
                    with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'wb') as f_MAP:
                        pickle.dump(best_MAP, f_MAP)
                    # save model params after several epoch
                    model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
                    torch.save(model.state_dict(), model_file)
                    print("successfully saved model to the path {}".format(model_file))
                valid_log.write("{} {} {} {}".format(ndcg_list[1][0], ndcg_list[1][1], ndcg_list[1][2], ndcg_list[1][3]))
                valid_log.write(" MAP: {}".format(mapvalue))
                valid_log.write('\n')
    f_log.close()
    valid_log.close()


def validation(model, model_name_str, data_base_path, model_base_path, compute_ndcg_flag=True):
    '''model: trained model object
    model_name_str: model name str
    '''
    '''hyper params'''
    batch_size = 128  # batch_size
    # q and doc cuts
    q_len = 15
    d_len = 1000
    #model_base_path = "/home/nyfbb/exp"
    #data_base_path = "/scratch/nyfbb"
    '''VALID DIR'''
    TEST_DIR = '{}/aol/valid/WT0912/'.format(data_base_path)
    RESULTS_DIR = '{}/{}/result/valid/'.format(model_base_path, model_name_str)
    test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))

    # run path
    run_path = RESULTS_DIR + 'run.txt'
    # run list containing run lines of all topics
    all_run_list = []
    # load one testfile at a time and conduct test
    for f in test_files:
        # list to contains scores
        # load testdata data = {'topic_num': {'query':[], 'docs':[], 'docno':[]}}
        data = load_dataset(f)
        print("len valid data", len(data))
        # generate full format [Q, D, meta_dict] meta_dict={'topic_num':[], 'docno':[]} for one topic group
        for topic_num in data:
            Q = []
            D = []
            meta_dict = {'topic_num':[], 'docno':[]}
            batch_id = 0
            num_batch = int(math.ceil(len(data[topic_num]['docs']) * 1.0 / batch_size))
            for i in range(len(data[topic_num]['docs'])):
                Q.append(data[topic_num]['query'])
                D.append(data[topic_num]['docs'][i])
                meta_dict['topic_num'].append(topic_num)
                meta_dict['docno'].append(data[topic_num]['docno'][i])
            # padding
            Q_test = Variable(torch.LongTensor(pad_batch_list(Q, max_len=q_len, padding_id=0)), requires_grad=False).cuda()
            D_test = Variable(torch.LongTensor(pad_batch_list(D, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
            scores = []
            for batch_id in range(num_batch):
                Q_value = Q_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                D_value = D_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                batch_rel, _ = model(Q_value, D_value, D_value)  # in test phase, no dropout
                batch_scores = batch_rel.data.cpu().numpy().tolist()
                scores += batch_scores
            np_scores = np.asarray(scores)
            np_scores = non_neg_normalize(np_scores)
            scores = np_scores.tolist()
            run_list = zip(meta_dict['topic_num'], meta_dict['docno'], scores)
            print("run_file for topic {} created".format(topic_num))
            all_run_list += run_list
    write_run(all_run_list, run_path)

    if compute_ndcg_flag==True:
        # prepare run file list
        runfile_dir = RESULTS_DIR
        rel_path = '{}/{}/tmp/valid/qrels.1-200.clueweb'.format(model_base_path, model_name_str)
        tmp_path = '{}/{}/tmp/valid/temp.txt'.format(model_base_path, model_name_str)
        # compute ndcg by calling external tools
        ndcg_list = compute_ndcg(run_path, rel_path, tmp_path)
        mapvalue = compute_map(run_path, rel_path, tmp_path)
        return ndcg_list, mapvalue


def test(model_path, config_path, compute_ndcg_flag=True):
    # parse config
    config = configparser.ConfigParser()
    config.read(config_path)
    '''hyper params'''
    model_name_str = config['hyperparams']['model_name_str']
    batch_size = ast.literal_eval(config['hyperparams']['batch_size'])  # batch_size
    vocab_size = ast.literal_eval(config['hyperparams']['vocab_size'])  # vocab_size
    emb_size = ast.literal_eval(config['hyperparams']['emb_size'])  # embedding dimension
    hidden_size = ast.literal_eval(config['hyperparams']['hidden_size'])
    dropout = ast.literal_eval(config['hyperparams']['dropout'])
    q_filt1 = ast.literal_eval(config['hyperparams']['q_filt1'])
    q_filt2 = ast.literal_eval(config['hyperparams']['q_filt2'])
    q_filt3 = ast.literal_eval(config['hyperparams']['q_filt3'])
    q_stride1 = ast.literal_eval(config['hyperparams']['q_stride1'])
    q_stride2 = ast.literal_eval(config['hyperparams']['q_stride2'])
    q_stride3 = ast.literal_eval(config['hyperparams']['q_stride3'])
    d_filt1 = ast.literal_eval(config['hyperparams']['d_filt1'])
    d_filt2 = ast.literal_eval(config['hyperparams']['d_filt2'])
    d_filt3 = ast.literal_eval(config['hyperparams']['d_filt3'])
    d_stride1 = ast.literal_eval(config['hyperparams']['d_stride1'])
    d_stride2 = ast.literal_eval(config['hyperparams']['d_stride2'])
    d_stride3 = ast.literal_eval(config['hyperparams']['d_stride3'])
    Mtopk = ast.literal_eval(config['hyperparams']['Mtopk'])
    preemb = ast.literal_eval(config['hyperparams']['preemb'])
    sim_type = config['hyperparams']['sim_type']
    hinge_margin = ast.literal_eval(config['hyperparams']['hinge_margin'])
    train_datablock_size = ast.literal_eval(config['hyperparams']['train_datablock_size'])
    # for sampleQ sampleD prepare_data()
    q_sample_size = ast.literal_eval(config['hyperparams']['q_sample_size'])
    docpair_sample_size = ast.literal_eval(config['hyperparams']['docpair_sample_size'])
    n_epoch = ast.literal_eval(config['hyperparams']['n_epoch']) # num of epochs
    alpha = ast.literal_eval(config['hyperparams']['alpha'])  # weight decay
    learning_rate = ast.literal_eval(config['hyperparams']['learning_rate'])
    # q and doc cuts
    q_len = ast.literal_eval(config['hyperparams']['q_len'])
    d_len = ast.literal_eval(config['hyperparams']['d_len'])
    # base_path
    data_base_path = config['hyperparams']['data_base_path'] #'/scratch/nyfbb'
    model_base_path = config['hyperparams']['model_base_path'] #'/home/nyfbb/exp'
    # use cuda flag
    use_cuda = True

    '''TESTING DIR'''
    TEST_DIR = '{}/aol/test51-200/WT0912/'.format(data_base_path)
    RESULTS_DIR = '{}/{}/result/test/'.format(model_base_path, model_name_str)
    test_files = glob.glob("{}/data*.pkl".format(TEST_DIR))

    '''build model'''
    model = MultiMatch(BS=batch_size, q_len=q_len, d_len=d_len,
                       q_filt1=q_filt1, q_filt2=q_filt2, q_filt3=q_filt3,
                       q_stride1=q_stride1, q_stride2=q_stride2, q_stride3=q_stride3,
                       d_filt1=d_filt1, d_filt2=d_filt2, d_filt3=d_filt3,
                       d_stride1=d_stride1, d_stride2=d_stride2, d_stride3=d_stride3,
                       intermat_topk=Mtopk, vocab_size=vocab_size,
                       emb_size=emb_size, hidden_size=hidden_size,
                       dropout=dropout, sim_type=sim_type, preemb=preemb,
                       preemb_path='{}/aol/glove_emb.pkl'.format(data_base_path))
    # placeholders
    # optimizer

    # Test
    # load model from file
    model.load_state_dict(torch.load(model_path))
    print("loaded model, and perform test now")
    if use_cuda:
        model.cuda()
    # define output run file path
    run_path = RESULTS_DIR + 'run.txt'
    # run list containing run lines of all topics
    all_run_list = []
    # load one testfile at a time and conduct test
    for f in test_files:
        # list to contains scores
        # load testdata data = {'topic_num': {'query':[], 'docs':[], 'docno':[]}}
        data = load_dataset(f)
        # generate full format [Q, D, meta_dict] meta_dict={'topic_num':[], 'docno':[]} for one topic group
        for topic_num in data:
            Q = []
            D = []
            meta_dict = {'topic_num':[], 'docno':[]}
            batch_id = 0
            num_batch = int(math.ceil(len(data[topic_num]['docs']) * 1.0 / batch_size))
            for i in range(len(data[topic_num]['docs'])):
                Q.append(data[topic_num]['query'])
                D.append(data[topic_num]['docs'][i])
                meta_dict['topic_num'].append(topic_num)
                meta_dict['docno'].append(data[topic_num]['docno'][i])
            # padding
            Q_test = Variable(torch.LongTensor(pad_batch_list(Q, max_len=q_len, padding_id=0)), requires_grad=False).cuda()
            D_test = Variable(torch.LongTensor(pad_batch_list(D, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
            scores = []
            for batch_id in range(num_batch):
                Q_value = Q_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                D_value = D_test[batch_id * batch_size: (batch_id + 1)* batch_size]
                batch_rel, _ = model(Q_value, D_value, D_value)
                batch_scores = batch_rel.data.cpu().numpy().tolist()
                scores += batch_scores
            np_scores = np.asarray(scores)
            np_scores = non_neg_normalize(np_scores)
            scores = np_scores.tolist()
            run_list = zip(meta_dict['topic_num'], meta_dict['docno'], scores)
            print("run_file for topic {} created".format(topic_num))
            all_run_list += run_list
    write_run(all_run_list, run_path)

    if compute_ndcg_flag==True:
        # prepare run file list
        runfile_dir = RESULTS_DIR
        #comb_run_path = 'model_base_path/{}/tmp/test/run_all.txt'
        rel_path = '{}/{}/tmp/test/qrels.1-200.clueweb'.format(model_base_path, model_name_str)
        tmp_path = '{}/{}/tmp/test/temp.txt'.format(model_base_path, model_name_str)
        # compute ndcg by calling external tools
        ndcg_list = compute_ndcg(run_path, rel_path, tmp_path)
        mapvalue = compute_map(run_path, rel_path, tmp_path)
        return ndcg_list, mapvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--resume", type=str, default="True")
    args = parser.parse_args()

    if args.mode == "train":
        if args.resume == "True":
            train(args.path, resume_training=True)
        elif args.resume == "False":
            train(args.path, resume_training=False)
        else:
            raise ValueError("invalid resume flag")
    if args.mode == "test":
        config = configparser.ConfigParser()
        config.read(args.path)
        model_name_str = config['hyperparams']['model_name_str']
        model_base_path = config['hyperparams']['model_base_path']
        ndcg_list, mapvalue = test('{}/{}/saves/model_file'.format(model_base_path, model_name_str), config_path)
        print(ndcg_list, mapvalue)

if __name__ == '__main__':
    main()
