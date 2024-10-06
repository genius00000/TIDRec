import argparse
import logging as log
import os
import torch
from my_config import My_config, update_default_config
from datautils.data import Data
from model.trainer import Trainer
from utils import set_random_seed
torch.cuda.current_device()


def initialize(seed=0):
    log.basicConfig(level=log.INFO)
    set_random_seed(seed)


def update_config(conf):
    if conf.data_name == 'dblp':
        conf.num_authors = 209957
        conf.num_papers = 100276
    elif conf.data_name == 'wos':
        conf.num_authors = 8851
        conf.num_papers = 6037
    else:
        raise NotImplementedError('Wrong data name')

    update_default_config(conf)
    conf.cur_out_path = os.path.join(conf.out_path, conf.data_name + str(conf.train_ratio) + str(conf.conf_name))
    os.makedirs(conf.cur_out_path, exist_ok=True)
    conf.log_path = os.path.join(conf.cur_out_path, 'log.txt')
    conf.result_path = os.path.join(conf.cur_out_path, 'result.txt')


def batch_test_model(conf):
    train_ratio_list = [0.2, 0.4, 0.6, 0.8]
    for train_ratio in train_ratio_list:
        conf.train_ratio = train_ratio
        update_config(conf)
        dataset = Data(conf)
        trainer = Trainer(conf, dataset)
        trainer.test()


def batch_train_model(conf):
    train_ratio_list = [0.2, 0.4, 0.6, 0.8]
    for train_ratio in train_ratio_list:
        conf.train_ratio = train_ratio
        update_config(conf)
        dataset = Data(conf)
        trainer = Trainer(conf, dataset)
        trainer.train()


def main(conf):
    if conf.batch_test_model:
        print('batch_test_model')
        batch_test_model(conf)
        return
    if conf.batch_train_model:
        print('batch_train_model')
        batch_train_model(conf)
        return
    update_config(conf)
    if conf.train_model:
        conf.save()
    dataset = Data(conf)
    print(dataset)
    trainer = Trainer(conf, dataset)
    if conf.train_model:
        trainer.train()
    if conf.test_model:
        trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment setup')
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--data_name', default='wos', type=str)
    parser.add_argument('--out_path', default='./output', type=str)
    parser.add_argument('--conf_name', default='tidrec', type=str)

    parser.add_argument('--emb_dim', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--num_train_negatives', default=8, type=int)
    parser.add_argument('--num_eval_negatives', default=1000, type=int)
    parser.add_argument('--num_test_negatives', default=1000, type=int)
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--num_proc', default=4, type=int)

    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--test_batch_size', default=512, type=int)
    parser.add_argument('--eval_epochs', default=1, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    parser.add_argument('--patience', default=30, type=int)
    parser.add_argument('--loss', default='cross', type=str)

    parser.add_argument('--extra_message', default='', type=str)
    parser.add_argument('--train_ratio', default=0.8, type=float)
    parser.add_argument('--train_model', default='True', action='store_true')
    parser.add_argument('--test_model', default='True', action='store_true')
    parser.add_argument('--batch_train_model', action='store_true')
    parser.add_argument('--batch_test_model', action='store_true')

    args = vars(parser.parse_args())
    conf = My_config(args)
    initialize(conf.seed)
    os.makedirs(conf.out_path, exist_ok=True)
    if torch.cuda.is_available():
        conf.device = 'cuda'
    else:
        conf.device = 'cpu'

    # net = model()
    # total = sum([param.nelement() for param in net.parameters()])
    # print("Number of parameter: %.2fM" % total)


    print(conf.__dict__)


    main(conf)
