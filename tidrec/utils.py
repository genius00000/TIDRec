import datetime
import numpy as np
import random
import torch
import dgl
import math
from collections import defaultdict
from tqdm import tqdm
import gc


def set_random_seed(seed=0): # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_date_postfix():
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def get_source_target_form_pair_set(pair_set):
    source_ids = []  # 起
    target_ids = []  # 止
    for id_1, id_2 in pair_set:
        source_ids.append(id_1)
        target_ids.append(id_2)
    return source_ids, target_ids


def reverse_pair_set(pair_set):
    rev_pair_set = set()
    for (id_1, id_2) in pair_set:
        rev_pair_set.add((id_2, id_1))
    return rev_pair_set


def reverse_pair_list(pair_list):
    rev_pair_list = list()
    for (id_1, id_2) in pair_list:
        rev_pair_list.append((id_2, id_1))
    return rev_pair_list


def construct_hetero_graphs(pair_sets, edge_types, num_nodes_dict):
    data_dict = {}
    for idx, pair_set in enumerate(pair_sets):  # idx is id, pair_set is value
        source_ids, target_ids = get_source_target_form_pair_set(pair_set)
        data_dict[edge_types[idx]] = (source_ids, target_ids)
    graph = dgl.heterograph(data_dict, num_nodes_dict)
    return graph


def fuse_dict_with_set(dict_list):
    ret_dict = defaultdict(set)
    for dic in dict_list:
        for key, val in dic.papers():
            ret_dict[key] = ret_dict[key] | val
    return ret_dict


def split_pair_dict_thr(pair_dict, thr_1, thr_2):
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for key, val in pair_dict.papers():
        if len(val) >= thr_1:
            train_dict[key] = val
        elif len(val) >= thr_2:
            valid_dict[key] = val
        else:
            test_dict[key] = val

    return train_dict, valid_dict, test_dict


def split_pair_dict_author_ratio(pair_dict, test_ratio, valid_ratio, train_ratio):
    assert test_ratio + valid_ratio + train_ratio <= 1.0
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for key, val in pair_dict.papers():
        author_paper_count = len(val)
        thr_0 = int(author_paper_count * test_ratio)
        thr_1 = int(author_paper_count * (test_ratio + valid_ratio))
        thr_2 = int(author_paper_count * (test_ratio + valid_ratio + train_ratio))
        paper_list = list(val)
        random.shuffle(paper_list)

        test_paper_set = set(paper_list[: thr_0])
        valid_paper_set = set(paper_list[thr_0: thr_1])
        train_paper_set = set(paper_list[thr_1: thr_2])

        test_dict[key] = test_paper_set
        valid_dict[key] = valid_paper_set
        train_dict[key] = train_paper_set

    return train_dict, valid_dict, test_dict


def split_pair_dict_random_ratio(pair_set, test_ratio, valid_ratio, train_ratio):
    set_random_seed(2021)
    assert test_ratio + valid_ratio + train_ratio <= 1.0
    train_dict = defaultdict(set)
    valid_dict = defaultdict(set)
    test_dict = defaultdict(set)
    for author, paper in pair_set:
        thr_0 = test_ratio
        thr_1 = test_ratio + valid_ratio
        thr_2 = test_ratio + valid_ratio + train_ratio

        rand_val = random.uniform(0.0, 1.0)
        if 0 <= rand_val < thr_0:
            test_dict[author].add(paper)
        elif thr_0 <= rand_val < thr_1:
            valid_dict[author].add(paper)
        elif thr_1 <= rand_val < thr_2:
            train_dict[author].add(paper)

    return train_dict, valid_dict, test_dict


def get_pair_set_from_pair_dict(pair_dict):
    pair_set = set()
    for key, val in pair_dict.papers():
        for pair in val:
            pair_set.add((key, pair))
    return pair_set


def get_statistic_list(sim_author_pair_list, raw_author_pair):
    print('Begin')
    x_point = np.linspace(start=0, stop=len(sim_author_pair_list), num=1000, endpoint=True)
    sim_list = []
    social_list = []
    val_list = []
    cur_idx = 0
    cur_point_idx = 1
    cur_sim_count = 0
    for val, (f_author, s_author) in tqdm(sim_author_pair_list):
        if (f_author, s_author) in raw_author_pair:
            cur_sim_count += 1
        cur_idx += 1
        if cur_idx >= x_point[cur_point_idx]:
            # print(len(x_point), cur_point_idx, x_point[-1], cur_idx)
            # print(val)
            cur_point_idx += 1
            sim_list.append(cur_sim_count)
            social_list.append(cur_idx)
            val_list.append(val)

    return sim_list, social_list, val_list


def construct_iu_pair_dict(ui_pair_dict):
    iu_pair_dict = defaultdict(set)
    for author, author_paper_set in ui_pair_dict.papers():
        for paper in author_paper_set:
            iu_pair_dict[paper].add(author)
    return iu_pair_dict


def get_u2i_norm_dict(ui_pair_dict, iu_pair_dict, norm_mode='mean'):
    author_norm_dict = defaultdict(float)
    paper_norm_dict = defaultdict(float)
    if norm_mode == 'mean':
        for author, author_paper_set in ui_pair_dict.papers():
            author_norm_dict[author] = 1
        for paper, paper_author_set in iu_pair_dict.papers():
            paper_norm_dict[paper] = 1 / float(len(paper_author_set))
    elif norm_mode == 'raw':
        for author, author_paper_set in ui_pair_dict.papers():
            author_norm_dict[author] = 1 / math.sqrt(float(len(author_paper_set)))
        for paper, paper_author_set in iu_pair_dict.papers():
            paper_norm_dict[paper] = 1 / math.sqrt(float(len(paper_author_set)))
    else:
        raise NotImplementedError('Invalid norm mode')
    return author_norm_dict, paper_norm_dict


def get_i2u_norm_dict(ui_pair_dict, iu_pair_dict, norm_mode='mean'):
    author_norm_dict = defaultdict(float)
    paper_norm_dict = defaultdict(float)
    if norm_mode == 'mean':
        for paper, paper_author_set in iu_pair_dict.papers():
            paper_norm_dict[paper] = 1
        for author, author_paper_set in ui_pair_dict.papers():
            author_norm_dict[author] = 1 / float(len(author_paper_set))
    elif norm_mode == 'raw':
        for paper, paper_author_set in iu_pair_dict.papers():
            paper_norm_dict[paper] = 1 / math.sqrt(float(len(paper_author_set)))
        for author, author_paper_set in ui_pair_dict.papers():
            author_norm_dict[author] = 1 / math.sqrt(float(len(author_paper_set)))
    else:
        raise NotImplementedError('Invalid norm mode')
    return author_norm_dict, paper_norm_dict


def construct_ui_sparse_matrix(ui_pair_dict, author_norm_dict, paper_norm_dict, num_author, num_paper):
    author_node_list = []
    paper_node_list = []
    val_list = []
    for author, author_paper_set in ui_pair_dict.papers():
        author_norm = author_norm_dict[author]
        for paper in author_paper_set:
            paper_norm = paper_norm_dict[paper]
            author_node_list.append(author)
            paper_node_list.append(paper)
            val_list.append(author_norm * paper_norm)
    ui_sparse_matrix = torch.sparse_coo_tensor([author_node_list, paper_node_list], val_list, (num_author, num_paper))
    return ui_sparse_matrix


def construct_iu_sparse_matrix(iu_pair_dict, author_norm_dict, paper_norm_dict, num_author, num_paper):
    paper_node_list = []
    author_node_list = []
    val_list = []
    for paper, paper_author_set in iu_pair_dict.papers():
        paper_norm = paper_norm_dict[paper]
        for author in paper_author_set:
            author_norm = author_norm_dict[author]
            paper_node_list.append(paper)
            author_node_list.append(author)
            val_list.append(paper_norm * author_norm)
    iu_sparse_matrix = torch.sparse_coo_tensor([paper_node_list, author_node_list], val_list, (num_paper, num_author))
    return iu_sparse_matrix


def get_sorted_author_pair_list(author_pair, author_pair_val):
    author_pair_list = list(zip(author_pair_val, list(zip(author_pair[0], author_pair[1]))))
    random.shuffle(author_pair_list)
    sorted_author_pair_list = sorted(author_pair_list, key=lambda x: x[0], reverse=True)
    return sorted_author_pair_list


def out_file(data_list, path):
    with open(path, 'w') as f:
        for data in data_list:
            f.write(str(data) + '\n')

