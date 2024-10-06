import os
from scipy.io import loadmat
from tidrec.utils import *


def load_data_v1(conf):
    data_write = loadmat(os.path.join(conf.data_dir, conf.data_name, 'author-paper.mat'))
    data_co = loadmat(os.path.join(conf.data_dir, conf.data_name, 'author-author.mat'))
    author_pair_set = read_co_data_v1(data_co['author-author'])
    reverse_author_pair_set = reverse_pair_set(author_pair_set)
    all_author_pair_set = author_pair_set | reverse_author_pair_set
    all_ui_pair_dict = read_write_data_v1(data_write['author-paper'])

    return all_author_pair_set, all_ui_pair_dict


def read_co_data_v1(co_data):
    author_pair_set = set()
    for row in range(co_data.shape[0]):
        u1_id, u2_id = int(co_data[row][0]) - 1, int(co_data[row][1]) - 1
        author_pair_set.add((u1_id, u2_id))
    return author_pair_set


def read_write_data_v1(write_data):
    ui_pair_dict = defaultdict(set)
    for row in range(write_data.shape[0]):
        write_paper = write_data[row]
        author_id, paper_id = int(write_paper[0]) - 1, int(write_paper[1]) - 1,
        ui_pair_dict[author_id].add(paper_id)
    return ui_pair_dict 


def load_data_v2(conf):
    dir_path = os.path.join(conf.data_dir, conf.data_name)
    co_filename = os.path.join(dir_path, conf.data_name + '.links')
    write_filename = os.path.join(dir_path, conf.data_name + '.rating')
    author_pair_set = read_co_data_v2(co_filename)
    reverse_author_pair_set = reverse_pair_set(author_pair_set)
    all_author_pair_set = author_pair_set | reverse_author_pair_set
    all_ui_pair_dict = read_write_data_v2(write_filename)

    return all_author_pair_set, all_ui_pair_dict


def read_co_data_v2(filename):
    with open(filename) as f:
        author_pair_set = set()
        for line in f:
            arr = line.split("\t")
            u1_id, u2_id = int(arr[0]), int(arr[1])
            author_pair_set.add((u1_id, u2_id))
        return author_pair_set


def read_write_data_v2(filename):
    with open(filename) as f:
        ui_pair_dict = defaultdict(set)
        for line in f:
            arr = line.split("\t")
            author_id, paper_id = int(arr[0]), int(arr[1])
            ui_pair_dict[author_id].add(paper_id)
        return ui_pair_dict
