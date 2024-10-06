import pymysql
import ast
import torch
from scipy.io import loadmat
from collections import defaultdict
import os
from tidrec.utils import *

from datautils.testdata import TestData


def read_social_data_v1(social_data):
    author_pair_set = set()
    for row in range(social_data.shape[0]):
        u1_id, u2_id = int(social_data[row][0]) - 1, int(social_data[row][1]) - 1
        author_pair_set.add((u1_id, u2_id))
    return author_pair_set
def reverse_pair_set(pair_set):
    rev_pair_set = set()
    for (id_1, id_2) in pair_set:
        rev_pair_set.add((id_2, id_1))
    return rev_pair_set
def read_rating_data_v1(rating_data):
    ui_pair_dict = defaultdict(set)
    for row in range(rating_data.shape[0]):
        rating_paper = rating_data[row]
        author_id, paper_id = int(rating_paper[0]) - 1, int(rating_paper[1]) - 1,
        # category, score = int(rating_paper[2]), int(rating_paper[3])
        ui_pair_dict[author_id].add(paper_id)
    return ui_pair_dict

sql_connection = pymysql.connect(host='124.222.163.199', author='root', password='123456',
db='paper-recommend', port=3306, autocommit=False, charset='utf8mb4')
cursor = sql_connection.cursor()

data_rating = loadmat(os.path.join('data/wos/author-paper.mat'))
data_social = loadmat(os.path.join('data/wos/author-author.mat'))
author_pair_set = read_social_data_v1(data_social['author-author'])
reverse_author_pair_set = reverse_pair_set(author_pair_set)
author_pair_set = author_pair_set | reverse_author_pair_set
ui_pair_dict = read_rating_data_v1(data_rating['author-paper'])

dataset = TestData(author_pair_set,ui_pair_dict)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = torch.load('./output/wos0.8design/twin.pt', map_location=torch.device('cuda'))

positive_authors = torch.tensor(dataset.author_list).to(device).long()
positive_papers = torch.tensor(dataset.paper_list).to(device).long()

with torch.no_grad():
    positive_predict, _, _ = model(positive_authors, positive_papers)

positive_predict_dict = defaultdict(list)

for author_id in dataset.author_list:
    positive_predict_dict[author_id] = positive_predict[dataset.author_idx_dict[author_id]] # author_idx_dict[author_id]是author_id在ui_pair_set对应元祖的索引，所以是元组对应的预测概率
    positive_predict_dict[author_id] = positive_predict_dict[author_id].cpu().numpy()
# print(len(positive_predict_dict))


negative_authors = torch.tensor(dataset.collected_author_list).to(device).long()
negative_papers = torch.tensor(dataset.collected_paper_list).to(device).long()


with torch.no_grad():
    negative_predict, _, _ = model(negative_authors, negative_papers)
negative_predict = negative_predict.cpu().numpy().reshape(-1, 1000)
# print(negative_predict.shape)
negative_predict_dict = defaultdict(list)

for idx, author_id in enumerate(dataset.neg_author_list):
    # print(idx, author_id)
    negative_predict_dict[author_id] = negative_predict[idx]


n = 0
for author_id in dataset.author_list:
    pos_sets = list()
    for set_id in dataset.author_idx_dict[author_id]:
        pos_sets.append(list(dataset.ui_pair_set)[set_id])
    pos_predict_end_dict = defaultdict(list)
    for i in range(len(pos_sets)):
        pos_predict_end_dict[pos_sets[i]] = list(positive_predict_dict[author_id])[i]
    neg_sets = list()
    for j in range(1000):
        neg_sets.append((dataset.collected_author_list[dataset.neg_author_list.index(author_id)*1000+j],dataset.collected_paper_list[dataset.neg_author_list.index(author_id)*1000+j]))
    neg_predict_end_dict = defaultdict(list)
    for i in range(len(neg_sets)):
        neg_predict_end_dict[neg_sets[i]] = list(negative_predict_dict[author_id])[i]
    # print(neg_predict_end_dict)
    merge_predict_dict = {**pos_predict_end_dict, **neg_predict_end_dict}
    predict_score_list = list(merge_predict_dict.values())
    sort_index = np.argsort(predict_score_list)[::-1]
    predict_paper_list = list()
    for rank in range(15):
        predict_index = sort_index[rank]
        predict_tuple = list(merge_predict_dict.keys())[predict_index]
        predict_paper_id = tuple(predict_tuple)[1]
        predict_paper_list.append(predict_paper_id+1)
    print(author_id + 1, predict_paper_list)
    n += 1
    sql = "insert into `recommend`(author_id,paper_list) values (" + str(author_id + 1)+",'" + str(predict_paper_list)+"')"
    cursor.execute(sql)
    sql_connection.commit()

