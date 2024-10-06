import numpy as np
import math
import torch
import os
import torch.nn.functional as F


class EarlyStopping(object):
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_metric = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, criteria):
        is_best = False
        if self.best_loss is None:
            is_best = True
            self.best_metric = criteria
            self.best_loss = loss
        elif criteria < self.best_metric:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if criteria > self.best_metric:
                is_best = True
            self.best_loss = np.min((loss, self.best_loss))
            self.best_metric = np.max((criteria, self.best_metric))
            self.counter = 0
        return self.early_stop, is_best

# class EarlyStopping:
#     """Early stops the training if validation loss doesn't improve after a given patience."""
#     def __init__(self, save_path, patience=10, verbose=False, delta=0):
#         """
#         Args:
#             save_path :
#             patience (int): How long to wait after last time validation loss improved.
#                             Default: 7
#             verbose (bool): If True, prints a message for each validation loss improvement.
#                             Default: False
#             delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#                             Default: 0
#         """
#         self.save_path = save_path
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = np.Inf
#         self.delta = delta
#
#     def __call__(self, val_loss, model):
#
#         score = -val_loss
#
#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0
#
#     def save_checkpoint(self, val_loss, model):
#         '''Saves model when validation loss decrease.'''
#         if self.verbose:
#             print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
#         path = os.path.join(self.save_path, 'rating.pkt')
#         torch.save(model.state_dict(), path)
#         self.val_loss_min = val_loss
#

class Evaluate:
    def __init__(self, max_length):
        self.idcg_list = self.get_idcg_list(max_length)

    @staticmethod
    def get_idcg_list(max_length):
        idcg_list = [0]
        idcg = 0.0
        for idx in range(max_length):
            idcg = idcg + math.log(2) / math.log(idx + 2)
            idcg_list.append(idcg)
        return idcg_list

    def get_hit_ndcg_mrr(self, positive_predict_list, negative_predict_list, top_k):
        positive_predict_list = list(positive_predict_list)
        negative_predict_list = list(negative_predict_list)
        positive_length = len(positive_predict_list)
        target_length = min(positive_length, top_k)

        all_predict_list = positive_predict_list
        all_predict_list.extend(negative_predict_list)
        sort_index = np.argsort(all_predict_list)[::-1]

        hit_k = 0.0
        dcg_k = 0.0
        mrr_k = 0.0
        ap_k = 0.0
        num_hits = 0
        for idx in range(min(len(sort_index), top_k)):
            ranking = sort_index[idx]
            rank_k = idx + 1
            if ranking < positive_length:
                hit_k = hit_k + 1.0
                num_hits = num_hits + 1
                dcg_k = dcg_k + math.log(2) / math.log(idx + 2)
                mrr_k = mrr_k + 1.0 / rank_k
                ap_k += num_hits / rank_k

        if num_hits == 0:
            ap_k = 0.0
        else:
            ap_k /= num_hits

        hit_k = hit_k / target_length
        idcg = self.idcg_list[target_length]
        ndcg_k = dcg_k / idcg
        mrr_k = mrr_k / target_length

        return hit_k, ndcg_k, mrr_k, ap_k


def compute_pre_distill_loss(pre_a, pre_b):
    distill_loss = - torch.mean(pre_b.detach() * torch.log(pre_a) + (1 - pre_b.detach()) * torch.log(1 - pre_a))
    return distill_loss
