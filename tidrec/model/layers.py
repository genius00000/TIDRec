import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class CoConv(nn.Module):
    def __init__(self):
        super(CoConv, self).__init__()

    def forward(self, graph, author_emb):
        graph = graph.local_var()
        graph.nodes['author'].data['feat'] = author_emb
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_feat'), etype='friend')
        return graph.nodes['author'].data['new_feat']


class WriteConv(nn.Module):
    def __init__(self, conf, emb_dim):
        super(WriteConv, self).__init__()
        self.conf = conf

    def forward(self, graph, author_emb, paper_emb, u_sw, i_sw):
        graph = graph.local_var()
        graph.nodes['author'].data['feat'] = author_emb
        graph.nodes['paper'].data['feat'] = paper_emb

        # ************************************************************************* #
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')
        # ************************************************************************* #

        graph.nodes['author'].data['feat'] = graph.nodes['author'].data['new_f'] + graph.nodes['author'].data['feat'] * u_sw
        graph.nodes['paper'].data['feat'] = graph.nodes['paper'].data['new_f'] + graph.nodes['paper'].data['feat'] * i_sw

        # ************************************************************************* #
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='like')
        graph.update_all(fn.copy_src('feat', 'n_feat'), fn.mean('n_feat', 'new_f'), etype='rev_like')
        # ************************************************************************* #

        graph.nodes['author'].data['feat'] = graph.nodes['author'].data['new_f'] + graph.nodes['author'].data['feat'] * u_sw
        graph.nodes['paper'].data['feat'] = graph.nodes['paper'].data['new_f'] + graph.nodes['paper'].data['feat'] * i_sw

        return graph.nodes['author'].data['feat'], graph.nodes['paper'].data['feat']
