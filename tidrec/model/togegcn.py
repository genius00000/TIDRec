import time
import torch
import torch.nn as nn
from tidrec.model.layers import CoConv, WriteConv
import torch.nn.functional as F


class TogeGCN(nn.Module):
    def __init__(self, conf, data):
        super(TogeGCN, self).__init__()
        self.conf = conf
        self.data = data
        self.author_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_authors, conf.emb_dim)))
        self.paper_emb = nn.Parameter(torch.normal(mean=0, std=0.01, size=(conf.num_papers, conf.emb_dim)))

        self.co_layer = CoConv()
        self.write_layer = WriteConv(conf, conf.emb_dim)

    def forward(self, authors, papers, authors_list=None, papers_list=None, mode='test'):
        t1 = time.time()
        graph = self.data.data_graph.local_var()
        if mode == 'train':
            remove_iu_eid_list = []
            remove_ui_eid_list = []
            for idx in range(len(authors_list)):
                iu_eid = self.data.train_iu_pair2eid.get((papers_list[idx], authors_list[idx]), -1)
                ui_eid = self.data.train_ui_pair2eid.get((authors_list[idx], papers_list[idx]), -1)
                if iu_eid >= 0:
                    remove_iu_eid_list.append(iu_eid)
                if ui_eid >= 0:
                    remove_ui_eid_list.append(ui_eid)
            graph.remove_edges(remove_iu_eid_list, 'rev_like')
            graph.remove_edges(remove_ui_eid_list, 'like')

        fuse_author_emb = self.author_emb
        fuse_paper_emb = self.paper_emb

        author_deg = graph.out_degrees(etype='like').float().to(self.conf.device).unsqueeze(1)
        u_sw = 1 - author_deg / (author_deg + 1e-8)
        paper_deg = graph.out_degrees(etype='rev_like').float().to(self.conf.device).unsqueeze(1)
        i_sw = 1 - paper_deg / (paper_deg + 1e-8)

        co_author_emb_gnn1 = self.co_layer(graph, fuse_author_emb)
        write_author_emb_gnn1, write_paper_emb_gnn1 = self.write_layer(graph, fuse_author_emb, fuse_paper_emb, u_sw, i_sw)
        author_emb_gnn1 = 0.5 * co_author_emb_gnn1 + 0.5 * write_author_emb_gnn1

        co_author_emb_gnn2 = self.co_layer(graph, co_author_emb_gnn1)
        write_author_emb_gnn2, light_paper_emb_gnn2 = self.write_layer(graph, write_author_emb_gnn1, write_paper_emb_gnn1,
                                                                      u_sw, i_sw)
        author_emb_gnn2 = 0.5 * co_author_emb_gnn2 + 0.5 * write_author_emb_gnn2

        final_author_emb = 0
        if 0 in self.conf.l_author:
            final_author_emb = final_author_emb + fuse_author_emb
        if 1 in self.conf.l_author:
            final_author_emb = final_author_emb + author_emb_gnn1
        if 2 in self.conf.l_author:
            final_author_emb = final_author_emb + author_emb_gnn2

        final_paper_emb = fuse_paper_emb
        latest_author_emb = final_author_emb[authors]
        latest_paper_emb = final_paper_emb[papers]

        predict = torch.sigmoid(torch.sum(torch.mul(latest_author_emb, latest_paper_emb), dim=1))

        t2 = time.time()
        inference_time = t2 - t1
        print('TogeGCN Inference time:', inference_time)

        # print("twingcn output")
        # # print(latest_author_emb)
        # # print(latest_paper_emb)
        # print(predict)
        # print(predict.size())

        return predict, latest_author_emb, latest_paper_emb
