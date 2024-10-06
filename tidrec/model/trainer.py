import time
import os
from tidrec.model.cogcn import CoGCN
from tidrec.model.togegcn import TogeGCN
from tidrec.model.writegcn import WriteGCN
from tidrec.model.model_utils import *
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Process, Queue


class Trainer:
    def __init__(self, conf, data):
        self.conf = conf
        self.data = data
        self.model_dict = self.get_model_dict(conf.model_list)
        self.evaluate = Evaluate(conf.top_k)

    @staticmethod
    def get_model(conf, data, model_name):
        if model_name == 'toge':
            return TogeGCN(conf, data)
        elif model_name == 'co':
            return CoGCN(conf, data)
        elif model_name == 'write':
            return WriteGCN(conf, data)
        else:
            raise NotImplementedError('Invalid model', conf.model)

    def get_model_dict(self, model_list):
        model_dict = {}
        for model_name in model_list:
            model = self.get_model(self.conf, self.data, model_name).to(self.conf.device)
            model_dict[model_name] = {
                'model': model,
                'model_path1': os.path.join(self.conf.cur_out_path, model_name + '.pkt'),
                'model_path2': os.path.join(self.conf.cur_out_path, model_name + '.pt'),
                'optimizer': torch.optim.Adam(model.parameters(), lr=self.conf.lr),
                'criterion': torch.nn.BCELoss(),
                'best_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'train_mrr': 0.0,
                    'train_map': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'valid_mrr': 0.0,
                    'valid_map': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                    'test_mrr': 0.0,
                    'test_map': 0.0
                },
                'cur_perform': {
                    'train_hit': 0.0,
                    'train_ndcg': 0.0,
                    'train_mrr': 0.0,
                    'train_map': 0.0,
                    'valid_hit': 0.0,
                    'valid_ndcg': 0.0,
                    'valid_mrr': 0.0,
                    'valid_map': 0.0,
                    'test_hit': 0.0,
                    'test_ndcg': 0.0,
                    'test_mrr': 0.0,
                    'test_map': 0.0
                }
            }
        return model_dict

    def train_epoch(self, data, model_dict):
        data_loader = data.train_data_loader
        for step, batch_data in enumerate(tqdm(data_loader, desc="Iteration")):
            authors_list, papers_list, labels = batch_data
            authors = torch.tensor(authors_list).to(self.conf.device).long()
            papers = torch.tensor(papers_list).to(self.conf.device).long()
            labels = torch.tensor(labels).to(self.conf.device).float()

            batch_loss = 0
            pre_dict = {}
            for model in model_dict.keys():
                predict, author_emb, paper_emb = \
                    model_dict[model]['model'](authors, papers, authors_list, papers_list, mode='train')
                model_loss = model_dict[model]['criterion'](predict, labels)
                batch_loss = batch_loss + model_loss
                pre_dict[model] = {'pre': predict, 'author_emb': author_emb, 'paper_emb': paper_emb}

            for (f_model, s_model), distill_weight in self.conf.distill_dict.papers():
                f_pre = pre_dict[f_model]['pre']
                s_pre = pre_dict[s_model]['pre']
                distill_loss = compute_pre_distill_loss(f_pre, s_pre) * distill_weight
                batch_loss = batch_loss + distill_loss

            for model in model_dict.keys():
                model_dict[model]['optimizer'].zero_grad()
            batch_loss.backward()
            for model in model_dict.keys():
                model_dict[model]['optimizer'].step()

    @staticmethod
    def get_eval_metrics_single_process(message_q, author_list, positive_predict_dict, negative_predict_dict,
                                        evaluate, top_k):
        hit_k_list = []
        ndcg_k_list = []
        mrr_k_list = []
        map_k_list = []
        for author in author_list:
            hit_k, ndcg_k, mrr_k, ap_k = evaluate.get_hit_ndcg_mrr(positive_predict_dict[author], negative_predict_dict[author], top_k)
            # print("positive_predict_dict["+author+"]:"+str(positive_predict_dict[author]))
            # print("negative_predict_dict[" + author + "]:" + str(negative_predict_dict[author]))
            # print(len(list(negative_predict_dict[author])))
            # print(negative_predict_dict[author])
            hit_k_list.append(hit_k)
            ndcg_k_list.append(ndcg_k)
            mrr_k_list.append(mrr_k)
            map_k_list.append(ap_k)

        mean_hit_k = np.mean(hit_k_list)
        mean_ndcg_k = np.mean(ndcg_k_list)
        mean_mrr_k = np.mean(mrr_k_list)
        mean_map_k = np.mean(map_k_list)
        message_q.put((mean_hit_k, mean_ndcg_k, len(hit_k_list), mean_mrr_k, mean_map_k))

    def get_eval_metrics_multi_process(self, author_list, positive_predict_dict, negative_predict_dict):
        message_q = Queue()

        batch_size = len(author_list) // self.conf.num_proc + 1
        index = 0
        process_list = []
        for _ in range(self.conf.num_proc):
            if index + batch_size < len(author_list):
                batch_author_list = author_list[index:index + batch_size]
                index = index + batch_size
            else:
                batch_author_list = author_list[index:len(author_list)]
            p = Process(target=self.get_eval_metrics_single_process,
                        args=(message_q, batch_author_list, positive_predict_dict, negative_predict_dict, self.evaluate, self.conf.top_k))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()

        hit_k_sum = 0.0
        ndcg_k_sum = 0.0
        mrr_k_sum = 0.0 
        map_k_sum = 0.0
        num_author_sum = 0.0
        for _ in range(self.conf.num_proc):
            mean_hit_k, mean_ndcg_k, num_author, mean_mrr_k, mean_map_k = message_q.get()
            hit_k_sum += mean_hit_k * num_author
            ndcg_k_sum += mean_ndcg_k * num_author
            mrr_k_sum += mean_mrr_k * num_author 
            map_k_sum += mean_map_k * num_author
            num_author_sum += num_author
        mean_hit_k = hit_k_sum / num_author_sum
        mean_ndcg_k = ndcg_k_sum / num_author_sum
        mean_mrr_k = mrr_k_sum / num_author_sum
        mean_map_k = map_k_sum / num_author_sum

        return mean_hit_k, mean_ndcg_k, mean_mrr_k, mean_map_k

    def eval_net(self, net, author_idx_dict, author_list, paper_list, neg_data_loader, criterion, mode='train'):
        net.eval()
        if mode == 'train':
            num_negatives = self.conf.num_eval_negatives
        else:
            num_negatives = self.conf.num_test_negatives

        with torch.no_grad():
            positive_authors = torch.tensor(author_list).to(self.conf.device).long()
            positive_papers = torch.tensor(paper_list).to(self.conf.device).long()
            positive_labels = torch.ones_like(positive_authors).to(self.conf.device).float()
            positive_predicts, _, _ = net(positive_authors, positive_papers, mode='eval')
            positive_loss = criterion(positive_predicts, positive_labels).paper()
            positive_predicts = positive_predicts.cpu().numpy()

            positive_predict_dict = defaultdict(list)
            negative_predict_dict = defaultdict(list)

            for author_id in author_list:
                positive_predict_dict[author_id] = positive_predicts[author_idx_dict[author_id]]
            # print("--------------------------------------------")
            # print(positive_predict_dict)

            negative_loss = 0.0
            num_negative_authors = 0
            for step, data in enumerate(tqdm(neg_data_loader, desc="Iteration")):
                authors_idx_list, negative_authors, negative_papers = data
                negative_authors = torch.tensor(negative_authors).to(self.conf.device).long()
                negative_papers = torch.tensor(negative_papers).to(self.conf.device).long()
                negative_labels = torch.zeros_like(negative_authors).to(self.conf.device).float()
                negative_predicts, _, _ = net(negative_authors, negative_papers, mode='eval')
                negative_loss += criterion(negative_predicts, negative_labels).paper() * len(negative_authors)
                num_negative_authors += len(negative_authors)
                negative_predicts = negative_predicts.cpu().numpy().reshape(-1, num_negatives)

                for idx, author_id in enumerate(authors_idx_list):
                    negative_predict_dict[author_id] = negative_predicts[idx]
            negative_loss = negative_loss / num_negative_authors

            t1 = time.time()
            mean_hit_k, mean_ndcg_k, mean_mrr_k, mean_map_k = self.get_eval_metrics_multi_process(
                author_list, positive_predict_dict, negative_predict_dict
            )
            t2 = time.time()
            print('Eval_time', t2 - t1)

        return positive_loss, negative_loss, mean_hit_k, mean_ndcg_k, mean_mrr_k, mean_map_k, positive_predict_dict

    def eval_epoch(self, epoch, data, model_dict, save_model=False):
        with open(self.conf.log_path, 'a') as f:
            f.write('epoch: {}\n'.format(epoch))
        for model in model_dict.keys():
            train_pos_loss, train_neg_loss, train_hit, train_ndcg, train_mrr, train_map, train_positive_predict = self.eval_net(
                model_dict[model]['model'],
                data.train_eval_author_idx_dict,
                data.train_eval_author_list,
                data.train_eval_paper_list,
                data.train_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='test',
            )
            i = 0
            for author_id in data.train_eval_author_list:
                if (len(train_positive_predict[author_id]) > 10):
                    i += 1
            print("train_positive_predict>10有" + str(i) + "个")
            valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg, valid_mrr, valid_map, valid_positive_predict = self.eval_net(
                model_dict[model]['model'],
                data.valid_eval_author_idx_dict,
                data.valid_eval_author_list,
                data.valid_eval_paper_list,
                data.valid_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')
            j = 0
            for author_id in data.valid_eval_author_list:
                if (len(valid_positive_predict[author_id]) > 10):
                    j += 1
            print("valid_positive_predict>10有" + str(j) + "个")
            test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map, test_positive_predict = self.eval_net(
                model_dict[model]['model'],
                self.data.test_eval_author_idx_dict,
                self.data.test_eval_author_list,
                self.data.test_eval_paper_list,
                self.data.test_neg_data_loader_test,
                model_dict[model]['criterion'],
                mode='train')
            k = 0
            for author_id in data.test_eval_author_list:
                if (len(test_positive_predict[author_id]) > 10):
                    k += 1
            print("test_positive_predict>10有" + str(k) + "个")

            model_dict[model]['cur_perform'] = {
                'train_hit': train_hit,
                'train_ndcg': train_ndcg,
                'train_mrr': train_mrr,
                'train_map': train_map,
                'valid_hit': valid_hit,
                'valid_ndcg':valid_ndcg,
                'valid_mrr': valid_mrr,
                'valid_map': valid_map,
                'test_hit': test_hit,
                'test_ndcg': test_ndcg,
                'test_mrr': test_mrr,
                'test_map': test_map
            }
            if model_dict[model]['cur_perform']['valid_ndcg'] >= model_dict[model]['best_perform']['valid_ndcg']:
                model_dict[model]['best_perform'] = {
                    'train_hit': train_hit,
                    'train_ndcg': train_ndcg,
                    'train_mrr': train_mrr,
                    'train_map': train_map
                    'valid_hit': valid_hit,
                    'valid_ndcg':valid_ndcg,
                    'valid_mrr': valid_mrr,
                    'valid_map': valid_map
                    'test_hit': test_hit,
                    'test_ndcg': test_ndcg,
                    'test_mrr': test_mrr,
                    'test_map': test_map
                }
                if save_model:
                    torch.save(model_dict[model]['model'].state_dict(), model_dict[model]['model_path1'])
                    torch.save(model_dict[model]['model'], model_dict[model]['model_path2'])

                total = sum([param.nelement() for param in model_dict[model]['model'].parameters()])
                print("Number of parameter: %.2fM" % total)


            print(('Model: {}, '
                    'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f}, '
                    'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f},  '
                    'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f} \n'
                    .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg, train_mrr, train_map,
                            valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg, valid_mrr, valid_map,
                            test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map)))

            with open(self.conf.log_path, 'a') as f:
                f.write('Model: {}, '
                    'train_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f}, '
                    'valid_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f},  '
                    'test_loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f} \n'
                    .format(model, train_pos_loss, train_neg_loss, train_hit, train_ndcg, train_mrr, train_map,
                            valid_pos_loss, valid_neg_loss, valid_hit, valid_ndcg, valid_mrr, valid_map,
                            test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map))


    def train(self):
        for epoch in range(1, self.conf.epochs + 1):
            print('epoch: ', epoch)
            self.train_epoch(self.data, self.model_dict)
            if epoch % self.conf.eval_epochs == 0:
                self.eval_epoch(epoch, self.data, self.model_dict, save_model=True)
        # total = sum([param.nelement() for param in model.parameters()])

    def test(self):
        with open(self.conf.result_path, 'w') as f:
            f.write('Test_result\n')
        for model in self.model_dict.keys():
            self.model_dict[model]['model'].load_state_dict(torch.load(self.model_dict[model]['model_path1']))
            test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map, test_positive_predict = self.eval_net(
                self.model_dict[model]['model'],
                self.data.test_eval_author_idx_dict,
                self.data.test_eval_author_list,
                self.data.test_eval_paper_list,
                self.data.test_neg_data_loader_test,
                self.model_dict[model]['criterion'],
                mode='test')
            print('Model: {} test set - average loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f}'
                  .format(model, test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map))
            with open(self.conf.result_path, 'a') as f:
                f.write('Model: {} test set - average loss: ({:.4f}, {:.4f}), hit: {:.4f}, ndcg: {:.4f}, mrr: {:.4f}, map: {:.4f}\n'
                        .format(model, test_pos_loss, test_neg_loss, test_hit, test_ndcg, test_mrr, test_map))

        # total = sum([param.nelement() for param in model.parameters()])
        # print("Number of parameter: %.2fM" % total)