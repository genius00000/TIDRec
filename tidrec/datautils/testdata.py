from tidrec.datautils.data_load import *
from tidrec.datautils.dataset import *
from tidrec.utils import *
from tidrec.datautils.Eval_dataset2 import *
from torch.utils.data import DataLoader


class TestData:
    def __init__(self, author_pair_set, ui_pair_dict):
        self.ui_pair_set = self.get_pair_set_from_pair_dict(ui_pair_dict)

        self.author_set = set()
        self.paper_set = set()
        for author, paper in self.ui_pair_set:
            self.author_set.add(author)
            self.paper_set.add(paper)
        print('Num authors', len(self.author_set))
        print('Num papers', len(self.paper_set))

        # author
        self.author_set = set(ui_pair_dict.keys())

        # paper
        self.iu_pair_set = self.reverse_pair_set(self.ui_pair_set)  # 反转训练集

        # ('author','co-write','author'), ('author','write','paper')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_graph = self.construct_hetero_graphs(
            [author_pair_set, self.ui_pair_set, self.iu_pair_set],
            [('author', 'friend', 'author'), ('author', 'like', 'paper'), ('paper', 'rev_like', 'author')],
            {'author': 8851, 'paper': 6037}
        ).to(device)

        self.train_ui_pair2eid = self.get_pair2eid(self.ui_pair_set)
        self.train_iu_pair2eid = self.get_pair2eid(self.iu_pair_set)

        # eval positive
        self.author_idx_dict, self.author_list, self.paper_list = \
            self.get_eval_positive_dataset(self.ui_pair_set)

        self.data_dict = Eval_dataset2(self.author_set, ui_pair_dict)

        self.neg_author_list, self.collected_author_list, self.collected_paper_list = self.collate_eval_neg_train(self.data_dict)



    def collate_eval_neg_train(self, samples):
        author_list, paper_list = map(list, zip(*samples))
        collected_author_list = []
        collected_paper_list = []
        for idx in range(len(author_list)):
            collected_author_list.extend([author_list[idx]] * 1000)
            for _ in range(1000):
                negative_paper_id = np.random.randint(6037)
                while (author_list[idx], negative_paper_id) in self.ui_pair_set:
                    negative_paper_id = np.random.randint(6037)
                collected_paper_list.append(negative_paper_id)
        return author_list, collected_author_list, collected_paper_list

    @staticmethod
    def read_co_data(co_data):
        author_pair_set = set()
        for row in range(co_data.shape[0]):
            u1_id, u2_id = int(co_data[row][0]) - 1, int(co_data[row][1]) - 1
            author_pair_set.add((u1_id, u2_id))
        return author_pair_set

    @staticmethod
    def read_write_data(write_data):
        ui_pair_dict = defaultdict(set)
        for row in range(write_data.shape[0]):
            write_paper = write_data[row]
            author_id, paper_id = int(write_paper[0]) - 1, int(write_paper[1]) - 1,
            category, score = int(write_paper[2]), int(write_paper[3])
            ui_pair_dict[author_id].add(paper_id)
        return ui_pair_dict

    # generate eval data
    @staticmethod
    def get_eval_positive_dataset(ui_pair_set):
        author_list = []
        paper_list = []
        author_idx_dict = defaultdict(list)
        for idx, (author_id, paper_id) in enumerate(ui_pair_set):
            author_list.append(author_id)
            paper_list.append(paper_id)
            author_idx_dict[author_id].append(idx)
        return author_idx_dict, author_list, paper_list

    @staticmethod
    def get_pair2eid(pair_list):
        pair2eid = {}
        for eid, pair in enumerate(pair_list):
            pair2eid[pair] = eid
        return pair2eid

    @staticmethod
    def get_pair_set_from_pair_dict(pair_dict):
        pair_set = set()
        for key, val in pair_dict.papers():
            for pair in val:
                pair_set.add((key, pair))
        return pair_set

    @staticmethod
    def reverse_pair_set(pair_set):
        rev_pair_set = set()
        for (id_1, id_2) in pair_set:
            rev_pair_set.add((id_2, id_1))
        return rev_pair_set

    @staticmethod
    def construct_hetero_graphs(pair_sets, edge_types, num_nodes_dict):
        data_dict = {}
        for idx, pair_set in enumerate(pair_sets):
            source_ids, target_ids = get_source_target_form_pair_set(pair_set)
            data_dict[edge_types[idx]] = (source_ids, target_ids)
        graph = dgl.heterograph(data_dict, num_nodes_dict)
        return graph